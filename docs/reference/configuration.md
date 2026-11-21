# Configuration System

All scripts use [Hydra](https://hydra.cc/) for configuration management. Configs are organized under `configs/` by pipeline domain.

## AxBench across-concept histograms

`configs/plot/axbench_histograms.yaml` configures the offline command
`python -m scripts.eval.plot_axbench_histograms`:

| Field | Default | Contract |
|---|---|---|
| `pipeline` | `upstream` | `upstream` or `adv` |
| `input_dir` | Required | Evaluated directory for upstream; completed schema-2 run root for Adv |
| `output_dir` | Required | New or empty directory, path-disjoint from the input |
| `method_factors` | `{}` | Upstream requires explicit method-to-factor entries (`'+method_factors={DiffMean:1,SelfConditioning:5}'`); Adv requires an empty map and reads recorded factors |
| `bins` | `10` | Integer 1–100; equal-width bins spanning 0–2, shared by every report |

Each observation is a concept's mean over ten prompts. No per-concept factor
selection, axis-specific optimization, or pooling across proposal counts is
performed. See [histogram usage and output contracts](../evaluation/axbench-upstream.md#across-concept-score-histograms)
and [Adv usage](../evaluation/axbench-adv.md#across-concept-score-histograms).

## Directory Layout

```
configs/
├── training/               # selected files — pretraining, SFT, RL, encoder-decoder
│   ├── _train_common.yaml               # Shared pretraining/SFT loop controls
│   ├── base_train.yaml                  # Pretraining extension of _train_common
│   ├── conditional_encdec_train.yaml    # Encoder-decoder pretraining
│   ├── conditional_encdec_sft.yaml      # Encoder-decoder SFT
│   ├── conditional_encdec_sft_grpo.yaml # Structured-SFT self-conditioning GRPO
│   ├── conditional_encdec_sft_grpo_pilot.yaml # G=256 pilot extension
│   ├── conditional_encdec_sft_grpo_chatcore.yaml # Short-answer benchmark geometry
│   ├── conditional_encdec_selfcond.yaml # On-policy self-conditioning post-training
│   ├── conditional_encdec_sft_profile_4090.yaml  # Short 1x4090 d8 SFT profiling run
│   ├── conditional_encdec_4090.yaml     # RTX 4090 hardware variant
│   ├── conditional_encdec_4xh200.yaml   # 4xH200 variant
│   ├── conditional_encdec_2xh200.yaml   # 2xH200 variant
│   ├── conditional_encdec_2xb200.yaml   # 2xB200 variant
│   ├── feature_selection_variant/       # Positive-selection sum-type arms
│   ├── conditional_encdec_sweep.yaml    # Encoder-decoder sweep config
│   ├── chat_sft.yaml                    # Supervised fine-tuning
│   ├── chat_sft_huggingface.yaml        # Shared native-Transformers SFT defaults
│   ├── chat_sft_smollm3_stage1_step_40000.yaml # Pinned SmolLM3 source
│   ├── chat_sft_pythia_2_8b.yaml        # Pinned final Pythia 2.8B source
│   ├── chat_sft_pythia_1_4b.yaml        # Pinned final Pythia 1.4B source
│   ├── chat_sft_smollm3_3b_h200_4x8_benchmark.yaml # W32 H200 canary
│   ├── chat_sft_pythia_2_8b_h200_4x8_benchmark.yaml # W32 H200 canary
│   ├── chat_sft_pythia_1_4b_h200_4x8_benchmark.yaml # W32 H200 canary
│   ├── chat_sft_capabilities.yaml       # Capabilities mixture overrides
│   ├── chat_sft_safety.yaml             # Safety mixture overrides
│   ├── chat_sft_4xrtx6000.yaml         # RTX PRO 6000 variant
│   ├── chat_rl.yaml                     # GRPO reinforcement learning
│   └── count_tokens.yaml               # Token counting utility
├── extraction/             # SAE feature extraction
│   ├── extract_features.yaml            # Main extraction config
│   ├── extract_features_4xh200.yaml     # 4xH200 variant
│   ├── extract_features_2xb200.yaml     # 2xB200 variant
│   ├── extract_features_4xb200.yaml     # 4xB200 variant
│   ├── extract_features_8xpro6000.yaml  # 8xPRO 6000 variant
│   ├── extract_features_gemma3.yaml     # Gemma 3 (default: 4b-it, l22, 16k)
│   ├── extract_features_gemma3_l22_f65k.yaml   # Gemma 3 4b-it l22 65k pipeline
│   ├── compute_statistics.yaml          # Feature statistics
│   ├── compute_statistics_gemma3.yaml   # Gemma 3 statistics
│   ├── compute_statistics_gemma3_l22_f65k.yaml  # Gemma 3 l22 65k exact-count reducer
│   ├── validate_features.yaml           # Feature shard validation
│   ├── validate_features_gemma3_l22_f65k.yaml   # Gemma 3 l22 65k full validation
│   ├── precompute_percentiles.yaml      # Percentile precomputation
│   ├── precompute_percentiles_gemma3.yaml
│   ├── precompute_percentiles_gemma3_l22_f65k.yaml
│   ├── precompute_token_boundaries_gemma3_l22_f65k.yaml
│   ├── precompute_packed_rows_gemma3_l22_f65k.yaml
│   ├── splice_feature_shards.yaml       # Post-extraction feature-shard splice
│   ├── precompute_autointerp_filter.yaml
│   ├── precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos.yaml
│   ├── compute_manual_filter_coverage.yaml
│   ├── sample_segments.yaml             # Segment sampling
│   ├── plot_features.yaml               # Feature visualization
│   └── experiment.yaml                  # Experiment template
├── eval/                   # selected files — evaluation
│   ├── base_eval.yaml                   # CORE + BPB + sampling
│   ├── chat_eval.yaml                   # Chat task evaluation
│   ├── harmfulness_eval.yaml            # Harmfulness/compliance evaluation (OpenRouter judge)
│   ├── harmfulness_eval_l22.yaml        # Phase-A compact L22 harmfulness preset
│   ├── build_harmfulness_context_cache_l22.yaml # Exact L22 1,200-prompt cache preset
│   ├── dclm_eval.yaml                  # DCLM CORE evaluation
│   ├── steering_eval.yaml              # SAE feature steering evaluation
│   ├── completion_steering_eval.yaml    # Completion steering evaluation
│   ├── shuffled_conditioning_eval.yaml  # Shuffled conditioning evaluation
│   ├── extract_common_features.yaml     # Sample k and extract absolute/delta conditioning features
│   ├── axbench_upstream/methods.yaml     # Corrected SelfConditioning comparison
│   └── plot_evals.yaml                 # Eval comparison charts
├── serving/                # 5 files — inference/deployment
│   ├── chat_cli.yaml                    # CLI chat
│   ├── chat_web.yaml                    # Web chat server
│   ├── steer_chat_cli.yaml              # Feature-conditioned CLI chat
│   ├── completion_tui.yaml              # Completion TUI
│   └── convert_decoder_to_encdec.yaml   # Checkpoint conversion
├── profiling/              # selected files — performance analysis
│   ├── benchmark_activations.yaml       # Activation benchmark
│   ├── benchmark_native_compile.yaml    # Native collector compile benchmark
│   ├── chatcore.yaml                    # Paired standalone ChatCORE timing and grading
│   ├── encdec_l22_training.yaml         # Real-data/model + topology profile
│   ├── encdec_l22_training_3b_h200.yaml # Planned 3B/H200 topology + measured preset
│   ├── nccl_transport_preflight.yaml     # Bounded allocation transport gate
│   └── weight_transform.yaml            # TransformerLens weight transform
├── data/                   # 4 files — data processing
│   ├── tok_train.yaml                   # Tokenizer training
│   ├── gen_synthetic_data.yaml          # Synthetic data generation
│   ├── convert_sft_to_parquet.yaml      # SFT → parquet conversion
│   └── mix_sft_into_pretraining.yaml    # Mix SFT into pretraining shards
└── local/                  # Git-ignored local overrides
    └── extract_features.yaml            # Local batch size overrides
```

## Hydra Patterns

### Two Invocation Styles

**Compose API** (for torchrun compatibility):
```python
# Used in scripts that support DDP via torchrun
with hydra.initialize(config_path="../configs/training"):
    cfg = hydra.compose(config_name="base_train", overrides=sys.argv[1:])
```

Invoked with `--` separator for Hydra args:
```bash
torchrun --nproc_per_node=8 -m scripts.train.base_train -- depth=12 fp8=true
```

**@hydra.main decorator** (for single-process scripts):
```python
@hydra.main(config_path="../configs/extraction", config_name="extract_features")
def main(cfg): ...
```

Invoked directly:
```bash
python -m steering.extract_features num_shards=10
```

### Config Inheritance

Configs inherit via `defaults:`:

```yaml
# conditional_encdec_train.yaml inherits ALL params from base_train.yaml
defaults:
  - base_train
```

Hardware-specific configs override only batch sizes:
```yaml
# conditional_encdec_4xh200.yaml
defaults:
  - conditional_encdec_train

device_batch_size: 16
fp8: true
```

### Optional Local Overrides

```yaml
defaults:
  - _self_
  - optional local@_global_: extract_features
```

Create `configs/local/extract_features.yaml` (git-ignored) to override params without modifying tracked files:
```yaml
# configs/local/extract_features.yaml
gpu_batch_size: 32
cpu_batch_size: 128
```

### Package Mapping

For complex inheritance across directories:
```yaml
# profiling configs can inherit production configs as globals
defaults:
  - /training/conditional_encdec_4090@_global_
  - _self_
```

## CLI Override Syntax

```bash
# Simple override
python -m scripts.train.base_train depth=12

# Nested config
python -m scripts.train.conditional_encdec_train encoder.n_embd=0.75 encoder.depth=3

# Boolean
python -m scripts.train.base_train fp8=true

# Null
python -m scripts.train.base_train model_tag=null

# Dict (for feature_conditions in steer_chat_cli)
python -m scripts.chat.steer_chat_cli 'feature_conditions={0: 1, 5: 2}'

# Feature configs (file/directory for base_eval, dclm_eval, chat_eval)
python -m scripts.eval.base_eval source=encdec feature_configs=out/data/common_features/safety_sft/

# Differential feature config for safety SFT vs pretraining base
python -m scripts.preprocess.extract_common_features \
  feature_dir=out/data/features/.../safety_sft \
  baseline_dir=out/data/features/.../base \
  top_k=64

# Use different config file
python -m scripts.train.conditional_encdec_train config_name=conditional_encdec_4090

# Multiple overrides
torchrun --nproc_per_node=8 -m scripts.train.base_train -- depth=26 fp8=true device_batch_size=16
```

## All Config Parameters

### Encoder-decoder l22 profiler (`configs/profiling/encdec_l22_training.yaml`)

| Parameter | Default | Description |
|---|---|---|
| `mode` | `topology` | `data` for local rank-like CPU processes, `train` for one exact-checkpoint CUDA process, `distributed` for measured synthetic training in one CUDA process or under multi-process CUDA/NCCL, or `topology` for the analytical distributed-optimizer sensitivity model. |
| `artifacts.{packed_shard,selection_shard,feature_mask,feature_clusters}` | `null` | Explicit local artifact paths required by `data` and `train`; topology mode does not read artifacts. |
| `data.workers` / `data.torch_threads` | `[1,2,4,8]` / `1` | Process-count sweep and intra-op threads per process. |
| `data.batch_size` / `data.batches` / `data.timeout_seconds` | `16` / `3` / `300` | Rank-local rows per reconstructed batch, batches measured after each body decode, and worker-result timeout. |
| `train.checkpoint_dir` / `train.step` | `null` / `-1` | Exact checkpoint directory and required non-negative step. `NANOCHAT_BASE_DIR` still selects the tokenizer cache. |
| `train.batch_size` / `train.warmup_steps` / `train.steps` | `1` / `3` / `10` | Single-GPU batch shape and timing sample counts. |
| `train.compile` / `train.profile_kernels` / `train.kernel_rows` | `false` / `false` / `20` | Compile the model, capture one additional CPU/CUDA profiler step, and bound its table. |
| `distributed.model` | `null` | Explicit decoder/encoder geometry for a synthetic measured run. `null` preserves the audited 412M c1024 model; the 3B H200 preset binds this to its d27/e27, width-1,664, c2048 topology geometry. |
| `distributed.device_batch_size` / `distributed.total_batch_size` | `16` / `524288` | Exact rank-local padded shape and fixed global token batch. The latter must be divisible by `device_batch_size * sequence_length * WORLD_SIZE`. The four-node 3B H200 preset uses rank-local batch `16` and the trainer's derived 1,048,576-token global batch, giving one world-size-32 microbatch per optimizer step. |
| `distributed.padded_segments` / `distributed.padded_features` | `13` / `96` | Static synthetic conditioning extents. The 3B c2048 preset uses 26 segment slots because no production c2048 packed manifest exists yet. |
| `distributed.fp8` / `distributed.fp8_recipe` | `false` / `tensorwise` | Apply the trainer's FP8 conversion before compilation and optimizer construction. |
| `distributed.muon_parameter_packing` | `false` | Select the production persistent Muon storage adapter. The result records the actual optimizer class and flag; serialized optimizer diagnostics must remain disabled. See the SFT packing lifecycle below. |
| `distributed.warmup_steps` / `distributed.steps` / `distributed.compile` | `2` / `10` / `true` | Full optimizer-step warmups, measured samples, and compile setting for the torchrun mode. |
| `distributed.compile_static` | `true` | Compile with `dynamic=False` when true, `dynamic=True` when false, or SFT's `dynamic=None` policy with `auto`. Avoid false with Hopper FA4, where symbolizing decoder masks is unsupported. |
| `distributed.validation_warmup_steps` / `distributed.validation_steps` | `1` / `0` | Optional eager BF16 no-grad conditioned-BPB forward probe after training, matching the trainer's `orig_model` evaluation call with optimizer state still resident. `validation_steps=0` preserves the existing training-only run; a positive value reports warmup and steady-state CUDA peaks separately. |
| `distributed.trial_id` / `distributed.optimizer_diagnostic_steps` | `null` / `0` | Optional result identity and additional serialized optimizer diagnostics. Diagnostics synchronize every parameter group and phase, so they attribute reduce/update-gather/copy costs but are not production-throughput samples. Positive diagnostic steps are rejected with parameter packing because this diagnostic bypasses the optimizer's step lifecycle. |
| `distributed.component_diagnostic_steps` / `distributed.component_warmup_steps` / `distributed.component_kernel_rows` | `0` / `1` / `20` | Separate eager module attribution, its warmups, and the operator-table row limit. Includes an adjacent uninstrumented eager control; neither supplies the compiled throughput result. Packed encoder/cross-attention hook entries are marked unmeasured. |
| `distributed.trace_dir` | `null` | Optional directory for per-rank forward/backward/optimizer Chrome traces when component diagnostics are enabled. Trace steps follow ordinary timing samples. |
| `distributed.static_max_seg_size` | `1024` | Audited packed-manifest segment bound; used for integrity checks, not CUDA cross-attention allocation size. |
| `distributed.avg_features_per_segment` / `distributed.segment_size` | `24.0` / `128` | Inputs to the model FLOP estimator used for MFU; `segment_size` also constructs the synthetic segment map. |
| `topology.model.{decoder_depth,encoder_depth,width,head_dim,sequence_length,window_pattern}` | `12` / `12` / `768` / `128` / `1024` / `SSSL` | Model used for optimizer inventory and batch-contract analysis. The `encdec_l22_training_3b_h200` preset overrides this block for d27/e27, width 1,664, context 2,048 and also binds `distributed.model` to it. |
| `topology.model.{feature_vocab_size,feature_prediction_block,feature_prediction_head_layers}` | `2074` / `mlp` / `0` | Compact feature vocabulary, residual head block (`mlp` or `transformer`), and head depth for optimizer inventory and for measured runs that bind `distributed.model` to this geometry. |
| `topology.world_sizes` / `topology.gpus_per_node` | `[8,64]` / `8` | Modeled rank counts and node width. The 3B H200 preset adds world size 32. Every world size must be divisible by the resolved node width. |
| `topology.intra_latency_us` / `topology.intra_bandwidth_gb_s` | `2` / `450` | Ideal hierarchical-ring within-node assumptions. |
| `topology.inter_latency_us` / `topology.inter_bandwidth_gb_s` | `[5,25,100]` / `50` | Cross-node latency sweep and bandwidth assumption. These produce modeled, not measured, times. |
| `topology.device_batch_size` / `topology.total_batch_size` | `16` / `524288` | Validate the trainer's integer gradient-accumulation contract at each modeled world size. The 3B H200 preset uses B=16 and the derived 1,048,576-token global batch for its world-size-32 main run; its retained world-size-64 inventory intentionally reports an invalid batch contract unless B is overridden to `8`. |

### NCCL transport preflight (`configs/profiling/nccl_transport_preflight.yaml`)

This config is consumed under `torchrun`; `NCCL_TIMEOUT_SEC` is a required
environment variable so process-group initialization and collectives are
bounded. The Slurm wrapper adds an outer launcher timeout and a single
host-staged retry.

| Parameter | Default | Description |
|---|---:|---|
| `transport.dtype` | `bfloat16` | Collective tensor dtype; supported values are `bfloat16` and `float32`. |
| `transport.shard_mib` | `[1,43]` | Per-rank sizes for all-reduce and for each reduce-scatter/all-gather shard. At world size 64, 43 MiB produces a 2.688 GiB stacked buffer. |
| `transport.warmup_steps` / `transport.steps` | `1` / `3` | Untimed warmups and timed samples for each operation and size. |
| `transport.max_buffer_gib` | `4` | Fail-before-allocation guard on the largest world-sized reduce-scatter/all-gather buffer. |

### Natural-chat negative AxBench (`configs/eval/axbench_negative/`)

The `prepare.yaml` profile pins UltraChat's `test_sft` revision and file hash;
`row_count` selects the first original rows before filtering, and `snapshot_dir`
can name a local pinned snapshot. `build.yaml` requires source/metadata hashes,
concept IDs, scoring renderer and explicit API models. `target_admission` pins
each target's model/metadata/tokenizer and checks complete prompt plus generation
budget against every context. `reserved_extraction_tokens` reserves separate
Gemma-token headroom; actual answers still receive full-length validation.

`producer.n_verification_seeds` and `min_consistency_passes` define fixed k/m
screening (defaults 5/3). `max_candidates_per_feature` caps active candidates in
source order; `avoid_max_attempts` bounds only private avoiding synthesis. Neutral
misses receive no replacements. `neutral`, `avoiding` and `quality` each declare
provider, model, temperature, top-p, token budget, optional master seed and transport
limits. The new Hydra profiles default to `provider=openai` with `OPENAI_API_KEY`;
`avoiding.provider` inherits `neutral.provider`. Explicit `provider=openrouter`
uses only `OPENROUTER_API_KEY`. Routes are fixed and credentials never fall back
between providers. Programmatic API configs omitting provider retain the legacy
OpenRouter default. Evaluation's `judge.provider` independently defaults to OpenAI.
Zero `transport_retry_budget_seconds` leaves the retry count as the only retry
allowance; the shared transport otherwise permits retries while either allowance
remains. Observed model/SAE/tokenizer content hashes bind the scoring contract;
optional expected hashes enforce prior pins.

`eval.yaml` requires a completed source, the same `extractor` mapping, independent
canonical TRAIN hashes, and an admitted target. Native methods are DiffMean and
LinearProbe; encoder-decoder uses SelfConditioning. Magnitudes are nonnegative
suppression strengths, with a separate fresh control. `replicates`, `seed`,
`max_new_tokens` and split define the complete evaluation population; `max_cells`
only limits resumable work. See [the evaluation guide](../evaluation/axbench-negative.md)
for commands, statuses, identities and interpretation.

### AxBench project input (`configs/eval/axbench_upstream/project_input.yaml`)

This model-free producer recreates the upstream AlpacaEval input grid from the
pinned prepared data without loading or running Qwen (or any generation
model). It verifies the exact vendored dataset revisions plus the prepared
manifest's byte counts and SHA-256 values before and after construction.
Missing and unknown keys are rejected.

| Key | Default | Meaning |
|---|---|---|
| `prepared_dir` | `out/axbench/prepared` | Pinned prepare manifest, metadata JSONL, and AlpacaEval JSON root |
| `output_path` | `out/axbench/runs/c5/project_input/steering_input.parquet` | Strict input-only Parquet with no method-result columns |
| `manifest_path` | `out/axbench/runs/c5/project_input/manifest.json` | Manifest-last prepared-manifest/output digests and explicit no-model provenance; consumers require and bind this companion artifact |
| `max_concepts` | `5` | Contiguous metadata concepts; each has exactly ten sampled prompts and the complete factor grid |
| `force` | `false` | Permit atomic replacement of an existing Parquet/manifest pair |

### Vendored AxBench project methods (`configs/eval/axbench_upstream/methods.yaml`)

This thin Hydra config runs corrected SelfConditioning over exactly one input
source. Missing or unknown top-level keys are rejected.

| Key | Default | Meaning |
|---|---|---|
| `main_path` | `out/axbench/runs/c2/main/inference/steering_data.parquet` | Main baseline input; schema/key/input invariants are required, while discarded baseline result values are not health-gated |
| `project_input_path` | `null` | Model-free input-only alternative produced by `build_project_input`; set `main_path=null` when enabled. Exactly one source must be configured. |
| `project_input_manifest_path` | `null` | Required companion producer manifest when `project_input_path` is enabled; its format, no-model fields, source/output hashes, geometry, and prepared identity are revalidated before model work |
| `nl_target_source_manifest_path` | `null` | Optional strict schema-1 receipt for the description-embedding and concept-embedding NPZ snapshots. A non-null value selects the three-project-method `NLSteering` product; relative artifact paths resolve from the receipt directory. |
| `prepared_dir` | `out/axbench/prepared` | Prepared metadata and concept-training Parquet root |
| `output_path` | `out/axbench/runs/c2/project_v4/inference/steering_data.parquet` | Project-method result frame |
| `manifest_path` | `out/axbench/runs/c2/project_v4/run_manifest.json` | Schema-3 legacy-main or schema-4 model-free-source run provenance JSON with the rank-pool contract and exact factor-limit table |
| `target_meta_path` | selected L22 clustered `meta_054600.json` path | Encoder-decoder checkpoint metadata |
| `feature_statistics_path` | configured L22/F65k statistics path | Percentile table source matching the checkpoint |
| `max_concepts` | `2` | Positive concept count; must match the main frame |
| `device` | `cuda` | Extraction and target-model device |
| `seed` | `42` | Base seed for factor-independent feature selection and factor-specific token generation |
| `max_new_tokens` | `128` | Positive output-token cap; prompt plus output must fit the decoder sequence length |
| `extraction_batch_size` | `8` | Dense DiffMean concept-fit batch size |
| `self_conditioning_policy` | `prompt_last_segment_max_pool_logit_rank_diffmean_lerp_v2` | Required, explicit inference policy |
| `self_conditioning_masked_policy` | `prompt_last_segment_max_pool_logit_rank_diffmean_sign_rank_v3` | Required, explicit sign-consistent stable-DiffMean-rank pool policy |
| `force` | `false` | Permit replacing existing project output and manifest |

`configs/eval/axbench_upstream/methods_nl.yaml` inherits this default config and
supplies the receipt plus separate output paths. The receipt binds each NPZ by
SHA-256, the canonical source-text population, and one exact embedding-space
identity. The two NPZ files and receipt must be distinct. Missing, extra,
malformed, non-finite, or mismatched inputs fail before target-model loading or
resume-store creation.

Generation is fixed at temperature `1.0` and top-k `50`, matching both frozen
vendored sweep profiles. Those values intentionally are not config keys.
Prompt features come from the checkpoint's nested conditioning block and are
held only in memory. Strength 0 ranks pooled FP logits directly. Interior
strengths rank `lerp(2*sigmoid(logits)-1, dense_diffmean, strength/6)`, and
strength 6 ranks pure dense DiffMean; sampled positive and negative
cardinalities are shared across factors. The complete continuation uses one
predicted slot. `SelfConditioningMasked` has no scale knob: factor and each
checkpoint-configured side mean map the eligible-head-row count to independent
ties-to-even rank-pool limits. Factor zero removes the rank cap while retaining
sign and feature-mask eligibility. Rejected final frames are written atomically to a derived
`.rejected.parquet` diagnostic without replacing canonical files. See
[`docs/evaluation/axbench-upstream.md`](../evaluation/axbench-upstream.md).

### AxBench encoder-decoder baselines (`configs/eval/axbench_upstream/encdec_baselines.yaml`)

This thin Hydra config fits and runs EncDecDiffMean, EncDecLinearProbe,
EncDecPromptSteering, and EncDecLoRA after vendored main and auxiliary
steering. Missing or unknown top-level keys are rejected.

| Key | Default | Meaning |
|---|---|---|
| `main_path` | accepted `05f03dede-c2-retry-20260811` main steering Parquet | Full-grid input and stable prompt/schema authority; requires the producer `run_manifest.json` two parents above the Parquet |
| `aux_path` | accepted `05f03dede-c2-retry-20260811` auxiliary steering Parquet | Factor-1 PromptSteering text authority; requires the producer `run_manifest.json` two parents above the Parquet |
| `prepared_dir` | `out/axbench/prepared` | Prepared metadata and positive/same-genre-negative fit pairs |
| `output_main_path` | `out/axbench/runs/c2/encdec_v1/main/inference/steering_data.parquet` | EncDecDiffMean/EncDecLinearProbe full-grid result frame |
| `output_aux_path` | `out/axbench/runs/c2/encdec_v1/aux/inference/steering_data.parquet` | EncDecPromptSteering/EncDecLoRA factor-1 result frame |
| `manifest_path` | `out/axbench/runs/c2/encdec_v1/run_manifest.json` | Run provenance JSON, written after both Parquets |
| `train_dir` | `out/axbench/runs/c2/encdec_v1/train` | Exact resumable per-concept direction/adapter artifacts; unknown files or artifacts without the matching cell-store identity fail |
| `target_meta_path` | selected L22 clustered `meta_054600.json` path | Encoder-decoder checkpoint metadata |
| `feature_statistics_path` | configured L22/F65k statistics path | Statistics source matching the checkpoint |
| `max_concepts` | `2` | Positive concept count; must match prepared and source-frame coverage |
| `device` | `cuda` | Extraction and target-model device |
| `seed` | `42` | Base seed for fit shuffles, factor-independent selection, and per-cell token generation |
| `max_new_tokens` | `128` | Positive output-token cap; prompt plus output must fit the decoder sequence length |
| `force` | `false` | Permit replacing existing canonical Parquets and manifest; never bypass resume/train-artifact identity checks |

Generation is fixed at temperature `1.0` and top-k `50`; layer 9 direction
steering, LoRA layers `[2, 5, 7, 9]`, and fit/training hyperparameters are
module constants rather than Hydra knobs. See
[`docs/evaluation/axbench-upstream.md`](../evaluation/axbench-upstream.md#run-the-encoder-decoder-baselines).

Prompt recovery reads model and revision provenance from both source manifests,
requires main and auxiliary identities to agree, and selects only the exact
registered Qwen2.5 wrapper or `HuggingFaceTB/SmolLM3-3B-checkpoints@it-soup-APO`
wrapper. Prefix, suffix, original-prompt tail, and nonblank steering text are
validated exactly; unknown identities and malformed wrappers fail before model
loading. Both source-manifest SHA-256 digests enter resume and output provenance.

Compatible same-prompt EncDecDiffMean/EncDecLinearProbe cells are automatically
decoded at physical batch size greater than one after independent batch-one
prefills. The optimization is bounded to continuations that cannot cross a
self-conditioning segment boundary. It adds no Hydra field: selection/token
seeds, cell identity, rank ownership, resume semantics, canonical row order,
and output schemas remain the configuration contract. Singleton resume
remainders, LoRA, PromptSteering, and selected-cell traces use the scalar path.

These configs intentionally add no rank, world-size, resume, or profiler knob.
Direct and torchrun execution use the same semantic config; placement is
operational, resume state lives beside `manifest_path`, and profiling is
activated only by `scripts.benchmark.benchmark_axbench_upstream profile`.
Low-overhead complete-run sampling uses the separate `throughput` subcommand;
it also adds no semantic config and requires fresh producer/output paths for
every sample so resume work is never counted as throughput.

### Vendored AxBench judge launcher

`python -m scripts.eval.axbench_upstream.run_evaluation` invokes
`vendor/axbench/axbench/sweep/nano_gpt_pretrain_steer/evaluate.yaml` through the
vendored environment. That config fixes the ten merged methods,
`LMJudgeEvaluator`, eight ordinary AxBench process-pool workers, no reporting backend, and
`openai/gpt-5.6-luna`.

| Flag | Default | Meaning |
|---|---|---|
| `--input` | required | Merged ten-method `steering_data.parquet` in canonical `(concept_id, input_id, factor)` order |
| `--output-dir` | required | New or empty directory for evaluated JSONL, Parquet, and plots |
| `--cache` | required | Shared SQLite cache outside `--output-dir`, keyed by exact response-affecting request content |
| `--cache-only` | `false` | Construct no provider client and fail on any cache miss |
| `--api-base-url` | `https://openrouter.ai/api/v1` | OpenAI-compatible provider base URL |

Live mode requires `OPENROUTER_API_KEY`. The launcher passes the cache and
endpoint to the vendored evaluator with native underscore flags; callers
should use the hyphenated root-level flags above. See
[`docs/evaluation/axbench-upstream.md`](../evaluation/axbench-upstream.md#judge-and-leaderboard-outputs).

`run_baseline_project_evaluation` retains its six-method default. Passing
`--include-nl-steering` selects the dedicated
`baseline_project_nl_evaluate.yaml` and requires the exact seven-method frame:
the four vendored baselines followed by `SelfConditioning`,
`SelfConditioningMasked`, and `NLSteering`.

### AxBench-Adv pressures (`configs/eval/axbench_adv/`)

The base profile defines a closed two-concept, six-method experiment. Required
`source` paths are `prepared_dir`, `main_parquet`, `aux_parquet`,
`evaluated_jsonl`, `project_parquet`, `target_meta`, and
`feature_statistics`. `source.target_receipt` is optional: null selects the
legacy `remote_complete_<step>.json` beside `target_meta`, while an explicit
path may name an `axbench_adv_evaluation_target` receipt containing exactly the
metadata and model identities. The evaluation receipt supports inference-only
archives without inventing optimizer payloads. Its step and remote-run
fingerprint must match the operative metadata, and its canonical aggregate,
file names, byte lengths, content hashes, and Xet hashes are fail-closed.
Its six configured hashes cover the historical main and
auxiliary Qwen sources, accepted six-method evaluation, two-method project
Parquet and manifest, and target checkpoint receipt. The project manifest and receipt transitively bind the
target metadata/model and statistics; those three runtime files are rehashed
before publication too.

`source.nl_attachment` is normally `null`. It accepts `paths` containing
`project_parquet`, `project_manifest`, and `evaluated_jsonl`, plus
`expectations.max_concepts` for the completed source population (100 for the
c50 campaign). Only NL columns, judged ratings and retrieval inputs are consumed;
other methods and producer-history metadata are not requirements. Source files
must remain unchanged during the run. Their hashes and the actual runtime
tokenizer identity are recorded once. Remove old receipt/resume/original-meta
paths, supplied attachment hashes, and producer commit/lock expectations.
It is mutually exclusive with `source.expectations.include_nl_steering`;
default `null` preserves the historical six-source behavior.

`source.expectations.project_target_step` (default `54600`) and
`source.expectations.project_target_model_tag` (default
`l22_main_h100_8rank_db16_v1`) pin the project checkpoint separately from the
baseline `target_model`. The project manifest, metadata filename and contents,
and receipt must agree with these pins during validation and revalidation.

`output_dir`, `judge_cache`, `reflection_cache`, and `reflection.model` are
required. Output must be fresh, both caches must be external and distinct, and
there is no resume mode. `cache_only=true` forbids live misses. The master and
GEPA seeds default to `42`; `device` defaults to `cuda:0`; the target generation
length is source-pinned at 128. `factor_overrides` accepts only the configured
methods and is an explicit diagnostic override of the historical concept/fluency
oracle. With a validated attachment, NLSteering's override is resolved against
the attachment's first-50 judged population.

`project_judge_inflight` defaults to `1` (serial) and accepts `1`, `2`, or `3`.
At `2`, the three project encoder-decoder methods generate on their existing
single GPU thread while one already-generated response is judged by the
default single-request worker. At most two immutable responses are in flight;
the worker's concept and fluency calls, cache, retry policy, and API concurrency
are unchanged. Results are collected in input order before any point is
committed. This operational setting does not change the counter science or the
sealed snapshot resume identity for the two supported SelfConditioning methods;
it does not qualify NLSteering partial resume. GEPA's one-input evaluations
gain no overlap.
At `3`, exactly two judge workers are required. The third slot lets GPU
generation queue one response while both judges are busy; only two responses
are judged concurrently. Results still commit in input order.

`project_judge_pair_concurrency` defaults to `1` and accepts only `1` or `2`.
At `2`, the single project judge worker sends the independent concept and
fluency requests for one generated response concurrently. It waits for both
requests, records successful cache writes even if the other request fails, and
retries the existing two-call unit under its existing policy. At most two
provider requests are active; their prompts, model, cache keys, rating parser,
and output order are unchanged. This setting applies only to the three project
encoder-decoder methods, independently of `project_judge_inflight`. It is
recorded in the manifest when enabled. Higher request concurrency can change
rate-limit behavior, so qualify provider capacity before a live rollout.

`project_judge_workers` defaults to `1` and accepts only `1` or `2`. The
two-worker option requires `project_judge_inflight: 2` or `3`. It starts two separate
framed judge workers for each project concept cell, each with its own SQLite
connection to the same judge cache. At most two generated responses can be
judged concurrently, with at most one more queued at in-flight depth `3`;
duplicate generated text is judged in sequence so the
second response sees the first response's cache writes. With
`project_judge_pair_concurrency: 2`, up to four provider requests can be active.
The default one-worker path retains its existing behavior. The enabled value
is recorded in the manifest. Qualify provider capacity before a live rollout.

`project_reuse_judge_workers` defaults to `false`. When enabled, the project
judge processes start during runtime initialization and stay alive across
sequential method and concept cells. The first pending cell's validated
identity is used for this startup; no judge request is sent until its text is
generated. Each cell rebind closes the prior per-cell judge and creates a fresh one from the
new validated identity and cache settings, then the parent verifies that
identity. Per-cell SQLite connections and accounting are reset, while the
number of workers and provider-request bound stay fixed. Any rebind or cell
failure aborts the pool; normal runtime cleanup closes it. The enabled value
is recorded in the manifest. Use it only for a fresh, qualified project run.

`project_reuse_single_token_mask` defaults to `false`. For the three project
self-conditioned methods, `true` reuses the packed cross-attention routing
mask while one-token decoding remains in the same encoder layout and target
slot. A conditioning refresh builds a new mask; the cache is cleared when the
generation call ends, including on failure. Other AxBench-Adv methods and
training forwards keep their existing path. The enabled setting is recorded
in the manifest.

`project_bf16_decoder_linears` defaults to `false`. For a fresh project
counter-only run on CUDA with `NANOCHAT_DTYPE=bfloat16`, `true` converts only
the target decoder blocks' `Linear` weights to BF16 once after checkpoint
loading. Those layers otherwise cast the same FP32 weights on every token.
The checkpoint file and encoder, cross-attention, LM-head, and feature-head
weights are unchanged. GEPA, GCG, soft-prompt pressures and non-BF16 or CPU execution are
rejected. The enabled setting is recorded in the manifest.

`project_reuse_diffmean_texts` defaults to `false`. For the three project
encoder-decoder methods, `true` reuses the max-pooled SAE activation for an
identical training string during one dense DiffMean startup fit. The validated
source must set `extraction_batch_size: 1`; other batch sizes fail before the
extraction stack loads. Only texts that recur are cached, and the enabled
setting is recorded in the manifest. The source text order and percentile
reduction remain unchanged.

`project_weight_free_extraction` defaults to `false`. For the project Gemma-3
nnterp runtime, `true` reads the TransformerLens model config and HuggingFace
tokenizer without loading the unused TransformerLens weights. It still loads
the same SAE and HuggingFace nnterp decoder. The config retains TransformerLens's
8,192-token context cap for Gemma-3-4B; the HuggingFace config advertises a
larger limit and cannot replace that cap. The enabled setting is recorded in
the manifest. Use it only for a fresh, qualified run.

`project_fit_selected_concepts_only` defaults to `false`. When enabled, the
project runtime fits dense DiffMean directions only for the concept IDs owned
by this execution slice. Source validation and prompt preparation still cover
the complete source. The enabled setting is recorded in the manifest. Use it
only for a fresh, qualified sliced run; it does not change an existing run's
resume identity or migrate completed points.

`project_native_extraction` defaults to `false`. With
`project_weight_free_extraction=true`, CUDA, and counter-only pressures, it
uses the truncated native Gemma-3 residual collector for prompt preparation
and dense DiffMean fitting instead of the nnterp tracer. The resolved SAE must
be at the matching `hook_resid_post` layer. The enabled setting is recorded in
the manifest. Use it only for a fresh, qualified project counter run.

`reuse_final_judge_worker` defaults to `false`. When enabled, the final
AxBench evaluator handles successive points in one isolated Python 3.12
process rather than importing the evaluator afresh for each point. Points are
still judged sequentially with the same prompts, SQLite cache, output paths,
per-point validation and atomic publication. A failed evaluation stops the
worker and the run before a complete manifest can be published. The enabled
setting is recorded in the manifest. Use it only for a fresh, qualified run.

The base profile records the accepted project and evaluated paths under
`out/axbench/runs/ce941fe47-c2-six-method-20260816/`; the remaining source,
output, cache, and reflection-model paths stay explicit operator inputs.

Counter-only continuation additionally accepts `counter_resume_snapshot` and
`counter_resume_sha256` (both default `null`; set them together). They bind an
immutable partial-snapshot inventory to a fresh child output. Scientific config
and execution scope must match the parent. The snapshot also binds the
operator-attested parent tokenizer bytes, fitted-direction manifest and complete
judge-cache backup. See [counter continuation](../evaluation/axbench-adv.md#counter-continuation-from-an-immutable-partial-snapshot)
for admission, CPU-only preflight and final-judgment requirements.

Additive project counters require `project_counter_direction_root` and
`project_counter_layer`. The root must contain the completed manifest from
`scripts.eval.axbench_adv.fit_directions`. `project_counter_fit_seed`
defaults to `42` and must equal the fit recipe's seed. Before loading the target
model, the runner checks target weights and operative metadata, concept text,
layer/site/width, fit recipe, normalization and complete direction coverage.
Training data, statistics and runtime tokenizer remain recorded provenance.
Inputs are trusted and must remain unchanged during execution.

`pressures` defaults to `[gepa]`; `gcg` opts into the separate GCG campaign.
Its closed mapping contains `objective` (`response_margin` or `activation_l2`),
nonnegative finite `alpha`, positive `device_batch_size`, and the complete
`search` mapping from `FasterGcgConfig`. The effective search always uses the
top-level `master_seed`; its token-count grid must be unique, ascending, and
start at zero. `alpha=0.01` weights only the positive response loss and has no
effect on activation-L2. See the [GCG objective contracts](../evaluation/axbench-adv.md)
for the exact reductions and activation alignment. GCG serializes only when selected, so GCG-off manifests
retain their legacy bytes and fields.

`soft_prompt` is another opt-in pressure, configured by a separate closed mapping:

| Field | Default | Meaning |
|---|---|---|
| `objective` | `response_margin` | Same `response_margin` or `activation_l2` objective as GCG. |
| `alpha` | `0.01` | Finite nonnegative positive-response weight; omitted from L2 identity. |
| `device_batch_size` | `32` | Positive physical batch bound for scores and gradients. |
| `optimized_token_counts` | `[0,1,4,16]` | Unique ascending suffix-position counts starting at zero. |
| `learning_rate` | `0.01` | Finite positive Adam learning rate. |
| `max_steps` | `64` | Nonnegative update count; zero retains the scored initialization. |
| `universal_minibatch_size` | `16` | Positive deterministic training-pair minibatch size. |

The top-level `master_seed` supplies its seed. The effective identity records
the fixed FP32 Adam recipe and native-cast policy; these derived fields are not
user knobs. Initialization and every update are scored on the full training
population, with strict-best retention and earliest ties. There are no discrete
proposal, top-k, distance, temperature, projection, or norm-constraint settings.
Only selected runs serialize `soft_prompt` and `soft_prompt_training_source`.
See [soft-prompt semantics and tensor artifacts](../evaluation/axbench-adv.md#soft-prompt-search)
for the full project embedding tuple and clean-prompt conditioning policy.

Activation-L2 uses `max_activation(concept) * selected_factor(method) * DiffMean_direction(concept)`
as its residual target offset. Selection uses scores averaged over the full source
concept population and honors `factor_overrides`, including during sliced execution.
The raw method factor applies to all methods; project conditioning policies retain
their separate interpolation-weight conversion. Zero makes the offset zero; one
preserves the previous numerical objective. Neither suffix length nor `gcg.alpha`
rescales this offset. HF multiplies FP32 `[1]` calibration and factor tensors before
the saved BF16 direction, producing FP32; the project path multiplies the two Python
scalars before multiplying the direction cast to the residual dtype. Both losses
then use FP32. The nominal receipt coefficient is the Python-float product.

The derived effective-identity field `direction_factor_policy: selected_method_factor`
appears only for L2, including zero-budget receipts, and is not accepted as user
configuration. The merger rejects legacy L2 runs even at factor one: new evaluations
require fresh output directories, while existing direction fits and sources remain reusable.

Project `activation_l2` additionally requires the completed project DiffMean fit
root and its exact fitted layer through `project_counter_direction_root` and
`project_counter_layer` for positive budgets. Response-margin GCG and soft prompts,
and zero-only L2 runs, do not require them solely for these pressures.
The fitter's closed config is `configs/eval/axbench_adv/fit_directions.yaml`:
`prepared_dir`, `target_meta_path`, `feature_statistics_path`, fresh
`output_root`, `max_concepts`, `layer`, `seed`, and `device`. Target metadata must
be self-contained; external pretrain metadata is outside this surface. Remove
the obsolete `original_metadata_path` and `project_counter_original_metadata_path`
settings from old configurations; metadata-repair history is no longer required.

The native decoder-only counter extension has its own closed config at
`configs/eval/axbench_adv/native_counter.yaml`. `source_manifest` and
`source_parquet` name the completed native standard producer;
`prepared_dir` supplies the canonical AxBench training pairs. `checkpoint_dir`,
`step`, and `tokenizer_path` may relocate the same native SFT artifacts, but the
six `expected` step/hash/geometry values must match both the runtime files and
the producer. `layer` must match the reviewed geometry contract and is used for
both direction capture and counter intervention: D12/768/1024 uses layer 9;
the `native_counter_d14.yaml` preset requires the actual D14 SFT step and hashes,
D14/896/1024, exactly 399,115,836 loaded parameters, and layer 11.
`methods` and `magnitudes` are closed at `[DiffMean, LinearProbe]` and
`[0.25, 0.5, 1.0, 2.0]`.
`max_concepts` defaults to 50, `concept_count` to null, `seed` to 42, and
`device` to `cuda:0`. Null consumes the complete authenticated source. The only
supported subset is `concept_count=50` with `max_concepts=100`: this selects the
canonical 0..49 prefix for Adv without weakening source authentication. The
semantic identity and every point bind the complete standard100 manifest and
Parquet hashes plus this prefix scope; the target-only output manifest declares
50 concepts so the existing native judge validates it unchanged.
`phase=fit|generate|all`, nullable positive `max_cells`, nullable
`point_methods`, and nullable `point_magnitudes` are placement-only: they do
not enter semantic identity, so separate fit/generate invocations, bounded
pilots, and disjoint point workers resume the same exact full-grid output.
Subset filters must be nonempty, unique subsets of the closed grids. Magnitude
zero is absent from the generation grid and instead references the verified
standard emissions and their existing judge cache.

The separate `configs/eval/axbench_adv/native_methods.yaml` standalone-method
consumer has `zero_control_source: archived|runtime` (default `archived`). The
default requires exact generation replay and reuses authenticated standard100
judgments. `runtime` is accepted only for the reviewed D12/768/1024 and
D14/896/1024 contracts; it keeps the authenticated factor and fit, but durably
generates and judges a fresh positive-only shared zero with the same runtime,
cell seed and sampling settings as the attack points. Runtime policy is
conditionally added to the run identity, so policy changes cannot resume the
same output directory. It is not a full baseline refit or standard100
reevaluation, and no archived generation or rating is reused for the fresh
control cohort.

Proposal counts `(1, 2, 4, 8)` and the regenerated zero control are code-owned.
They are independent GEPA runs and cannot be configured as a counter-strength
sweep. The current c2 evidence selects factor `5.0` for both root methods, but
the runtime never hard-codes either. Both use a 1,024-token decoder, so their
128-token output leaves an exact 896-token rendered-prompt budget. See
[`docs/evaluation/axbench-adv.md`](../evaluation/axbench-adv.md) for prompt RNG,
seed, provenance, CUDA replay, and provider-approval contracts.

### AxBench-Base positive synthesis (`configs/eval/axbench_base/`)

The exhaustive AxBench-Base tables live in
[`docs/evaluation/axbench-base.md`](../evaluation/axbench-base.md#configuration-reference).
Its positive Stage 1b thresholds are fractions, while SAE QC returns raw
percentile points:

- `accept_threshold` defaults to `0.3`, giving required raw delta
  `D = 100 * 0.3 = 30`.
- `positive_prefix_headroom_ceiling` defaults to `0.9`. Candidate prefixes
  must score at most
  `S = min(100 * positive_prefix_headroom_ceiling, 99.5) - D`, hence `60`
  under the defaults.
- `min_prefix_chars` / `max_prefix_chars` apply to Stage 1b as well as Stage
  0c. Stage 1b keeps the most recent characters, excludes too-short prepared
  prefixes, and scores/emits that exact string.
- `qc_batch_size` defaults to `16` and caps both pre-provider prefix QC
  (Stage 0c and Stage 1b cache misses) and caller-thread post-provider QC
  (Stage 1b/1c). The post-provider effective cap is
  `min(qc_batch_size, max(1, synth.max_concurrent))`; both the cap and its
  stable-chain-frontier algorithm are recorded in Stage 1b identity because
  BF16 raw percentiles can depend on batch shape. Set `qc_batch_size` lower
  before a fresh transaction if post-provider QC exceeds GPU memory. Changing
  it or the effective cap cannot resume an authoritative partial: restore the
  original shape or explicitly restart Stage 1b. The scorer does not retry an
  OOM automatically with a different batch shape.
- `n_positives_per_concept` is the Stage 1b target of complete accepted
  rich/avoiding pairs, not a fixed seed roster. Failed seeds are replaced from
  the remaining deterministic candidate stream. Scoring always begins with the
  exact target-72 compatibility path; targets above 72 then use full scorer
  batches except at clean-pool exhaustion. Schema-v2 partial resume accepts the
  same or a higher target, never a decrease.
- `positive_max_attempts_per_seed` defaults to `8` total semantic attempts,
  independently for the rich side and (only after natural avoiding QC fails)
  the LLM-avoiding side. Natural avoiding QC consumes no API attempt.
- `positive_zero_hit_bail_after` defaults to `2000` distinct candidates. While
  a concept has no complete pair, Stage 1b admits no candidate beyond this
  boundary; `null` disables the guard. A success among already outstanding
  boundary candidates lifts the guard and refills every deficit.
- `max_retries` defaults to `100`, meaning 101 total content/QC proposals for
  Stage 0c and Stage 1c only. Stage 1b does not consult this compatibility
  field.
- `synth.transport_max_retries` independently defaults to `3`, meaning at most
  four HTTP requests per proposal. Retryable transport failures retain
  exponential 1s, 2s, and 4s backoff. Stage 0c resume-pins this field because
  it can determine whether the current proposal succeeds or rejection
  sampling advances to a different attempt-specific prompt.
- `synth.max_concurrent` defaults to `100`. Stage 1b uses the same complete-pair
  request-chain engine at `1` and above. Same-seed rich/avoid attempts keep
  their slot; replacement chains are appended to the global queue tail so the
  initial target for every concept is exposed before replacement work cycles.
  Stage 1c retains its existing fixed-prefix chain pool.
- `max_concept_drop_fraction` defaults to `0.5`. A Stage 1 concept is excluded
  only when the fraction of logical items lacking a complete, QC-accepted
  rich/avoiding pair is strictly greater than this value. Replaced seeds are
  internal work and do not count as drops; only final target shortfall does.
- `max_dropped_concept_fraction` defaults to `0.5`. Stage 1 fails after writing
  its diagnostic report when the excluded-concept fraction is strictly greater
  than this value; zero survivors always fails.

Validation requires
`0 < accept_threshold < positive_prefix_headroom_ceiling <= 1` and positive
ordered prefix bounds. The positive attempt cap must be a strict positive
integer; the zero-hit guard must be `null` or a strict positive integer. Both
concept-drop thresholds must be finite, non-boolean numbers in `[0, 1)`. The
resolved selection contract is written into the positive synth parquet and
validated by downstream same-config consumers:
`positive_target_refill_post_provider_qc_v5` at target 72 and
`positive_target_compatibility_post_provider_qc_v6` at every other target.

### Conditioned-Eval Feature Extraction (`configs/eval/extract_common_features.yaml`)

This config drives `scripts.preprocess.extract_common_features`, which produces the
`features.yaml` files passed to eval scripts through `feature_configs`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dir` | required | Steering dataset feature directory containing feature shards and `feature_statistics.safetensors` |
| `baseline_dir` | `null` | `null` means absolute selection; a path enables delta selection against that reference dataset |
| `stats_dir` | `null` | Statistics directory used for percentile ranking and discretization bucket boundaries; defaults to the sibling `base/` directory |
| `feature_selection` | geometric mean 10, max 40 | Sampling config used to sample `k` with the shared `SamplingSelection` parser |
| `negative_selection` | `null` | Optional `NegativeSamplingSelection` block for mixed delta extraction; valid only when `baseline_dir` is set |
| `top_k` | `null` | Explicit number of features to select; overrides `feature_selection` |
| `rank_metric` | `mean_pct_incl_zeros` | `mean_pct_incl_zeros` averages percentile scores over all segments; `mean_pct` averages active occurrences only |
| `magnitude_percentile` | `0.5` | Per-feature steering magnitude percentile used to assign bucket values |
| `num_buckets` | `10` | Number of discretization buckets; must match the encoder-decoder training config |
| `seed` | `42` | RNG seed for sampled `k` |
| `output` | `null` | Explicit output YAML path; otherwise generated under `out/data/common_features/{phase}/...` |

Delta mode is inferred from `baseline_dir`, not from a separate mode flag. It
computes steering-minus-baseline mean percentile scores for the positive pool.
When `negative_selection` is set, it also computes a separate baseline-minus-
steering pool over baseline candidate features (including baseline-only IDs).
Both pools sort by descending delta with feature-id ascending as the stable tie
break, and both exclude zero deltas.

The output YAML schema stays flat: `{feature_id: bucket_index}`. In mixed mode,
positive features are discretized against `effective_buckets = num_buckets - 1`
and negative features are written directly to the reserved inactive bucket
`num_buckets - 1`. Mixed inactive-bucket YAMLs are only valid for checkpoints
trained with `negative_selection`; eval scripts fail fast on incompatible
checkpoints.

### Shared training-loop parameters (from `_train_common.yaml`)

`base_train.yaml` and `conditional_encdec_sft.yaml` both inherit this narrow
loop surface. `conditional_encdec_train.yaml` continues to inherit
`base_train.yaml`, so its composed config is unchanged. Architecture, FP8,
pretraining schedules, and pretraining-only eval/sample controls live only in
`base_train.yaml`; conditioned SFT rejects those keys if a `+key=` override or
another config inherits them back into the composed surface.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `run` | dummy | Wandb run name ("dummy" = disabled) |
| `device_type` | "" | Force device type (empty = auto) |
| `max_seq_len` | 1024 | Context length |
| `device_batch_size` | 32 | Per-GPU micro-batch |
| `total_batch_size` | -1 | Tokens per step (-1 = auto) |
| `num_iterations` | -1 | Training steps (-1 = auto) |
| `embedding_lr` | 0.3 | Embedding learning rate |
| `unembedding_lr` | 0.004 | Unembedding learning rate |
| `matrix_lr` | 0.02 | Weight matrix learning rate |
| `weight_decay` | 0.2 | Weight decay (0.0 for SFT) |
| `eval_every` | 250 | Eval frequency (steps) |
| `eval_tokens` | 20971520 | Evaluation token budget |
| `save_every` | 500 | Checkpoint frequency (steps) |
| `resume_from_step` | -1 | Resume checkpoint (-1 = start fresh) |
| `model_tag` | null | Model identifier |

Conditioned-SFT checkpoints use the fail-closed ``buffered_v2`` cursor
contract. Resume restores each DDP rank's ordered best-fit buffer and next
striped cursor from artifact-pinned state; scalar-only resume is rejected. It
requires the same world size, buffer size, per-device batch
size, sequence length, gradient accumulation, ``min_loss_tokens``, and
``sft_loss_mask``. The top-level ``sft_segment_mode`` is also resume identity:
it is discovered from the feature artifacts rather than configured through a
new Hydra key, and an absent legacy value means ``fixed_tokens``.
Pre-``buffered_v2`` checkpoints lack enough information for
exact best-fit/DDP continuation and must start a fresh SFT run from the source
model.

The pretraining-only `base_train.yaml` extension owns `attn_impl`, `fp8`,
`fp8_recipe`, decoder architecture/scaling targets, `scalar_lr`, Adam betas,
warmup/warmdown/final-LR controls, and CORE/sample cadences. The
`remote_training` block is separate and belongs only to
`conditional_encdec_train.yaml`; see [encoder-decoder
training](../pipelines/encoder-decoder-training.md#verified-remote-c1024-run-issue-781).

### Encoder-decoder topology migration

`configs/tools/migrate_encdec_train_state.yaml` drives the single-process,
offline `steering.tools.migrate_encdec_train_state` utility. Its nested
`destination_training` default is
`full3b_clustered_c2048_long_w32_bf16_b14`; the migrator compares its resolved
checkpoint config against the source metadata and permits differences only in
`run`, `model_tag`, `resume_from_step`, and the world-scaled
`total_batch_size`. The comparison uses the trainer's checkpoint normalization,
so Hydra's operational `hydra` stanza is omitted whether the training preset is
composed at the root or nested below `destination_training`.

| Parameter | Default | Description |
|---|---:|---|
| `source_checkpoint_dir` | mandatory | Local native checkpoint leaf containing flat `model_NNNNNN.pt`, `meta_NNNNNN.json`, and the complete contiguous set of `optim_NNNNNN_rankR.pt` source shards. Remote-checkpoint metadata is rejected. |
| `source_step` | mandatory | Non-negative raw checkpoint step. It must match the metadata and is copied unchanged to the destination. |
| `packed_row_dir` | mandatory | Exact local packed-row family named by every saved pending-row reference. Headers, context length, row ownership, and the strict content fingerprint are validated read-only. |
| `selection_dir` | mandatory | Exact canonical `<packed_row_dir>/sel/<selection-hash>` family. Its manifest checksum must equal the leaf name, its direct shard stems must equal the packed stems, and its manifest must bind the packed content fingerprint. |
| `destination_dir` | mandatory | Fresh output leaf. It must not exist; execution publishes a complete sibling staging directory with one atomic rename. |
| `destination_world_size` | `32` | Positive target world size. The optimizer inventory contains exactly this many contiguous rank shards and every destination pending row uses this striping. |
| `dry_run` | `true` | Validate every source/config/model/optimizer/dataloader/artifact contract and report the plan without creating the destination. Set `false` only after reviewing the repeat count and effective optimizer scaling. |
| `max_replay_rows` | `null` | Optional non-negative operator guard. If the conservative previous-shard boundary would repeat more rows, validation fails before any write. `null` reports the exact repeat without imposing a policy. |
| `destination_training` | W32 BF16/B14 preset | Nested resolved training config used to rebuild the exact native mixed-dtype model contract, optimizer contract, and canonical destination groups. The production preset keeps accumulation 2 and 250,000 updates; its 1,835,008-token batch scales effective LR/weight decay by `sqrt(32/31)` relative to W31. |

The dataloader transform is intentionally replay-only, not an exact
cross-topology cursor migration. It requires `use_precomputed_batches=true` and
`label_dropout=0`; the audit metadata always includes the chosen boundary,
source frontiers, exact repeated rows/tokens, and zero skipped rows. See
[Offline topology migration](../pipelines/encoder-decoder-training.md#offline-topology-migration-issue-1514).

### Conditioned-SFT optimizer and dry-run controls

For `conditional_encdec_sft.yaml`, every optimizer group's effective initial
learning rate is its base group LR multiplied by both `lr_multiplier` and
`init_lr_frac`. The resulting value is stored as `initial_lr` before the loop's
schedule reads it. `dry_run: true` suppresses checkpoint files; it deliberately
does not disable Wandb, matching `chat_sft.py`.

`lr_warmup_steps: 20` is a strict nonnegative integer; `0` disables LR warmup.
Booleans, floats (including `20.0`), strings, null, and containers are rejected.
`lr_warmup_frac` is unsupported, including Hydra `+` overrides. All ordinary
hardware/head presets inherit 20; the six-update profiling preset uses zero.
Warmup counts completed optimizer updates and preserves the existing decay axis.
For positive warmup, fixed `num_iterations=N` requires
`5 * lr_warmup_steps < 4 * N` before device/model initialization. Epoch mode
(`num_iterations=-1`) rejects positive warmup when synchronized committed progress
is at or above 80% and completed updates are at or below the warmup length,
including termination.
See [LR warmup](../pipelines/encoder-decoder-training.md#learning-rate-warmup)
for update indexing, optimizer behavior and scope.

Exact optimizer resume requires matching saved `lr_warmup_steps` and
`num_iterations`. An absent legacy warmup field means zero: explicitly pass
`lr_warmup_steps=0`. See the [SFT resume guide](../pipelines/encoder-decoder-training.md#resuming-an-interrupted-sft-run).

`fp_head_lr: null` shares `unembedding_lr` with the LM head. A positive finite
number instead sets the base AdamW rate for the retained feature-head projection
and any non-matrix parameters in its residual blocks. The AdamW width scaling,
`lr_multiplier`, `init_lr_frac`, and schedule still apply.

`fp_head_matrix_lr: 0.02` sets the base Muon rate for 2-D attention and MLP
matrices in the feature head's residual blocks. These matrices are grouped by
shape and coalesced with compatible Muon groups; `lr_multiplier`, `init_lr_frac`,
and the schedule apply without AdamW's width scaling. A depth-0 linear head has
no such matrices and retains its prior single AdamW group. Resume restores saved
optimizer rates and `initial_lr`, including both rates; CLI learning-rate changes
do not retune a resumed optimizer. To use a new rate, start a fresh SFT run from
the checkpoint's model weights.

`muon_parameter_packing: false` retains the upstream optimizer by default.
Opting in gives Muon matrices persistent contiguous storage, avoiding their
per-step parameter stack and copy-back while retaining the original kernels,
group order, rates, and optimizer checkpoint format. Both single-rank and
distributed adapters bind to the raw model after checkpoint loading and final
device/dtype conversion, before the first training forward. Replacing a
Parameter, moving its storage, using `load_state_dict(assign=True)`, or changing
the distributed process group requires reconstructing the optimizer. Ordinary
`assign=False` model loads and optimizer resume at the same rank/world topology
remain compatible. Packing does not enable topology migration. A distributed
step failure invalidates the optimizer; outstanding collectives may still write
its storage, so restart from a completed checkpoint.
The adapter has CPU regression checks and scoped GPU evidence from before its
storage-lifecycle simplification. The current revision has not been rerun on
GPUs, and a complete-step speedup remains unqualified; see the
[evidence and limitations](../performance/encdec-sft-head-performance.md).

### Conditioned-SFT batch prefetch and profiling

These controls belong only to `conditional_encdec_sft.yaml`:

| Parameter | Default | Description |
|---|---:|---|
| `sft_prefetch_batches` | `0` | Strict non-negative integer. `0` preserves synchronous batch construction. A positive value bounds a FIFO of complete CPU/device batches produced on a background thread, but the producer is recreated with a finite view of only the current optimizer step and can never cross an evaluation or checkpoint boundary. The queue holds at most `min(sft_prefetch_batches, remaining accumulation batches)` items; the consumer can hold one more, and a producer blocked on a full queue can hold one completed pending item. Issue #1475 validates depth `2` for one exact 2xH200 B32/T1024/accumulation-8 geometry; depth `8` was slightly slower. This is not evidence for changing the cross-hardware default. |
| `profile_timing` | `false` | Write per-rank region timing CSVs. With SFT batch prefetch enabled, producer-side rows are tagged and published with the matching delivered batch; discarded lookahead is omitted. This is attribution instrumentation, not a throughput mode. |
| `profile_timing_sync_cuda` | `true` | Synchronize CUDA around main-thread timing-CSV regions so their wall durations include the enclosed device work. Prefetch-worker phase rows remain unsynchronized CPU wall times to preserve private-stream attribution. The main-thread synchronization still perturbs normal overlap. |
| `profile_trace` / `profile_trace_steps` | `null` / `3` | Optional first optimizer step and positive consecutive-step count for an all-thread CPU/CUDA Chrome trace. In a trace-only run, the profiler-listener context used by SFT `timed_region` markers is not inherited by the prefetch thread, so its fine-grained regions are absent; `profile_timing=true` installs the worker-local listener and supplies those rows (and markers if both diagnostics are enabled). Trace retention scales with operations and microbatches, not just optimizer-step count. A four-step B32/accumulation-8 capture exhausted 256 GiB while finalizing; start with the smallest capture and do not combine diagnostics with the throughput arm. |
| `profile_output_dir` | `out/profiles/encdec_sft` | Directory for enabled timing CSV and trace outputs. |

Startup timing for conditioned SFT has no configuration knob: it is always-on
and is emitted through the normal rank-aware `INFO` logger. It covers the
prologue through the first completed optimizer step with monotonic wall-clock
phases, including feature/parquet shard and document counts, alignment-probe
sample counts, percentile-coverage sizes, and a separate first-step warmup
line. `profile_timing` and `profile_trace` remain opt-in live-loop diagnostics
and are intentionally independent of this startup trace. See the
[startup-trace reference](../performance/profiling.md#conditioned-sft-startup-trace)
for the stable phase vocabulary and summary fields.

The positive-depth path reuses the dataloader's `ClosableBatchPrefetcher` for
bounded FIFO and worker lifecycle. It keeps a private producer copy of the complete
`buffered_v2` dataloader state. A state snapshot becomes checkpoint-visible
only when its matching FIFO batch is delivered, and the training loop still
commits that state only after the optimizer update succeeds. The producer
materializes device tensors on one private CUDA stream and records an event;
the consumer stream waits for that event and records every CUDA tensor on
itself before use. This makes overlap and cross-thread lifetime explicit
without a device-wide synchronization.

The optimizer-step bound is semantic, not just a memory limit. Training and
validation intentionally share stochastic feature-selection and negative-
sampling generators. Producing across the boundary would advance those RNGs
before validation and could leave a checkpoint with unconsumed random draws.
Queue depth is operational and does not enter artifact or checkpoint identity,
but changing it should still be validated against synchronous exact-batch,
loss-trajectory, and resume tests before a production run.

### Structured-SFT self-conditioning GRPO

For fully generated answers with deterministic final-answer rewards, use the
separate [encoder-decoder RLVR](../pipelines/encoder-decoder-rlvr.md) pipeline.
`configs/training/conditional_encdec_rlvr.yaml` requires `base_checkpoint_dir`,
`bundle_dir`, `output_dir`, `device_type`, `global_prompts_per_step`, `group_size`
(at least two), and `num_iterations`. `model_step=null` resolves the latest
base checkpoint; `stats_path` can relocate its exact statistics snapshot.
The only objective is the feature-policy surrogate; the LM head is frozen.
`train_feature_head_only=false` preserves full-policy training. Set this strict
boolean to `true` to train the entire feature head, including nested blocks,
with frozen-trunk replay under `no_grad` and no trunk optimizer state.
The complete head remains differentiable. The trainable manifest binds the
scope on exact resume; switching it requires a new run. See the
[RLVR training contract](../pipelines/encoder-decoder-rlvr.md) for replay and
memory implications. This default differs from structured-SFT GRPO below.

`frozen_decoder_max_shadow_bytes=0` disables optional BF16 copies of frozen
decoder Linear weights. A positive byte budget enables admission for single-rank
CUDA head-only training; `2147483648` allows up to 2 GiB of copies.
`frozen_decoder_min_free_bytes=4294967296` reserves an estimated 4 GiB for later
allocations. Both settings are nonnegative integers and are pinned on exact
resume. Admission includes reusable allocator reserves, so it is an estimate,
not an OOM guarantee. Unsupported configurations, overlapping registered tensor
storage, insufficient memory, or staging-only OOM use the existing native path.
See [frozen decoder weight copies](../pipelines/encoder-decoder-rlvr.md#frozen-decoder-weight-copies)
for the memory and checkpoint lifecycle.

`hidden_replay_max_cache_bytes=0` disables reuse of frozen trunk outputs during
feature-head replay. A positive budget enables a bounded cache within each
training prompt group in single-rank CUDA head-only runs; `268435456` allows
256 MiB of hidden states and prompt snapshots. The complete live feature head
still runs for every replay decision. `hidden_replay_min_free_bytes=4294967296`
requires estimated 4 GiB headroom, including reusable allocator reserves.
Both byte settings are strict nonnegative integers. The strict boolean
`hidden_replay_history_encoder=false` optionally reuses frozen encoder outputs
for repeated feature actions during canonical behavior collection. It requires
a positive hidden-replay budget and shares that budget within the same group;
it does not change token-generation encoding. The strict boolean
`hidden_replay_prompt_encoder=false` additionally reuses canonical prompt
encodes and requires `hidden_replay_history_encoder=true`. It owns fresh input
snapshots and complete packed outputs within the same budget. Empty-history
calls still encode the prompt natively; later calls check actual input contents
before reuse. All four settings remain pinned on exact resume.
Unsupported execution, memory refusal, or optional-copy OOM uses native replay;
model/head failures still propagate. Cache ownership is released before the
optimizer and checkpoints. See [hidden replay cache](../pipelines/encoder-decoder-rlvr.md#hidden-replay-cache)
for mutation checks, limits, and lifecycle.

RLVR `tasks` defaults to `[ARC-Easy, ARC-Challenge, MMLU, GSM8K]`. It is a
nonempty ordered list of distinct exact names, used only for training: sample
a task uniformly, then a row uniformly within that task. For MC-only training,
pass `'tasks=[ARC-Easy,ARC-Challenge,MMLU]'`. The full four-task bundle/cache,
token-budget admission, validation panels, and standalone evaluator remain
required/unchanged. Membership and order are scientific resume identity; a
change requires a fresh run from the original SFT checkpoint. Startup validation
warns on rank zero for every task with zero parse rate, including excluded
tasks, then continues. See the pipeline's native reward-format requirements.

`conditional_encdec_rlvr_depth2_s1125_full_h200` inherits this configuration
and explicitly selects all four tasks with `train_feature_head_only=false`.
It fixes `model_step=1125`, `global_prompts_per_step=8`, `group_size=4`,
`num_iterations=100`, `device_type=cuda`, and the experiment's WandB name.
The launcher pins one H200, BF16 and FA4. It requires explicit artifact paths
and the exact source revision. Both RLVR entrypoints recover the residual head
through the shared checkpoint loader: `feature_prediction.block` and
`feature_prediction.head_layers` (also mirrored in top-level
`feature_prediction_config`) explicitly declare the architecture. Legacy
metadata requires [conversion](../pipelines/feature-head-metadata-migration.md).
Loading a learned head preserves its tensors
under strict state loading. The experiment trains the complete policy and
excludes the LM head from its optimizer.

RLVR generation defaults are `temperature=1.0`, `top_k=null`,
`mc_max_tokens=16`, and `gsm8k_max_tokens=512`. `seed=42` controls training;
`validation_seed=314159` fixes validation selection and sampling independently.
`validation_per_task=32`, `eval_every=10`, and `save_every=10` set routine
validation/checkpoint cadence. The standalone `configs/eval/encdec_rlvr.yaml`
also accepts `validation_per_task=null` for the full prepared heldout split,
optional `rlvr_checkpoint_dir`/`rlvr_step`, and a required fresh `output_path`.

RLVR reuses the policy-only learning-rate defaults below, including
`init_lr_frac=0.05`, `lr_decay_start=0.8`, `grad_clip_max_norm=1.0`, and zero
`weight_decay`. `resume_from_step=-1` starts a new run; positive values require
a complete compatible checkpoint. `resume_checkpoint_dir=null` reads it from
`output_dir`; a deliberate earlier-checkpoint branch uses a fresh output
directory and `wandb.new_run=true`.

`wandb.enabled=true`, `wandb.mode=online`, and
`wandb.project=nanochat-encdec-rlvr` enable rank-zero metrics and bounded sample
tables by default. `wandb.entity` and `wandb.name` are optional. Set
`wandb.mode=offline` or `wandb.enabled=false` explicitly; initialization errors
never silently disable logging. No checkpoint upload occurs automatically.
Preparation defaults and optional per-task qualification caps are documented
in `configs/data/prepare_encdec_rlvr.yaml`; its source revision/count pins are
checked before optional caps.

`configs/training/conditional_encdec_sft_grpo.yaml` is an independent,
experimental surface. It does not inherit the ordinary SFT or pretraining
loop. The checkpoint and artifact paths plus all batch geometry are explicit;
there is no latest-checkpoint or measured-hardware default. See the
[pipeline contract](../pipelines/sft-self-conditioning-grpo.md). CUDA runs pin
FA4/CuTE on Hopper-or-newer devices and SDPA on earlier devices such as L40S.
Hopper without working FA4/CuTE support fails before model or artifact
construction rather than silently falling back. Hard deterministic algorithms
remain enabled for SDPA and SM100 FA4. They are disabled for SM90 FA4 because
PyTorch does not implement deterministic block-mask FA4 backward there. SM12
uses a project adapter to native CuTe causal/sliding-window attention, bypassing
unsupported FlexAttention block masks; its backward also requires deterministic
algorithms to be disabled. KV-cache decoding and packed encoder/cross-attention
keep their existing backends. The resolved backend, adapter code, and
deterministic-mode flag are resume identity.

| Parameter | Default | Description |
|---|---:|---|
| `base_checkpoint_dir` / `model_step` | mandatory / `null` | Conditioned-SFT-shaped base and explicit step (`null` uses that directory's last complete step). Eligibility is checked before artifacts or online SAE construction. |
| `train_artifact_dir` / `validation_artifact_dir` | mandatory | Separate strict `structured_nanochat_rollout_v1` train/val families. Legacy rendered-text parquet is rejected. |
| `action_mode` | `weighted_without_replacement` | Exact ordered weighted likelihood, or explicit `bernoulli_topk_bottomk` selected-feature surrogate. |
| `action_policy_version` | `2` | Sampling-RNG identity. Version 1 preserves per-trajectory stateless draws for old exact resumes; version 2 batches weighted count and action draws per trajectory microbatch while preserving the row-wise action distribution. Geometry is therefore part of v2 sample identity. |
| `initial_conditioning_mode` / `sae_loaded` | `none` / `false` | `none` is the exact no-SAE path. `online_sae` requires `sae_loaded=true` plus complete checkpoint-published online lineage. |
| `segmentation_mode` / `segment_length` | `fixed_length` / `256` | Fixed target chunks, or independently chunked logical turns. Routing is input-aligned. |
| `reward_mask_mode` / `min_reward_tokens` | `assistant_only` / `8` | Direct target-aligned assistant mask, or explicit `full_sequence`; ineligible conversations are deterministically refilled. |
| `kl_beta` | `0.0` | Zero performs no reference resolution/load/allocation/forward. Positive values use the immutable base as a frozen all-feature Bernoulli reference. |
| `train_feature_head_only` | `true` | Strict boolean, inherited by the pilot and ChatCORE presets. `true` trains the complete feature head and freezes all other policy parameters; requires `ntp=null`. Set `false` for the legacy full-policy scope, including old full-scope checkpoint resumes. See [training scope](../pipelines/sft-self-conditioning-grpo.md#training-scope) for optimizer, logging, and exact-resume behavior. |
| `ntp` | `null` | Auxiliary next-token-prediction objective; requires `train_feature_head_only=false`. `null` disables it entirely; a mapping enables it and must define every field below. There is no separate enable flag. |
| `ntp.trajectories_per_group` | — | First `k` trajectories scored differentiably per conversation; must be in `[1, group_size]`. Their losses are averaged. |
| `ntp.loss_weight` | — | Strictly positive; only its schedule may reach zero. This is a MIXTURE weight against the policy gradient, not a learning rate — see below. |
| `ntp.decay_start` | — | Fraction of the run at which the NTP weight starts decaying linearly to zero; flat before that. `0.0` decays across the whole run, `1.0` never decays within it. |
| `ntp.fp_head_lr` | — | FP-head AdamW LR before `lr_multiplier`; `null` shares `unembedding_lr`. The LM head keeps `unembedding_lr`. |
| `lr_decay_start` | `0.8` | Fraction of the run at which the optimizer's learning rate starts decaying linearly to zero; flat at full rate before that. Always active, with or without `ntp`: this is the RL run's LR decay. `0.8` is the historical shape. |
| `global_conversations_per_step` | mandatory | Unique eligible conversations globally; must divide by world size. |
| `group_size` | mandatory | Complete rank-local rollout group per conversation; must exceed one. The pilot extension pins 256 only. |
| `device_conversation_batch_size` | mandatory | Complete conversation groups collected together per rank; must divide the local conversation count. |
| `trajectory_microbatch_size` | mandatory | Collection and replay trajectory chunk; must divide `group_size`. |
| `cursor_prefetch_conversations` | `0` | Ordered eligible lookahead saved transactionally with attempted/accepted cursor state. |
| `grad_clip_max_norm` / `weight_decay` | `1.0` / `0.0` | Required finite positive global clip and required zero optimizer decay. |
| `resume_from_step` | `-1` | Explicit optimizer-step boundary beneath `selfcond_grpo_checkpoints`; exact identity/topology only. Every nonfinal save synchronizes ranks and re-executes workers from this boundary so ordinary continuation and recovery share the same process-local CUDA state. SDPA and SM100 FA4 are exact; SM90/SM12 FA4 cannot promise bitwise resume because their backward cannot use hard deterministic mode. |
| `profile_timing` | `false` | Write a rank-local CUDA-synchronized phase-timing CSV. This attribution mode perturbs execution and must remain separate from throughput benchmark arms. |
| `profile_output_dir` | `out/profiles/encdec_sft_grpo` | Directory created only when GRPO timing is explicitly enabled. Profiling settings are operational and excluded from exact-resume identity. |

For benchmark-focused RL, convert `mixture=chatcore` into structured train/val
artifacts and select `config_name=conditional_encdec_sft_grpo_chatcore` with
their paths. This extension sets `segmentation_mode=turn_delimited_fixed_length`
and `min_reward_tokens=1` so short ARC/MMLU answers are eligible. It retains the
base config's mandatory checkpoint and batch geometry. See the
[sources and invocation](../pipelines/sft-self-conditioning-grpo.md#benchmark-focused-subset).

`ntp.loss_weight` sets the RATIO between the two objectives, and that ratio is
the only thing it sets. Measured on one real optimizer step, comparing parameter
UPDATES:

| change | update difference |
|---|---|
| common rescale, `(w_pg, w_ntp)` `(2, 1)` -> `(1, 0.5)` | `9.6e-8` (float noise) |
| ratio change, `(2, 1)` -> `(1, 1)` | `2.9e-1` |

A common rescale of both objectives is inert because Muon orthogonalizes, AdamW
is scale-invariant, and `clip_grad_norm_` renormalizes on top — which is why the
policy side carries no weight of its own. Changing the ratio, however, changes
the update everywhere, including the LM head: raising `ntp.loss_weight` from 1.0
to 3.3 moves the LM head's update by `7.1e-2`. The weight is a real knob; it is
just not a per-parameter step size.

The step size comes from the optimizer's own per-group learning rates, and no
weight-to-learning-rate conversion is meaningful. Those rates follow the repo's
RL convention from `chat_rl.yaml` rather than conditioned SFT's: `init_lr_frac
0.05` (a 20x cut on every group; SFT uses `1.0`) and `embedding_lr 0.2` (SFT
uses `0.3`). Everything else matches SFT -- `unembedding_lr 0.004`, `matrix_lr
0.02`, `encoder_* 0.2 / 0.02`, `xattn_matrix_lr 0.02`, `lr_multiplier 1.0`.

An effective rate is `base x lr_multiplier x init_lr_frac x schedule`. For the
400M model (`d_model 768`, so the AdamW width scale is exactly `1.0`) the LM
head -- which the auxiliary objective alone trains -- lands at `0.004 x 0.05 =
2.0e-4`, against `4.0e-3` for conditioned SFT. Set `init_lr_frac: 1.0` to train
the readout at the full SFT rate. For reference, conditioned SFT weights its own
auxiliary feature-prediction loss at `0.1` against the main loss.

### The LR schedule shape

Both schedules are **flat, then linear to zero** — the decay point is the knob:

```
progress < decay_start          -> 1.0
decay_start <= progress < 1.0   -> linear from 1.0 down to 0
progress >= 1.0                 -> 0.0
```

`lr_decay_start` says *when decay begins* as a fraction of the run, and decay
always ends at zero at the final step. Over a 10-step run:

| `lr_decay_start` | per-step LR multiplier |
|---|---|
| `1.0` | `1,1,1,1,1,1,1,1,1,1` — never decays within the run |
| `0.8` | `1,1,1,1,1,1,1,1,1,0.5` — the historical shape |
| `0.5` | `1,1,1,1,1,1,0.8,0.6,0.4,0.2` |
| `0.0` | `1,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1` — linear across the whole run |

`ntp.decay_start` has the identical shape and meaning for the auxiliary weight.
Setting it earlier than `lr_decay_start` anneals NTP away relative to the policy
objective over the intervening steps.

`configs/data/convert_sft_to_parquet.yaml` keeps
`artifact_format=full_nanochat_text` as its compatibility default. The only
other value is `structured_nanochat_rollout_v1`, which selects the separate
authoritative-token schema and output namespace used here.

For `full_nanochat_text`, changing `max_conversations` or
`conversations_per_shard` requires a fresh output directory. Resume validates
the recorded sizes and actual retained indices before writing progress. See
[conversion resume and accounting](../pipelines/encoder-decoder-training.md#sft-parquet-format)
for the input-prefix versus written/skipped row counts.

### Decoder SFT backends and pinned Hugging Face presets

`configs/training/chat_sft.yaml` defaults `model_backend: nanochat`,
`huggingface: null`, and `grad_clip_max_norm: null`; native source resolution,
compilation, parameter-group learning rates, and checkpoint format are
unchanged. `model_backend` accepts only `nanochat` or `huggingface`.
The Hydra string is parsed into the internal `DecoderSFTBackend` enum before
any backend dispatch.

`chat_sft_smollm3_stage1_step_40000.yaml` selects the Transformers-native
`SmolLM3ForCausalLM` from
`HuggingFaceTB/SmolLM3-3B-checkpoints@524d818535ee98e56a4f0d621c84dd5c8fb8d3d4`.
`chat_sft_pythia_2_8b.yaml` selects final `GPTNeoXForCausalLM` from
`EleutherAI/pythia-2.8b@2a259cdd96a4beb1cdf467512e3904197345f6a9`.
`chat_sft_pythia_1_4b.yaml` selects the same family/tokenizer contract with
1.4B geometry from
`EleutherAI/pythia-1.4b@fedc38a16eea3bd36a96b906d78d11d2ce18ed79`.
Both inherit `chat_sft_huggingface.yaml`, which fixes BF16 autocast, FP32
parameters/Adam moments, SDPA, non-reentrant gradient checkpointing, context
1024, device batch 1, the standard mixture, full-sequence loss, and one AdamW
group at LR `2e-5`, betas `(0.8, 0.95)`, epsilon `1e-10`, and zero weight
decay. Native `embedding_lr`, `unembedding_lr`, and `matrix_lr` must be null on
this backend. Distributed HF runs require a positive `num_iterations`; the
rank-asymmetric dataset horizon remains legal only at world size one.

| Parameter | Default | Description |
|---|---:|---|
| `huggingface.attention_implementation` | `sdpa` | Exactly `sdpa` or `flash_attention_4`; passed to source loading and recorded in export/restart identity. |
| `huggingface.distributed_implementation` | `dist_muon_adamw` | `dist_muon_adamw` retains sharded optimizer-owned AVG reduction; `ddp_fused_adamw` compiles first if selected, wraps the adapter in DDP, uses `no_sync()` across accumulation, and runs fused CUDA AdamW. |
| `huggingface.torch_compile` | `false` | Optional adapter compilation before DDP construction. Benchmark it independently; it is not implied by an attention or optimizer choice. |
| `huggingface.restart_from_step` | `-1` | Explicit complete restart step. No latest discovery or topology migration; the loaded boundary is not rewritten before the next update. |
| `huggingface.restart_save_every` | `0` | `0` disables restart files. Positive values publish FP32 model, strategy-specific optimizer state, rank-local RNG/ordered-buffer state, and a complete marker at the interval and final boundary. |
| `huggingface.data_snapshot_sha256` | `null` | Optional lowercase SHA-256 of an operator-generated immutable data manifest. It enters restart identity; the launcher verifies the file digest before allocation use. |
| `huggingface.output_dir` | `null` | Root shared by portable `step_NNNNNN/` exports and the separate `restart/` namespace. |
| `huggingface.validate_export` | `true` | Release training objects, reload the final BF16 export using SDPA, and require finite forward/generation output. |

`huggingface.output_dir: null` writes model-only, non-resumable Hugging Face
checkpoints beneath `$NANOCHAT_BASE_DIR/hf_checkpoints/<escaped-model-id>/<revision>/standard_sft_checkpoints/step_NNNNNN/`.
An explicit value replaces the `standard_sft_checkpoints` root. Portable
`step_NNNNNN/` directories are still non-resumable. Complete restart state lives
under `restart/` and is accepted only when its marker, exact file-size
inventory, semantic identity, strategy, world size, and local-world size all
match. Every rank must also derive the same semantic digest before restart IO.
`dry_run: true` creates neither portable nor restart artifacts.

Decoder-SFT diagnostics are top-level because the native config supplies the
shared defaults, but only the HF backend currently consumes them:

| Parameter | Default | Description |
|---|---:|---|
| `profile_timing` | `false` | Write `chat_sft_timing_rank<R>.csv`. Every completed update includes an outer `chat_sft/paid_step` row measured by the loop's existing pre/post CUDA synchronization. |
| `profile_trace_steps` | `[]` | Unique positive completed optimizer-step numbers to trace, each no larger than a positive configured `num_iterations`. Must be empty exactly when `profile_trace_ranks` is empty. |
| `profile_trace_ranks` | `[]` | Unique zero-based ranks to trace, each smaller than the current world size. Nonempty step/rank lists enable tracing. |
| `profile_output_dir` | `out/profile/chat_sft` | Rank-local timing CSV and selected `chat_sft_trace_rank<R>_step<S>.json` outputs. |

The trainer always emits lightweight `chat_sft_startup_*` phase, first-step,
and summary lines for HF runs. No CUDA synchronization is added by that startup
trace.

For both decoder backends and conditioned SFT, `grad_clip_max_norm` is either
null or a finite positive number. Null skips clipping. Distributed HF
`dist_muon_adamw` requires null because clipping before optimizer-owned
reduction is not a global norm. HF `ddp_fused_adamw` may clip after DDP has
formed the globally averaged gradient. Existing native/conditioned behavior is
unchanged.

### SFT Loss Mask

Both `configs/training/chat_sft.yaml` and
`configs/training/conditional_encdec_sft.yaml` define
`sft_loss_mask: full_sequence`. The alternative `completion_only` mode
supervises assistant text and tool calls (`<python_start>` through
`<python_end>`) while excluding user turns and injected tool outputs. Training
and validation always share one resolved mode, and checkpoint metadata records
it as `sft_loss_mask`. Conditioned SFT applies the flag only to decoder targets;
feature routing, encoder tensors, segment ids, and feature-prediction targets do
not change. An encoder-decoder resume must use the checkpoint's mode; historical
checkpoints without the key are treated as `full_sequence`.

### Conditioned-SFT residual feature heads

`configs/training/conditional_encdec_sft.yaml` defines
`feature_prediction_block: mlp` and `feature_prediction_head_layers: 0`.
The block choices are `mlp` and `transformer`; depth is the total requested
number of residual blocks. Both choices at depth zero use a linear head.
Only nonnegative
integers are accepted; booleans, floats, strings and null are rejected.

Fresh SFT can upgrade a pretrained linear feature head to a positive depth
after strictly loading its source checkpoint. A headless source rejects a
positive depth. An already expanded source or an SFT resume requires the
configured block kind and depth to equal both saved fields. Every resume must
match both fields exactly, including at depth zero.

Both fields are persisted explicitly as `block` and `head_layers` in
`user_config.feature_prediction` and `feature_prediction_config`. Conflicting
mirrors and missing architecture fields fail. Generic inference, pretraining
resume, self-conditioning, GRPO and RLVR loaders restore the saved
architecture without an upgrade knob. Weight, attachment layer, pooling,
loss function, feature vocabulary, and normal all-model SFT optimization are
unchanged. The whole head uses the existing head optimizer group.

`conditional_encdec_sft_3b_residual_head.yaml` is the fixed depth-2 production
comparison against the historical linear-head instruct checkpoint. It inherits
the base SFT config and requires explicit artifact/output environment variables;
the [experiment runbook](../performance/residual-feature-head.md) describes
the preflight, launch, resume and evaluation procedure.
`conditional_encdec_sft_3b_transformer_head.yaml` inherits that recipe and
defaults to `transformer` with four layers. Ordinary overrides such as
`feature_prediction_head_layers=6` select other depths (including 8 or 10)
and update the run name. Each launch is one consumption-based epoch.

The removed names `feature_prediction_mlp_layers` and
`feature_prediction.mlp_layers` are rejected, including Hydra `+key`
overrides. Use the [metadata conversion utility](../pipelines/feature-head-metadata-migration.md)
for old checkpoints; runtime loaders never convert them implicitly.

### Conditioned-SFT source BOS origin

`configs/training/conditional_encdec_sft.yaml` sets
`feature_source_bos_char_len: 5`. This is the expected decoded character width
of the feature extractor's source-tokenizer BOS, used when full-sequence SFT
routes source character ranges onto nanochat tokens. Before constructing either
generator, conditioned SFT independently derives the width from the
lexicographically first train feature shard and, when validation is active, the
first validation shard. It pairs each feature shard with the same-stem parquet,
examines the parquet's first row group, and uses the first complete document
with a non-negative candidate width. The derived width must equal the configured
value for both sources.

This validation is always on: the key is not an escape hatch and there is no
warning-only or auto-correction mode. A missing/unreadable shard pair, no usable
complete document, or a width mismatch fails startup before generator
construction. Self-conditioning with `selfcond.data_source=sft` applies the same
check using the checkpoint-recovered value (legacy default `5`).

### `steered_jsonl` — steered SFT mixtures

`steered_jsonl` names the steered single-turn JSONL that
`mixture=standard_steered` / `mixture=standard_steered_balanced` splice into the
standard mixture. It is **required** for those two mixtures and **rejected** for
every other one — a set-but-unconsumed steered dataset raises instead of being
silently ignored. It is declared in `configs/data/convert_sft_to_parquet.yaml`,
`configs/training/conditional_encdec_sft.yaml`, `configs/training/chat_sft.yaml`
(inherited by every `chat_sft*` / `sft_d1*` variant) and
`configs/training/count_tokens.yaml`, and defaults to `null` everywhere.
`scripts/analysis/sft_parquet_alignment_audit.py` takes the same value as
`--steered-jsonl`.

Conditioned SFT must be pointed at the SAME file the parquet was converted
from: identity is the mixture fingerprint (row counts and order), and no
content hash is recorded — a default `artifact_format=full_nanochat_text`
conversion only stamps `steered_rows` into `conversion_meta.json` (the
structured rollout format records no steered field at all). The converter
accepts `split=train` ONLY for these mixtures; their validation family is
`standard`, so run `mixture=standard split=val` for the val side. See
[Steered mixtures](../pipelines/chat-training.md#steered-mixtures-mixturestandard_steered-mixturestandard_steered_balanced).

### SmolLM3 pretraining cooldown

`configs/training/smollm3_cooldown.yaml` is a deliberately fixed experiment
surface, not a general decoder-pretraining preset. The model ID and full
revision, 4,096-token context, 576-context global batch, device batch 3,
22,222-update horizon, 2,000-update warmup, AdamW hyperparameters, BF16/FP32
precision contract, and FA4 attention implementation are code-level immutable
constants. `data.prepared_dir` and `output_dir` are required shared-storage
paths. `data.parquet_dir` defaults to the approved HenryAC production corpus;
`data.tokenizer_batch_size`, `data.verify_hashes`,
`data.dataloader_workers`, `logging_steps`, and `save_steps` are operational.
`canary_steps` may stop and checkpoint early without changing the production
scheduler horizon. See [`docs/pipelines/smollm3-cooldown.md`](../pipelines/smollm3-cooldown.md).

### Conditioned-SFT content alignment

`configs/training/conditional_encdec_sft.yaml` sets
`alignment_probe_samples: 8`. Each train and validation generator deterministically
spreads up to that many indices across the live mixture, renders them with the
converter's exact full-nanochat helper — the same conversation slice the
training and validation generators now pack for every row, not just the
sampled ones — and byte-compares them with the matching
parquet `text` rows before yielding a batch. `0` logs an INFO and disables only
this content probe when fingerprints are available; legacy artifacts with neither
fingerprint are rejected at `0` because that would leave no verification channel.

Fresh `conversion_meta.json` files record the mixture name, split, constituent
task class names and lengths in construction order, `len(mixture)`, and a SHA-256
of the shuffled `TaskMixture.index_map`. Feature extraction carries that identity
in every feature shard header. Conditioned SFT uniformly validates those headers
as the authoritative feature-side fingerprint, including for validation families
that correctly have no statistics artifact. Train statistics metadata
corroborates the shard value when present; disagreement raises. Legacy artifact
pairs with no fingerprint on either side emit one loud warning and rely on an
active independent content probe. A fingerprint present on only one side or any
mismatch raises. Completed legacy conversions cannot be upgraded in place; use a
fresh converter `output_dir` and a fresh extraction directory.

### Encoder-Decoder Pretraining Parameters

Applies to `configs/training/conditional_encdec_train.yaml` and hardware
variants that inherit from it.

`conditioning_cooldown` adds an opt-in linear retention schedule to this
pretraining trainer. It does not change feature selection or the model shape.
See [Conditioning cooldown](../pipelines/encoder-decoder-training.md#conditioning-cooldown)
for exact update indexing, continuation/resume rules, and post-training lineage.

| Cooldown field | Default | Meaning |
|---|---|---|
| `conditioning_cooldown.enabled` | `false` | Enable the scheduled mask. Requires constant `label_dropout=0`. |
| `conditioning_cooldown.start_fraction` | `1.0` | Expected retained fraction of real units at and before the first endpoint; finite `[0, 1]`. |
| `conditioning_cooldown.end_fraction` | `0.5` | Expected retained fraction at and after the last endpoint; finite `[0, 1]`. |
| `conditioning_cooldown.start_step` | `null` | Inclusive zero-based optimizer update. Null derives a duration ending at the resolved end. |
| `conditioning_cooldown.end_step` | `null` | Inclusive zero-based optimizer update. Null resolves to `num_iterations - 1`. |
| `conditioning_cooldown.cooldown_tokens` | `20000000000` | Nominal duration when start is null, rounded up to whole global batches; inferred window must fit the horizon and contain at least two updates. Explicit step endpoints control duration otherwise. |
| `conditioning_cooldown.granularity` | `segment` | Independent segment decisions, or `document` to share one decision across every segment of a packed document. Junction segments (or documents containing a represented junction) drop last, preserving the requested overall expected fraction per microbatch. |

Issue #1615's persistent setup reuse for local precomputed validation is an
internal, always-on pretraining execution detail, not a configuration mode. It
changes no composed Hydra key, checkpoint identity, or resume contract. See
[Persistent local validation setup](../pipelines/encoder-decoder-training.md#persistent-local-validation-setup-issue-1615)
for its exact scope and artifact-trust boundary.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `packed_row_dir` | null | Producer-only required path for `scripts.preprocess.precompute_selected_features`. It is declared here so the documented typo-checked `packed_row_dir=...` override composes; encoder-decoder training auto-discovers packed rows from `feature_dir`, rejects a non-null value, and omits this key from checkpoint and W&B metadata. |
| `remote_preselection` | null | Producer-only remote transport for clustered, precomputed, sampling-based selection. It supports existing splice-CF or one-worker ordinary `feature_delta=null`; ordinary mode streams packed bodies only. A non-null mapping contains exactly `staging_dir` and `max_output_shard_bytes`; bucket, prefixes, retries, and timeout come from `remote_training`, whose `train_packed` block must be present. Transport settings are excluded from selection identity and training metadata. |
| `train_prefetch_batches` | absent / `null` | Optional strict non-negative complete-batch FIFO depth. Absent or `null` preserves the historical one-optimizer-step lookahead (`grad_accum_steps`); `0` selects the synchronous loader, and a positive value bounds the producer queue. It applies to live and ordinary local/remote precomputed routes when packed CUDA staging is not active. It is execution-only, remains in W&B's resolved runtime snapshot, and is omitted from checkpoint/resume and remote semantic identity. The exact local c2048 preset pins `64`, up to two nominal 32-rank-local-batch packed shards of decode lead. |
| `train_shard_lookahead` | absent / `null` | Optional strict non-negative decoded-shard-pair depth for the local packed-row train loader. Absent, `null`, or `0` preserves inline base/selection loading on the consuming producer thread; a positive value banks that many decoded pairs on one bounded daemon thread. Materializer-backed paths raise because `remote_training.body_prefetch` already stages their bodies. This is execution-only: it remains in W&B's resolved runtime snapshot but is removed from checkpoint/resume identity and the remote semantic fingerprint. The shipped `full3b_clustered_c2048*` presets pin `4`. At depth `K`, budget `K + 3` live decoded shards and a process-global shared-decode retain floor of `max(4, 2 * (K + 2))`; at the observed ~4.3 GiB/shard, depth 4 is about 30 GiB live and 52 GiB retained per node. See [Shard-level lookahead](../pipelines/encoder-decoder-training.md#shard-level-lookahead-train_shard_lookahead). |
| `train_node_cache_dir` | `null` | Optional node-local directory for a read-through cache of compressed packed-row and selection-shard bytes. `null` preserves the exact uncached path. A configured directory is created at startup and failure is fatal; materializer-backed remote paths reject this cache because they already stage bodies. This is execution-only: it remains in W&B's resolved runtime snapshot but is removed from checkpoint/resume identity. See [Node-local shard cache](../pipelines/encoder-decoder-training.md#node-local-shard-cache). |
| `train_node_cache_max_bytes` | `null` | Required strict positive integer when `train_node_cache_dir` is set. It bounds all regular-file bytes in the cache directory (payloads, sidecars, and live temporaries) without eviction: once the budget stops further fills, the first `K` shards in fill order stay cached and the remaining `N-K` continue streaming from the canonical source. Admission pre-counts the incoming payload bytes only; the new entry's roughly 300-byte sidecar is first included in the directory total at the next admission, which self-corrects subsequent decisions without a fixed allowance. Concurrent fills can transiently overshoot by up to the in-flight set before commits become visible: in the exact-resume worst case, one base shard per local rank (about `n_local_ranks x 1.5 GB`). This execution-only resource limit is omitted from checkpoint/resume identity. The world-32 BF16 B14 preset uses `3000000000000`. |
| `train_node_cache_reserve_bytes` | `3000000000000` | Required non-null strict non-negative minimum filesystem free space when `train_node_cache_dir` is set. A fill that would cross the reserve is skipped loudly while training continues from the canonical source. This execution-only safety limit is omitted from checkpoint/resume identity. |
| observability: startup trace | — | `conditional_encdec_train` emits a startup trace in the standalone entrypoint; there is deliberately no Hydra knob. It reports phase timing, the final packed/selection versus assembled/live route, first-update time, and invocation-local post-warmup mean. The trace is observational and does not enter hashes, checkpoint/W&B identity, RNG state, or fast-sweep JSONL. See [Startup timing and effective data path](../pipelines/encoder-decoder-training.md#startup-timing-and-effective-data-path). |
| `selection_resume` | true | Producer-only restart policy. `true` reuses ordinary legacy shards and, for clustered splice selection, only shards whose exact compressed bytes, per-shard splice counters, **post-serialization tensor validation digest**, and current `cluster_delta_semantics` were atomically certified in the `in_progress` or `complete` manifest. Each newly published selection shard is reloaded, checked for complete representative-only CSR storage against its signed segment count, and hashed under an unchanged-file identity guard before certification. The signed v3 record also accumulates the exact maximum positive-plus-negative CSR width and completion requires it to agree with the remote-loader certificate. This removes the former final replay of every large packed-row body while preserving both the semantic validation and remote static-padding contracts. A complete certificate also binds the packed-row fingerprint and effective bucket count, so a no-op rerun does not decompress packed/selection bodies or replay sidecar statistics. Files in the crash window after the last progress checkpoint are recomputed; a changed certified file is fatal. Pre-#790 schema-v1/v2 certificates are stale because they do not bind the corrected cluster-delta ordering and are rejected, not upgraded. A current-semantic manifest without per-shard stats still replays those stats once, validates the serialized selection shards, and creates a schema-v3 certificate without a final packed-body pass. `false` still validates selection/input/cluster identity, then deliberately recomputes every in-domain shard and can repair corrupt current-semantic shard bytes or progress certificates. Excluded from selection identity and training metadata. |
| `selection_splice_batch_size` | 64 | Semantic splice-CF selection mode. The default batches up to 64 consecutive ordinary (non-delta) executions while leaving splice-delta rows on the specialized serial delta-negative path. It uses tensor-native CSR assembly and a sparse masked multi-row negative sampler. Batching changes which random draws are assigned to rows, so the versioned algorithm descriptor **and the exact batch size** enter the delta-aware selection hash, splice manifest, and retained training metadata; it can never resume or overwrite a legacy/different-size `sel/` family. Set `null` or `1` explicitly to execute one segment at a time and preserve the exact legacy RNG schedule, `compute_selection_hash`, manifest shape, checkpoint metadata, shard bytes, and discovery/resume of existing legacy artifacts. The delta-free original-validation twin ignores this key. See [Multi-segment splice selection](../pipelines/encoder-decoder-training.md#multi-segment-splice-selection). |
| `selection_source_cache_shards` | 5 | Producer-only. Bounds the private/serial `FeatureIndex` resident feature-shard cache (`max_resident_shards`) used while preselecting `feature_delta.mode=splice_counterfactual` shards. Shared-store spawn workers instead keep one mapped stem and remap a store-published lazy document locator, so interleaved stem switches do not rebuild the complete per-segment object graph. Not read by training; excluded from `compute_selection_hash` and from checkpoint/W&B metadata. See [Splice-Preselection Source Caches](#splice-preselection-source-caches-issue-740). |
| `selection_sidecar_cache_shards` | 16 | Producer-only. Bounds the `SpliceProvenanceIndex` resident `_splice_provenance` sidecar cache for the same preselection pass. Not read by training; excluded from `compute_selection_hash` and from checkpoint/W&B metadata. |
| `selection_num_workers` | 1 | Producer-only packed-shard parallelism for splice-counterfactual preselection. `1` preserves the legacy in-process path. Values `>1` use `spawn` workers and invoke fail-closed resource admission before any output manifest/shared-store creation. The requested count is either accepted unchanged or rejected; it is never silently reduced. Excluded from selection identity and training metadata. |
| `selection_packed_spool_dir` | null | Producer-only packed-row storage mode. `null` keeps the eager compressed reader. A path must be an existing disk-backed directory (not `tmpfs`/`ramfs`); each active worker streams one decompressed safetensors temporary there and maps it only for that shard's context. Admission requires exact concurrent spool bytes plus free-space reserve. Excluded from selection identity and training metadata. |
| `selection_shared_store` | false | Producer-only source-sharing mode used only when `selection_num_workers>1`. Workers map immutable per-stem tensors plus an exact stable-order document locator through `NANOCHAT_SHM_DIR` (default `/dev/shm`) instead of cloning source tensors or rebuilding every segment object privately. Exact process identities, worker leases, a separate-session janitor, and a next-run sweeper govern cleanup. Excluded from selection identity and training metadata. |
| `selection_shared_store_capacity` | 0 | Maximum visible shared source stems; `0` resolves to `selection_num_workers + 4`. This count is not a physical-memory bound: admission includes up to `capacity + 2 * active_workers + 1` physical stems for visible entries, each worker's old and in-flight mapped generations, and one publication temporary, capped by source-stem count. |
| `feature_selection` | geometric mean/std 12, `min_k=1`, `max_k=49`, `mode=top_k_percentile`, `delta_weights=percentile_delta` | Default positive-feature sampling arm. Numeric constructor domains are phantom-refined and enforced by `beartype`: `mean` is finite and non-boolean; `std` is finite and positive; `min_k` is a strict non-negative integer; `max_k` is null or a strict positive integer; threshold values are finite and non-negative. Geometric sampling additionally requires positive `mean` and `std == mean`; every finite range requires `max_k > min_k`. `max_k` is exclusive. `percentile_delta` uses positive percentile deltas only on segments for which the configured delta mode produces delta evidence; other segments retain activation-percentile ranking. It does not itself enable counterfactual splicing. |
| `feature_delta` | `{mode: prev_segment, neg_delta_ratio: 1.0}` | Delta-reference mode. `neg_delta_ratio` is a phantom-refined finite non-boolean number in inclusive `[0, 1]`, enforced by the beartyped `FeatureDeltaConfig` constructor. Counterfactual splice training must explicitly set `mode=splice_counterfactual` and provide a spliced derived feature directory plus valid provenance; the default sampling arm's `percentile_delta` setting does not bypass those requirements. |
| `initialize_encoder_embeddings_from_sae` | null | Optional one-shot initialization for `encoder.feature_embed.weight` after random model init and before optimizer setup. `null` keeps the current random table; `random` projects the configured SAE `W_dec` through a deterministic random matrix; `svd` projects SAE decoder directions onto their top right-singular directions. Non-null modes rescale to the **full** `(n_sae_features, d)` random table's average row norm — never the compact table's, even when compacted. **Compaction-aware (#695):** the row subset lands in a *different place* per mode. `svd` subsets `W_dec` **before** projection, so its covariance is taken over the **selected rows only** and a compacted `svd` init differs from the row-subset of an uncompacted one — the one place compaction is not an identity. `random` projects the full matrix and subsets **last** (after the rescale), which is exactly what makes it bitwise identical to the uncompacted reference. |
| `initialize_encoder_embeddings_cluster_vectors` | mean | Static-cluster policy for SAE-backed init. Absent/`null`/`mean` (the default) replaces each retained representative with the fp32 mean of its mask-surviving members at the init mode's fixed stage when `feature_clusters` are active: `random` averages the normalized full projection **after** rescaling and **before** its final effective-vocab gather; `svd` averages oriented `W_dec` **before** its initial effective-vocab subset/covariance, so its basis is computed over the averaged effective dictionary. Without clusters or with `null` init, `mean` is an inert bitwise no-op. `representative` explicitly selects the representative's own direction byte-for-byte instead. |
| `feature_prediction` | **null** | Auxiliary feature-prediction head: `{weight, layer, loss_mode, loss_fn, block, head_layers}`. Off by default after the PR #697 ablations. The output width is the original SAE width without compaction and `V_effective` with a resolved mask/clusters. Compact FP pretraining requires `use_precomputed_batches=true` and a completed `fp_target_source_integrity` certificate in the packed manifest; conditioned SFT validates and folds its live per-conversation sources. Direct `return_fp_logits`/`fp_logits_only` and inference-time self-conditioned generation support both widths. Self-conditioning uses continuous normalized ranks and requires explicit selection-producer and bucket-producer identities. Raw-magnitude replay is limited to per-feature quantile cutoffs; cross-feature raw ordering fails closed. Clustered SFT thresholding uses the same full-distribution ranks as clustered sampling and FP targets. The non-Hydra runtime field `SelfCondConfig.fp_logit_bias` optionally adds one fixed, detached float32 value per full/compact head row after pooling and before sigmoid. The union-typed `score_policy` ranks pooled float32 logits directly for pure-model and sign-masked ordering, while interior interpolation combines `q=2*sigmoid(MAX_POOL(logits))-1` with a strict dense DiffMean vector in exact head-row order. `None` is the exact legacy path; compact callers retain the row mapping described in the [generation pipeline](../pipelines/self-conditioned-generation.md#dense-target-score-interpolation). The packed-pretraining post-training source remains uncompacted-only; source-aware SFT supports exact compact mask/cluster identity and compact-row STE. |
| `feature_clusters` | null | Optional path-only static cluster definition (`{path: ...yaml}` or JSON), supported by `precompute_selected_features`, precomputed encoder-decoder pretraining, sampling-based prompt caches for in-training CORE/sample generation, and the completion TUI as a checkpoint-metadata consumer (the TUI has no `feature_clusters` config key of its own). The resolved `feature_mask` chooses representatives; the encoder vocabulary then keeps exactly the allowed representatives and allowed unclustered ids, so active post-mask consolidation shrinks every feature-indexed embedding table. See [Static Feature Clusters](#feature_clusters--static-preselection-only-clusters). |
| `original_val_feature_dir` | null | Base feature-dir override for feature-shard splice dual validation. `null` auto-detects canonical `<base>/splice/cut_frac*/seed*` training feature dirs; set this when the splicer used a custom `output_dir`. In spliced runs, `val/*` logs the original/base validation and `val_spliced/*` logs the active derived validation. With `feature_delta=null`, both passes read precomputed packed rows. With `feature_delta.mode=splice_counterfactual` and `use_precomputed_batches=true`, both passes also read precomputed packed rows but require complete `sel/` shards: train + derived val use the delta-aware hash; original/base val uses the delta-free twin hash. With `feature_delta.mode=splice_counterfactual` and `use_precomputed_batches=false`, both passes run through live loaders. `prev_segment`/`cum_max` combined with a spliced dual-val dir raise at startup. |
| `original_val_data_dir` | null | Text source (parquet dir) for the `val/*` pass of live dual validation (`feature_delta.mode=splice_counterfactual` with `use_precomputed_batches=false`); unused by packed-row dual validation. `null` = nanochat default corpus. |

#### `remote_training` (verified c1024 run only)

Presence enables the issue-#781 mode; `null` is the unchanged local path. The
mapping is strict and requires the issue-#853 `body_prefetch` block in addition
to the existing remote identity/timeout keys. `async_checkpoint_upload` and
`train_packed` remain optional:

| Key | Dedicated preset value | Meaning |
|---|---:|---|
| `bucket_id` | `d0rb/nano-gpt-pretrain-steer-artifacts` | HF storage-bucket owner/name; token discovery is external to Hydra. |
| `body_prefetch.lookahead_shards_per_rank` | 2 | Required strict integer `>= 1`. Each spawned rank owner finalizes this many nearest obligations ahead, including receipts and pair-ready publication. There is no enable switch or removed-demand-downloader compatibility mode. |
| `body_prefetch.cache_shards` | 2 | Optional strict integer `>=2` (default `2`). The nominal owner evicts finalized train selection and packed bodies using this shared trailing logical-generation window. |
| `selection_prefix` | `features` | Bucket prefix corresponding to local `feature_root`. |
| `checkpoint_prefix` | `checkpoints/conditional_encdec` | Bucket prefix corresponding to the local conditional checkpoint root. |
| `selection_wait_timeout_seconds` | 14400 | Monotonic bound for owner body transport retries and consumer waits for packed or selection publication. |
| `checkpoint_retry_timeout_seconds` | 1800 | Monotonic bound for checkpoint transfer and verification. |
| `retry_interval_seconds` | 30 | Retry interval, capped by remaining time. |
| `resume_latest` | true | Validate every leaf-root completion marker, then restore the highest complete matching group; any malformed marker or marked-group corruption fails the walk. Requires `resume_from_step=-1`. |
| `async_checkpoint_upload` | true | After every rank completes its atomic local save, publish the immutable group on one rank-0 worker while training continues. The shipped remote preset opts in with `true`; omission defaults to `false` so existing custom remote mappings retain synchronous publication. |
| `train_packed` | absent | Optional strict mapping whose presence enables issue-#806 train packed-row streaming. It optionally accepts `trust_existing_cache` (default `false`) and the issue-#842 `cuda_staging` block. Absence preserves the issue-#781 selection-only path. |

After Hydra ingress, these scalar domains are represented nominally rather
than as plain annotations: bucket IDs and normalized relative prefixes use
remote-specific phantom types; timeout/retry values are positive finite
numbers; lookahead and CUDA-staging capacities are positive strict integers;
and the remote body-cache depth is a strict integer `>= 2`. Direct Python construction
and transport helper calls are enforced by `beartype` (including rejection of
`bool` as an integer). Raw YAML and signed JSON parsers deliberately translate
the same domain failures to `RemoteTrainingError`, preserving their stable
configuration/artifact boundary. Cross-field and inventory relationships such
as `rank < world_size`, index bounds, and generation equations remain explicit
checks.

Async checkpoint publication owns exactly one pending group. All ranks poll its
result together while it is outstanding, wait before starting another
checkpoint, and drain the final group before reporting success. Only remote
hashing/upload/verification is asynchronous: local model/metadata/optimizer
serialization and its cross-rank failure exchange remain synchronous. The
publisher pins each required file's device/inode/mode/size/timestamps/link
count at enqueue, rejects existing aliases (`link_count != 1`), and rechecks
through completion-marker verification. A mutation during marker upload may be
reported after that valid marker has linearized remotely; the run still fails
loudly, while complete-marker-only resume can safely recognize the verified
snapshot. `remote_complete_<step>.json` remains the marker-last commit record.
The operational boolean is retained in resolved runtime/W&B config but omitted
from checkpoint metadata, so `true` and `false` produce the same checkpoint
bytes and resume identity.

The canonical top-level marker is the sole publication authority. New data
objects are immutable at
`_groups/<full marker aggregate_sha256>/<logical filename>` under the configured
checkpoint leaf; the marker path, schema version 1, logical filenames, marker
bytes, and checkpoint bytes are unchanged. Readers accept both complete legacy
flat groups and complete immutable generations. If any object exists in the
marker-derived generation, all generation objects must match the marker; a
partial or corrupt generation never falls back to flat debris. With no remote
marker, a stale or malformed local marker and any old flat debris are
automatically bypassed on relaunch. Unreferenced objects are retained and never
mutated by publication. This is forward-only reader compatibility: do not use a
pre-fix binary after a generation-layout checkpoint commits. Checkpoint leaves
remain single-writer namespaces.

`remote_training.body_prefetch.cache_shards` is a strict integer greater than or
equal to 2, defaulting to 2. It is the trailing finalized-body retention window
for both pre-selected and packed bodies, measured against minimum logical
progress across all ranks. Only the nominal owner evicts. A paused rank can
prevent eviction while other ranks make every signed body durable, so
`2 * cache_shards - 1` is not a valid disk bound. Startup instead performs
byte-exact admission across all signed packed/selection finals, per-owner
lookahead stages, the largest active train or validation transport per owner,
receipts/control records, and a one GiB reserve. It fails before worker spawn
when free space is insufficient. Logical generations are `(epoch - 1) *
inventory_length + shard_index`, so a two-shard window at shard 200 removes
completed generations through 198, while epoch 2/shard 0 retains the previous
shard 369 and freshly materialized generation 370. The HF bucket is never
modified.

`remote_training.train_packed.trust_existing_cache=true` skips content hashing
for an already-present packed or selection body captured before worker startup
after checking that it is a direct, single-link regular file whose
before/after size equals the signed inventory declaration. The observed
descriptor signature is exactly mode, device, inode, size, `mtime_ns`,
`ctime_ns`, and link count; any change forces full verification. An unchanged
signature cannot prove byte identity, so trusted mode may accept same-length
bad bytes already named when the descriptor opens or same-length in-place
mutation during the read.

A fresh owner-published body may skip the consumer rehash after its
nonce-scoped publication receipt is present. The owner does not reopen or hash
the completed download: it trusts the HF/Xet client, atomically renames the
private transport to the final path, and records publication. Missing, stale,
or malformed fresh receipts fail loudly. Only strict mode provides per-read
content authentication. The shipped remote-packed preset enables the trusted
optimization and emits a warning; set it to `false` to hash every cache read.
The default remains `false`.

`remote_training.train_packed.cuda_staging` is absent from the generic preset
and has
exactly two required positive strict integers: `host_queue_bytes` bounds unique
CPU tensor-storage bytes in the pageable lookahead FIFO, and `pinned_slots`
bounds reusable pinned source leases and the ready GPU FIFO. Presence is
restricted to the verified remote packed/selection path on an explicitly
indexed CUDA device with static batch shapes and `label_dropout=0.0`; resumed
state carrying label-dropout RNG metadata is rejected. Absence uses the
explicitly closable remote CPU batch producer without the CUDA transfer stage.
See
[Precomputed Packed CUDA Staging](../pipelines/precomputed-packed-prefetch.md)
for ownership, tuning, failure, and test contracts. The hardware-independent
`conditional_encdec_splice_cf_gemma3pt_c1024_remote_packed` preset keeps the
block absent. The target-specific `_2xh200` and `_8xh100` presets enable
512 MiB/three slots; the H100 preset additionally pins B=16/BF16, the exact
production derived-feature root, the 524,288-token global batch, and disabled
steering eval while retaining the parent's `run` and `model_tag`. The H100
throughput benefit remains unproven until the production A/B gate completes.

Packed object waits reuse `selection_wait_timeout_seconds` and
`retry_interval_seconds`; there are no additional timeout knobs. The entire
`remote_training` block, including `body_prefetch` and `train_packed`, is
operational and excluded from the semantic remote-run fingerprint.
`body_prefetch` is also omitted from checkpoint `user_config`. An
all-local/selection-only checkpoint therefore remains resume-compatible with
packed streaming when every authenticated semantic artifact contract is
identical.

With `train_packed` enabled, `verify_features=true` is rejected because that
validator requires the complete raw family. No raw feature shard is required
when `steering_eval_every=0`. When clustered inline steering is enabled,
startup fetches the fixed `remote_inline_eval_source.json` certificate from the
configured `selection_prefix` and requires exactly the certified validation
raw shard, token-boundary leaf, and parquet. This adds no Hydra keys: the
certificate fixes the stem and content identities, and startup cross-checks its
derived-family fingerprint against the verified remote selection manifests.

Bucket IDs and prefixes must be normalized relative paths without traversal.
Operational timeouts, local paths, and resume controls do not enter the
legacy remote-run fingerprint.

#### `remote_preselection` (clustered remote producer)

`remote_preselection: null` takes the existing local producer branch unchanged.
The enabled mapping is strict:

```yaml
remote_preselection:
  staging_dir: /absolute/dedicated/scratch
  max_output_shard_bytes: 1000000000
```

`staging_dir` must already be a direct directory and must be disjoint from the
feature family, packed rows, logical `sel/<hash>` output, packed spool, and
shared-store root. The size bound is a positive strict integer. The producer
reuses `remote_training.bucket_id`, `selection_prefix`, retry settings, and the
canonical `feature_root` relative-path mapping; no alternate output layout is
configurable. It requires `remote_training.train_packed` and rejects
unclustered, gradient-ranked, threshold, unprecomputed, and unsupported delta
modes without falling back to local or live selection. Ordinary
`feature_delta=null` is limited to `selection_num_workers=1`; it streams only
packed inventory-v2 bodies and does not use raw/percentile bodies, source
inventory, splice sidecars, or `FeatureIndex`.

Worker count, parsed-source cache sizes, sidecar cache size, shared-store
capacity, staging path, and spool path remain resource settings. They do not
enter `compute_selection_hash`; `selection_splice_batch_size` remains semantic
and is addressed exactly as on the local path.

In ordinary mode the source/sidecar caches, shared-store settings, packed spool,
and splice-batch setting are parsed normally but inert. Remote receipts are the
resume authority even when `selection_resume=false`: that setting never permits
overwrite, deletion, or ignoring an immutable receipt. Partial receipt prefixes
are replayed to restore the continuous sampling RNG; complete receipt coverage
can publish the timestamp-free manifest without packed-body replay.

The separately versioned semantic fingerprint-v2 builder uses an exhaustive
classifier: every supported root and nested key must have exactly one declared
category, and unknown or multiply classified keys abort rather than silently
entering or escaping identity.

| Fingerprint-v2 category | Fields |
|---|---|
| Observation-only | `conditioning`; eval, CORE, sample, and steering-eval cadences, limits, prompts, and configs; steering-probe offload |
| Execution-only | `device_batch_size`, `save_every`, `remote_training`, resume/run/model/output names, profiling, document/feature/batch/shard lookahead and prefetch, and tokenizer batching |
| Location/resolved producer | feature/data/packed paths; raw mask/frequency/cluster settings; selection worker/cache/resume/shared-store/spool settings; `selection_splice_batch_size` |
| Semantic | model/encoder/cross-attention geometry; attention/compile/device/FP8; initialization; SAE identity; selection/delta/prediction/dropout behavior; seed; segment/bucket/data universe; optimizer/LR/schedule; horizon; `total_batch_size` |

The complete authenticated `train`, `derived_val`, and `base_val` selection
contracts, world size, and c1024 context length are always semantic inputs.
`decoder_segment_size` is an unsupported remote placeholder and must remain
`null`.

Metadata uses two distinct identities. `remote_semantic_identity:
{schema_version: 2, fingerprint: ...}` is the current semantic identity;
`remote_run_fingerprint` remains the immutable checkpoint lineage carried by
schema-v1 completion markers. Fresh v2 runs use one digest for both. Compatible
pre-v2 resumes retain their original v1 lineage and add the v2 record to every
new metadata file. Logs report `semantic_fingerprint_v2`,
`checkpoint_lineage`, and
`compatibility_mode=direct_v2|legacy_v1_migrated`.

`device_batch_size` may change at fixed `total_batch_size`; divisibility by
`device_batch_size * max_seq_len * world_size` is checked before restoration.
Exact saved `batch_rows` are regrouped without changing their ordered row
stream. Cursor-only legacy states cannot migrate.

Full eligibility, certificate, cache, and
checkpoint-marker behavior is documented in
[encoder-decoder training](../pipelines/encoder-decoder-training.md#verified-remote-c1024-run-issue-781).

### Splice-Preselection Source Caches (issue #740)

`selection_source_cache_shards` / `selection_sidecar_cache_shards` size two LRU
caches used only by `scripts.preprocess.precompute_selected_features` for
`feature_delta.mode=splice_counterfactual`: `FeatureIndex`'s source features and
`SpliceProvenanceIndex`'s `_splice_provenance` sidecars. Both classes retain a
default capacity of one for every other consumer. Private and W=1 feature indices
use the configured source capacity. A shared-store spawn worker deliberately keeps
one mapped source stem: each published stem also carries the exact stable eager row
order and sorted document IDs as two `int64[n_segments]` tensors, so an LRU switch
remaps those immutable arrays and constructs only the requested document. It never
re-sorts or rebuilds the full per-segment Python object graph. Spawn workers parse
sidecars through the compact array-backed representation; the W=1 path retains the
legacy object representation exactly.

The historical formula “source capacity × 12.7 GB plus sidecar capacity × 70 MB”
was not a memory bound. The 12.7 GB observation was a decode transient, while
eager safetensors tensors can retain their complete decompressed byte buffer.
Shared-store capacity was also only a visible-directory count: on Linux an
unlinked tmpfs file remains allocated while any worker maps it. For capacity `C`
and `W` active workers, the supported shared-store envelope therefore accounts
for up to `min(n_source_stems, C + 2W + 1)` physical stem allocations, not merely
`C`. `/dev/shm` consumes physical memory and may consume swap; enlarging it does
not add RAM.

For `selection_num_workers > 1`, startup now performs fail-closed admission from
validated compressed-safetensors headers and the resolved runtime dtypes. It
accounts separately for packed compressed/decompressed bytes, packed provenance,
source decode expansion, the fixed-size shared document locator and its build
transient, the unlink-aware shared union or private source caches,
sidecar state, measured worker-private state, current host `MemAvailable`, cgroup
v2 headroom, and filesystem reserves. A disk spool is rejected on `tmpfs`/`ramfs`
and must have room for one complete decompressed packed shard per active worker.
The `SPLICE_RESOURCE_ADMISSION` JSON log records every component and capacity.

Admission is a conservative, point-in-time supported-envelope check, not a
guarantee against later unrelated host pressure. A request that cannot be shown
to fit fails before creating the `sel/` output directory, in-progress manifest,
or shared store. It never silently lowers worker count. `selection_num_workers=1`
does not invoke admission and preserves the legacy output-creation/error path.

Parallel workers hydrate an immutable parent-validated `FeatureIndexCatalog`
instead of repeating the full source scan. Source materialization—including the
peer-publication eviction fallback—is serialized under one cross-process lease.
Shared-store ownership records boot ID, UID, PID, and process start time; workers
hold shared leases; normal cleanup requires the exclusive lease; and the janitor
can terminate only exact registered worker identities after parent death. Worker
signals are issued through a pidfd after bind-then-revalidate identity checking;
the implementation uses native CPython wrappers or registered raw Linux syscalls
and fails closed rather than using a PID-only fallback. Cleanup commands are
one-shot and close their parent pipe descriptors even on timeout. A
schema-valid dead-owner store left by all-coordinator `SIGKILL` is eligible for
the next run's conservative sweeper.

Output is byte-identical across these resource modes. None of the producer-only
execution keys enters `compute_selection_hash`, completed selection metadata,
checkpoint config, or W&B config. Size access locality with
`scripts/analysis/splice_preselection_locality.py`; measure a controlled run with
`scripts/analysis/benchmark_splice_preselection.py` and its process-tree resource
sampler. See the [pipeline guide](../pipelines/encoder-decoder-training.md#splice-preselection-source-caches-issue-740).

**Not the same thing as the [Base-Shard Hash Cache](../pipelines/feature-extraction.md#base-shard-hash-cache-sibling-writers-issue-530).**
That cache is always active, has no config knob, and stores a small on-disk
digest (a SHA-256 per base shard) — it never holds shard content in memory. These two
keys instead size in-memory parsed shard *content* only for the
`splice_counterfactual` preselection producer. The underlying
`max_resident_shards=1` class default governs all consumers that do not pass the
producer overrides.

### Self-Conditioning Post-Training Parameters

Applies to `configs/training/conditional_encdec_selfcond.yaml`. Architecture
and data identity are recovered from the base checkpoint; the config inherits
`base_train` only for the common loop surface. See the
[pipeline guide](../pipelines/selfcond-post-training.md) for prerequisites and
checkpoint layout.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_tag` | required at runtime | Base encoder-decoder tag. Because `base_train.model_tag` is concrete `null`, Hydra composition also yields `null`; the script raises unless this or `base_checkpoint_dir` is supplied. |
| `model_step` | `null` | Base checkpoint step; `null` is the only latest-step sentinel accepted by `load_checkpoint`. |
| `base_checkpoint_dir` | `null` | Explicit base checkpoint leaf. When set, `model_tag` is optional. |
| `segment_size` | `128` | Base-path locator used before metadata is loaded; a structural mismatch raises. |
| `feature_selection` | geometric mean/std 12, min 1, max 48, `top_k_percentile` | Base-path locator only; checkpoint metadata becomes authoritative after load. |
| `num_iterations` | `500` | Number of post-training optimizer steps. |
| `resume_from_step` | `-1` | `-1` starts fresh; a positive step resumes from the nested selfcond leaf. |
| `lr_multiplier` | `1.0` | Multiplier applied to optimizer LRs before the flat-then-linear-decay schedule. |
| `grad_clip_max_norm` | `1.0` | Global gradient-norm clip; `null` disables clipping. |
| `compile` | `false` | Must remain false in v1; rollout and the two gradient passes are distinct graphs. The startup trace records compile/graph warmup as intentionally skipped, while the first completed optimizer step is timed separately. |
| `eval_every` | `100` | `val/bpb` and `val/bpb_selfcond` cadence. |
| `save_every` | `250` | Nested selfcond checkpoint cadence. |
| `train_prefetch_batches` | `null` | In pretraining mode, `null` preserves parity by using the gradient-accumulation step count; an explicit non-negative integer overrides it. SFT mode requires `null` because prefetch would violate the buffered conversation-cursor contract. |
| `profile_timing` | `false` | Write one CUDA-synchronized region-timing CSV per rank. This intentionally perturbs overlap and is not a throughput mode. It is mutually exclusive with `profile_trace`. |
| `profile_trace` | `null` | Zero-based absolute optimizer step at which to begin a per-rank CPU/CUDA Chrome trace. `null` disables tracing. On resume, the step must be at or after the resumed step. |
| `profile_trace_steps` | `1` | Positive number of consecutive optimizer steps to trace. The complete half-open window must fit before `num_iterations`. Start with one: trace memory scales with operations and rollout slots. |
| `profile_output_dir` | `out/profiles/encdec_selfcond` | Non-empty diagnostics directory. It is not created while profiling is disabled and does not enter checkpoint or resume identity. |
| `core_metric_every` | `0` | Not wired; a positive value raises. |
| `sample_every` | `0` | Not wired; a positive value raises. |
| `steering_eval_every` | `0` | Not wired; a positive value raises. |

| `selfcond` parameter | Default | Description |
|----------------------|---------|-------------|
| `selfcond.train_scope` | `head_only` | `head_only` freezes all non-head parameters and uses one AdamW group; `full` uses the standard model optimizer. |
| `selfcond.ste_form` | `dense` | `dense` projects all FP scores; `selected` gathers only chosen scores. These are distinct estimators. |
| `selfcond.pool_mode` | `null` | Inherit the head loss mode. Supported values are `max_pool`, `last_token`, and `full_segment`; `best_token` must be overridden. |
| `selfcond.positive_counts` | `null` | Inherit checkpoint positive `{distribution, mean, std, min_k, max_k}`. A threshold checkpoint has no count distribution, so set this explicitly; there is no fallback. |
| `selfcond.negative_counts` | `null` | Inherit checkpoint negative count fields; an explicit block may override them but cannot make a checkpoint without `negative_selection` eligible. |
| `selfcond.data_source` | `pretraining` | `pretraining` consumes precomputed packed rows; `sft` consumes SFT conversations and their extracted features. SFT-only keys set in pretraining mode raise. |
| `selfcond.sft_dataset` | `null` | Recover the SFT base checkpoint's top-level `dataset`. An explicit `standard`, `capabilities`, or `safety` value must match the recorded dataset. |
| `selfcond.sft_feature_dir` | `null` | Recover checkpoint `user_config.feature_dir`. Canonical paths must match the dataset's SFT TRAIN phase and segment size; an explicit non-canonical path is the warning-only D8 escape hatch. |
| `selfcond.sft_parquet_dir` | `null` | Use checkpoint `user_config.sft_parquet_dir` when non-null, otherwise `<base_dir>/sft_data/<dataset>`. |
| `selfcond.val_sft_feature_dir` | `null` | With validation enabled, derive the matching SFT VAL phase from a canonical train feature path. Required explicitly when the train feature override is non-canonical. |
| `selfcond.val_sft_parquet_dir` | `null` | With validation enabled, use `<base_dir>/sft_data/<dataset>_val`. Required explicitly when the train feature override is non-canonical. |

Other data-identity keys are accepted as `+key=value` overrides. They default
to checkpoint `user_config`, overrides are logged, and disagreements in bucket
count, segment size, or SAE feature width fail loudly. In SFT mode, the five
`selfcond.sft_*`/`selfcond.val_sft_*` keys above own source-directory recovery.
The turn-only generated-assistant cadence has no key at all: post-training
measures it from the TRAIN artifacts and records it at the checkpoint metadata
top level, so supplying `selfcond.turn_self_cond_segment_size_rustbpe` is an
error, and consumers and resume read the recorded value. The other
checkpoint-recorded SFT generator scalars remain authoritative. The
recovered `feature_source_bos_char_len` (legacy default `5`) is nevertheless
checked against the first train and active-validation feature/parquet shard
pairs before either SFT generator is constructed.

Self-conditioning startup diagnostics are always enabled and have no Hydra
knob. Rank 0 emits canonical `selfcond_startup` JSON records with host-wall
`perf_counter()` durations; see the [startup trace contract](../pipelines/selfcond-post-training.md#startup-trace-and-timing).
This instrumentation does not change checkpoint identity, lazy validation
construction, RNG state, or data ordering. The four opt-in profiling keys are
also execution diagnostics only: current-run values are not copied into
self-conditioning checkpoint metadata.

### SAE Release Selection (extraction configs)

The `sae.release` key selects the SAE family, which in turn constrains the
valid `sae.l0` type and `model` names:

| Key | v1 (`gemmascope`) | v2 (`gemmascope_2`) |
|-----|-------------------|---------------------|
| `sae.release` | `gemmascope` | `gemmascope_2` |
| `sae.l0` | `int` or `null` (`null` = canonical) | Tier string in YAML: `small` / `medium` / `medium_seed_1` / `big` (no canonical/`null`). The public `sae_resolver.coerce_config_l0` helper coerces the YAML string → `L0Tier` enum before calling `resolve_sae`. An out-of-contract value (float, unrecognized string) raises `ValueError` immediately. |
| `model` | `gemma-2-2b` / `gemma-2-9b` / `gemma-2-27b` | `gemma-3-*` (see below) |

`gemmascope_2` requires an `L0Tier` value for `sae.l0` — passing an integer or
`null` raises a `ValueError`. `medium_seed_1` is only published for residual
SAEs at width 262k.

The `gemma-3-*` model names are: `gemma-3-270m`, `gemma-3-270m-it`,
`gemma-3-1b-pt`, `gemma-3-1b-it`, `gemma-3-4b-pt`, `gemma-3-4b-it`,
`gemma-3-12b-pt`, `gemma-3-12b-it`, `gemma-3-27b-pt`, `gemma-3-27b-it`. Note
the 270m pretrained model is `gemma-3-270m` with **no `-pt` suffix** (HF
publishes it as `google/gemma-3-270m`); every other size keeps the explicit
`-pt`/`-it` suffix. See [`docs/pipelines/gemma-scope-2.md`](../pipelines/gemma-scope-2.md)
for the full v2 model/layer/tier matrix. The `extract_features_gemma3.yaml`
sibling config sets these v2 deltas on top of `extract_features.yaml`.

### Backend Selection (extraction configs)

The `backend` key in extraction configs selects the activation-collection backend:

| Value | Description |
|-------|-------------|
| `auto` | Default; prefers nnterp when available, falls back to transformerlens |
| `nnterp` | nnterp-based hook (faster than transformerlens) |
| `transformerlens` | TransformerLens weight-rewrite backend |
| `native` | Home-grown raw-HF, structurally-truncated **residual-only** fast path; Gemma-2/Gemma-3 only, requires CUDA. Correctness-equivalent to nnterp/TransformerLens within oracle tolerances (jaccard≥0.99, mags rtol=1e-2). Opt-in only — not selected by `auto` in v1. |

Use `backend=native` on the CLI to opt in. The native backend is not the default; `auto` never selects it in the current release. It runs a structurally-truncated forward (inner Gemma decoder sliced to layers `0..L`, final norm replaced with `Identity()`, layers `>L`/`lm_head` freed — no forward hook; byte-identical to the prior full-forward + hook capture, correctness-gated). It is **residual-only**: mlp/attention activations, or a layer other than the configured one, raise a typed `ExtractionError` — use `backend=auto`/`nnterp` for those. Measured speedup vs the Phase-1 full-forward + hook baseline (bf16, RTX 4090; measured, not guaranteed) is ≈4.9× for gemma-2-2b @ layer 6 (≈179 → 37 ms) and ≈3.7× for gemma-3-1b-pt @ layer 7 (≈75 → 20 ms; also ≈1.4× faster than nnterp), at ≈2.2× lower peak GPU memory (≈17.5 → 8.0 GB and ≈7.4 → 3.3 GB respectively); the benchmark's pre-timing correctness gate verified the truncated path byte-identical (`max_abs=0`) to the Phase-1 reference on both real models before timing. Native `torch.compile` and SDPA row-packing are active only after mandatory startup gates pass; the row-pack gate is production-shaped (fixed-token/context-independent rows, BOS filtering, configured pooling) and hard-aborts with `RowPackGateError` on failure. See [feature-extraction.md](../pipelines/feature-extraction.md) Performance Notes for the full gate details and deferred FlashAttention/FlexAttention items.

### Segmentation Mode (extraction configs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `segment_mode` | `fixed_tokens` | `fixed_tokens` or `turn_delimited_fixed_tokens`. Turn-delimited mode requires an SFT `dataset_phase` and a parquet containing `turn_char_starts`; auto-constructed paths append `/turn_delimited` after `/context_independent` and `/no_bos`. |

### BOS-Token Filtering (extraction configs)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ignore_bos_features` | _removed_ | **No longer configurable** — BOS-feature exclusion is frozen always-on in the producer (a value passed via config/CLI is logged as an error and ignored). New runs always append a `/no_bos` directory leaf (after `/context_independent`; before optional `/turn_delimited`) and record `ignore_bos_features=true`; it changes the first segment of pretraining **and** SFT documents (not a no-op for SFT). Legacy artifacts with the flag off remain readable. See [`docs/pipelines/feature-extraction.md`](../pipelines/feature-extraction.md) § Ignore BOS Features. |

### Encoder-decoder SFT pipeline driver

`scripts.train.run_encdec_sft_pipeline` accepts the pipeline-only parameter
`feature_segment_mode` (default `fixed_tokens`). Set it to
`turn_delimited_fixed_tokens` when the chained SFT phases should resolve feature
directories carrying the final `/turn_delimited` leaf. This parameter is not a
`conditional_encdec_sft.yaml` key and is not forwarded to the trainer; it only
selects precomputed feature-directory identity and does not change SFT
rendering, loss masking, or trainer-side segment geometry.
Before launching each phase, the driver ensures that phase's statistics file
exists and runs `verify_feature_dir` on the feature directory. A shard or
statistics metadata disagreement with the path-encoded identity, including a
missing `segment_mode` under `/turn_delimited`, aborts the whole pipeline rather
than only that phase.

### Percentile Overlay Config (`configs/extraction/precompute_percentiles.yaml`)

The ordinary local producer retains its existing device auto-selection, DDP
striping, `resume`, `force`, dry-run, output, and profiling behavior.
`remote.enabled=true` instead selects the single-rank bounded HF producer:

| Key | Default | Meaning when `remote.enabled=true` |
|---|---|---|
| `remote.enabled` | `false` | Opt in. When false, no other `remote.*` field is parsed or validated. |
| `remote.bucket_id` | `henryac/nano-gpt-pretrain-steer` | Canonical Hugging Face `owner/bucket` ID for both source and output. |
| `remote.local_features_root` | `out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos` | Symlink-resolved local root used to map the logical `feature_dir`; escapes fail. |
| `remote.remote_features_root` | `l22_ci_nobos_raw` | Normalized relative POSIX bucket root. Output is beneath the resolved source prefix at `pct/<checksum16>/`. |
| `remote.staging_dir` | `null` | Required dedicated runner-owned directory. It must be disjoint in both containment directions from `feature_dir` and the profile path, and support direct files, nonblocking `flock`, atomic rename, and free-space reporting. |
| `remote.expected_shard_count` | `null` | Required positive non-boolean integer. The direct raw family must be exactly contiguous `0..N-1`; intended production passes `1400` explicitly. |
| `remote.max_output_shard_bytes` | `null` | Required positive non-boolean integer admission and post-save ceiling for one compressed percentile payload. |
| `remote.max_decompressed_shard_bytes` | `null` | Required positive non-boolean integer for CUDA remote runs. Caps the single plain-source lookahead and contributes to the host/cgroup memory-admission envelope; production passes `4000000000`. It is execution-only and excluded from the immutable output contract. |
| `remote.retry_timeout_seconds` | `900.0` | Positive finite per-object verified-transfer timeout. |
| `remote.retry_interval_seconds` | `5.0` | Positive finite retry interval. |

Remote mode additionally requires a readable root
`feature_statistics.safetensors`, positive `segment_size`, zstd level in
`[1,22]`, an explicit `device=cpu` or `device=cuda` (CUDA must be available),
and `WORLD_SIZE=1 RANK=0 LOCAL_RANK=0` (absent variables take those defaults).
It rejects `force=true`. `resume=false` permits a fresh or already-complete
family and rejects partial state; `resume=true` strictly adopts compatible
contract/payload/receipt/statistics state. Retry timing, staging and profile
paths, and the resume flag are operational and excluded from the immutable run
contract.

Every remote canonical object must have exactly one matching uploader-generated
claim. Claim-only objects count as partial state; stale, conflicting, duplicate,
or malformed claims fail closed. Direct percentile inventory ignores nested
`tok/` consumer trees, but malformed direct shard-like objects are errors.

Remote `dry_run=true` performs only configuration, exact source inventory,
output-name/state, device, disk-capacity, and CUDA lookahead-memory preflight. It performs no object
download, staging mutation, statistics rewrite, or remote write. Remote
`profile=true` uses the separate schema-v1 remote sidecar described in
[`docs/performance/profiling.md`](../performance/profiling.md#percentile-precompute-issue-1269).

### Feature-Shard Splice Config (`configs/extraction/splice_feature_shards.yaml`)

Post-extraction segment recombination (issue #590). Runs after `extract_features`
+ `compute_statistics`. CPU-only and weight-free. See
[`docs/pipelines/feature-shard-splice.md`](../pipelines/feature-shard-splice.md).

| Key | Default | Description |
|-----|---------|-------------|
| `feature_dir` | `null` | Base feature directory (**required**). |
| `data_dir` | `null` | Source parquet dir (`null` = nanochat default base data). |
| `output_dir` | `null` | Random mode: `<feature_dir>/splice/cut_frac{cut_fraction}/seed{seed}`. Opposite-percentile mode appends `_opposite_percentile_pool{B}` to the seed leaf. |
| `model` | `null` | `null` = parse from the `m_*` feature-dir path leaf; explicit value must match metadata. |
| `segment_size` | `null` | `null` = read from base shard metadata; explicit value must match. |
| `context_window` | `1024` | Must match base shard metadata; an explicit `null` resolves from the shard. |
| `cut_fraction` | `0.15` | Fraction of all splice points selected as cuts, in `[0.0, 1.0]`. Values outside this range raise before any shard work. This is NOT a per-boundary probability. |
| `seed` | `42` | Global RNG seed. Per-shard seed = `recombination_seed(seed, shard_index, cut_fraction)`; folding `cut_fraction` in means different fractions give independent cut selections even with the same global seed. |
| `pairing_mode` | `random` | `random` retains the legacy seeded derangement without loading statistics. `opposite_percentile` improves the summed absolute percentile difference between the incoming segment and the original continuation it replaces. Uses unquantized base statistics and zero for inactive features; no feature masks or clusters. |
| `pairing_candidate_pool_size` | `64` | Integer ≥ 2 in opposite-percentile mode, ignored in random mode. Bounds groups of candidate partners for one pass of improving two-partner swaps. Not an exhaustive search; subsequent tokenizer safety repairs may reduce the initial improvement. |
| `skip_existing` | `true` | Skip shards whose derived shard + parquet + provenance all exist **and whose sidecar params match this run** (mismatch raises). Requires `write_provenance: true`. |
| `write_provenance` | `true` | Write per-shard JSON sidecars under `_splice_provenance/`. Must stay `true` while `skip_existing` is `true`: `skip_existing: true` + `write_provenance: false` is rejected at config validation (no sidecar ⇒ no way to verify the existing outputs belong to this run). Set `skip_existing: false` to recompute without provenance. |
| `shard_parallelism` | `1` | `>1` spawns worker processes (one shard each); requires a picklable tokenizer. |
| `zstd_threads` | `0` | Integer ≥ 0 controlling zstd workers per active shard save. Zero retains existing compression; positive values can change compressed bytes, preserving decoded values. Nonzero settings enter provenance; resume requires the same value, treating an absent legacy key as zero. Budget workers alongside `shard_parallelism`. Async per-shard CPU timings exclude native compression workers; synchronous process CPU timings include them. |
| `zstd_level` | `null` | `null` keeps the writer default (currently 12) and omits the provenance key. Otherwise a strict integer in `[1, 22]`; recorded values must match exactly on resume, so explicit `12` differs from `null`. Compressed bytes, CPU cost and size can change; use a fresh output directory/prefix when changing levels. |
| `async_feature_save_depth` | `0` | `0` keeps synchronous save behavior. `>0` overlaps one bounded serial feature-save worker with the next shard's compute; rejected with `shard_parallelism>1`. A shard is not successful until its save future completes. |
| `profile` | `false` | Write an incremental per-shard JSON timing summary to `profile_summary_output`. Records fixed phase keys (`load_features`, `read_parquet`, `load_source_segments`, `recombine`, `build_derived_shard`, `build_provenance`, `write_parquet`, `write_provenance`, `save_features`, `total`), finer `detail_seconds`, counters, bytes, RSS/HWM, CPU seconds, and failed-shard error metadata. |
| `profile_summary_output` | `out/traces/splice_feature_shards_profile.json` | Path for the timing JSON (multi-rank runs append `.rankNNNNN`). The sidecar is atomically rewritten after each completed shard/future and once at run end. |
| `remote.enabled` | `false` | Opt into bounded verified HF-backed feature-body staging. `false` ignores every other `remote.*` field and preserves the local path. |
| `remote.bucket_id` | `d0rb/nano-gpt-pretrain-steer-artifacts` | Canonical Hugging Face bucket ID (`owner/name`). |
| `remote.local_features_root` / `remote.remote_features_root` | `out/data/features` / `features` | Map logical feature/output paths below the local root to normalized POSIX bucket prefixes below the remote root. |
| `remote.staging_dir` | `null` | Required in remote mode. Runner-owned bounded scratch, isolated as `rankNNNNN/{source,output,verify}` and protected by a nonblocking rank lock. |
| `remote.expected_shard_count` | `null` | `null` seals the current nonempty valid snapshot (gaps allowed); integer `N` requires exactly `0..N-1`. |
| `remote.retry_timeout_seconds` / `remote.retry_interval_seconds` | `900.0` / `5.0` | Finite verified-transfer timeout and retry interval. Timeout must be positive; interval may be zero. |

All shards in `feature_dir` are processed (no `start_shard` / `end_shard` range
filter). DDP striping via `RANK`/`WORLD_SIZE` distributes shards across ranks; a
striped run is byte-identical to the equivalent single-rank run for each shard.

Remote mode keeps source parquets local and stages only raw feature bodies plus
the existing output triplet. It requires `shard_parallelism=1`,
`async_feature_save_depth=0`, `skip_existing=true`, and
`write_provenance=true`. Source and output inventories are listed once per
invocation; remote ownership is stable `shard_index % WORLD_SIZE == RANK`.
Downloads and uploads use fixed depth one. Publication order is parquet,
provenance, then feature body; the verified feature body remains the completion
marker. Small immutable content-addressed records under sibling
`_verified_upload_claims/` prefixes fence concurrent canonical creation: an
identical claim is idempotent and a differing claim fails before canonical
upload. Remote objects are never overwritten with a different identity or
deleted. Complete-output verification independently resolves parameters from
one snapshotted raw source body before trusting any provenance sidecar.
It also requires the sidecar's logical `base_feature_dir` to match the request.

### `staging.*` — Local-Disk Output Staging (extraction configs)

Applies to `extract_features` (`configs/extraction/extract_features.yaml`). Controls the background save→relocate pipeline that decouples GPU forward passes from network I/O by staging output shards on local disk before moving them to the final network `output_dir`.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `staging.scratch_dir` | `/tmp/nanochat_extract_scratch` | Real local-disk path. Each run gets `run_<output-dir-hash>/range_<start_shard>_<num_shards>/rank{r}/{out,in}` below it, so disjoint shard windows targeting one output do not share or sweep each other's scratch. Identical/overlapping windows still compete for final shard ownership. Must have enough free space for `output_depth` shards across **all** local ranks and all simultaneous jobs (see `est_shard_gb`). |
| `staging.output_depth` | `3` | Concurrency of the staged save+move pipeline. `>= 1` enables staging (up to N in-flight save+move jobs); `0` writes synchronously and directly to `output_dir` (disables staging entirely); a negative value is rejected at startup. This replaces the former top-level `save_concurrency` key. Peak local disk usage: `output_depth × shard_size × ranks`. |
| `staging.input_depth` | `1` | `>= 1`: N-deep parquet→local-scratch→read prefetch — up to N shards are staged ahead via `shutil.copyfile` into `rank{r}/in` and each is deleted as soon as its texts are read into RAM. Each in-flight shard holds its decoded text in host RAM, so raise this only with headroom. `0`: inline read from the network parquet. A negative value is rejected at startup. |
| `staging.min_free_gb` | `6.0` | Minimum free space (GB) required on both `scratch_dir` and the final `output_dir` filesystem before each staged save/move. The saver logs while it waits and raises if either threshold is never met within the bounded timeout. |
| `staging.est_shard_gb` | `2.0` | Estimated compressed shard size (GB). Used only at startup (on local rank 0) for the independent capacity sanity checks on scratch and output: `output_depth × est_shard_gb × local_world_size + min_free_gb ≤ available_free_gb`. It does not reserve space or affect runtime behavior. |

Extraction progress is always logged independently of these staging depths:
each shard logs start/save-start boundaries and emits a fixed-cadence INFO
heartbeat (every 1,000 completed documents or 60 seconds at a completed-work
boundary). tqdm output is automatically disabled when stderr is redirected,
which keeps Slurm log files free of terminal control sequences. See the
[feature-extraction staging notes](../pipelines/feature-extraction.md#batch-log-progress-and-liveness)
for the exact fields and the limits of this liveness signal.

Extraction parses Hydra and torchrun values once into beartyped runtime
dataclasses backed by reusable phantom types. Consequently, shard counts,
rank/layer indices, staging depths, and the staging free-space floor are strict
non-negative values; context, segment, fixed/auto-search batch sizes, world size,
and estimated shard size are strict positive values; `gc_mem_threshold` is finite
and within `[0, 1]`; and `zstd_level`, when present, is an integer in `[1, 22]`.
`bool` is rejected for all integer domains. Cross-field relationships such as
minimum batch size not exceeding maximum batch size remain explicit checks.

See [`docs/pipelines/feature-extraction.md`](../pipelines/feature-extraction.md) § Local-Disk Staging for the full description, the EXDEV-safe move semantics, and instructions for verifying byte identity with `scripts/benchmark/verify_staging_identity.py`.

### Exact Feature-Statistics Reducer

Applies when `compute_statistics.streaming=true` (the default).

| Key | Default | Contract |
|-----|---------|----------|
| `n_sae_features` | `null` | Exact SAE dictionary width. `null` resolves the `f<width>` component of a canonical feature path before allocation. A noncanonical direct-library path performs a full width-discovery prepass and therefore reads bodies twice; production presets should pin this value. |
| `device` | `auto` | Device for the unchanged occurrence/sum/squared-sum/min/max accumulators and the bounded exact composite-key reduction. The exact magnitude-count table, sparse table updates, and finalization remain on CPU. Resolved CUDA full-detail runs with at least two remaining shards automatically overlap one next-shard read+zstd preparation; there is no worker-count/config surface, and CPU/skip-detail/one-shard paths stay sequential. Checkpointing does not force deterministic CUDA moment reductions: moment-derived output fields may differ within the documented `rtol=1e-6, atol=1e-6` regression tolerance, while exact-count-derived fields remain exact and `device=cpu` retains exact moment reproducibility. |
| `gpu_chunk_size` | `auto` | Existing CUDA moment-accumulator chunk bound. It does not size the CPU exact-count table or the exact reducer's fixed 16,000,000-activation CUDA chunks. |
| `skip_per_feature_detail` | Python API only | When true, do not allocate the magnitude-count table and omit per-feature percentiles, medians, and histograms. |
| `checkpoint_dir` | `null` | Opt-in dedicated persistent directory for crash-safe exact-pass snapshots and automatic compatible resume. It must not contain the feature directory or output path. Requires streaming with no activation-percentile filter, a width resolved before body reads, and local POSIX atomic rename/fsync/flock semantics. `null` preserves the legacy no-checkpoint path. |
| `checkpoint_interval_shards` | `50` | Positive integer cadence. A generation is committed after each interval and always after the final shard. Restart replays at most one full interval if interruption lands during a commit, otherwise at most interval minus one completed shards. Does not enter run identity or output contents. |
| `profile` | `false` | Enable the exact streaming feature-statistics performance sidecar. Requires `streaming=true`; the reducer and statistics artifact are unchanged. |
| `profile_summary_output` | `null` | JSON sidecar path; `null` resolves to `out/traces/compute_statistics_profile.json` when profiling is enabled. The path must differ from the statistics safetensors output. |

The exact CPU allocation is `n_sae_features * 31_743 * 8` bytes
(15.4995 GiB at 65,536 features). Admission uses the smaller of current host
available memory and the tightest finite cgroup-v2 headroom from the process'
leaf group through every ancestor, when present, and refuses the run if the
table alone does not fit. The process still needs extra headroom for the current
decoded shard and reducer/GPU transients; eligible CUDA prefetch additionally
holds at most one raw next-shard safetensors payload plus its in-progress
compressed input. No spill fallback exists.
On the measured RTX 4090, a 16M exact-count CUDA chunk reserved about 967 MB (922 MiB);
the requirement is runtime-dependent. A CUDA OOM propagates without adaptive
resizing or CPU fallback, so use `device=cpu` on lower-headroom devices.

Checkpoint generations are complete reducer snapshots, not deltas. At 65k
width a generation includes the 15.4995 GiB table, and crash-safe replacement
temporarily retains two generations, so allow slightly over 31 GiB free disk.
The checkpoint manifest binds exact shard file identities, all semantic input
metadata, dictionary width, reducer versions, and the resolved numeric runtime.
Restore rejects any mismatch or corrupt state. See
[`feature-extraction.md`](../pipelines/feature-extraction.md#exact-pass-checkpoints-and-resume)
for the transaction ordering, I/O cost, and recovery contract.

The reducer accepts only the production magnitude contract: positive, finite,
exactly float16-representable values loaded as float32. Violations and feature
IDs outside the declared dictionary fail before an output statistics artifact
is written.

`remote` is a disabled-by-default HF input block for that same reducer:

| Key | Default | Meaning when `remote.enabled=true` |
|---|---|---|
| `remote.enabled` | `false` | Enable identity-sealed HF input instead of local raw shard bodies. |
| `remote.bucket_id` | `henryac/nano-gpt-pretrain-steer` | Typed `owner/bucket` HF bucket ID. |
| `remote.local_features_root` | `out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos` | Symlink-resolved local root used to derive the logical relative feature path. The default maps that exact production family directly to the Henry bucket prefix. |
| `remote.remote_features_root` | `l22_ci_nobos_raw` | Normalized relative POSIX bucket root. Override both roots together for another feature family. |
| `remote.staging_dir` | `null` | Required dedicated locked directory for current/lookahead compressed bodies. |
| `remote.expected_shard_count` | `null` | Optional exact contiguous full-family count; intended production uses `1400`. |
| `remote.retry_timeout_seconds` | `900.0` | Positive finite per-object verified-download retry timeout. |
| `remote.retry_interval_seconds` | `5.0` | Positive finite retry interval. |

Disabled mode reads only `remote.enabled`; malformed unused fields do not
affect local runs. Enabled mode requires exact streaming, zero activation
filtering, no `_preloaded` input, an explicit positive width, and a canonical
feature path whose `s{N}` component equals `segment_size`. Staging must be
disjoint in both containment directions from the logical feature directory,
forced-`.safetensors` output, checkpoint directory, and profile path. See
[feature extraction](../pipelines/feature-extraction.md#identity-sealed-hf-streaming-input)
for inventory, disk, checkpoint, fence, and cleanup contracts.

### Remote token-boundary input (`precompute_token_boundaries.yaml`)

`precompute_token_boundaries` has a separate disabled-by-default `remote`
block. The configured remote prefix remains the raw-family identity root, but
the runner requires and streams the deterministic compact v1 source derived
from that exact raw inventory. Paired parquets, overlay output, manifest,
statistics, and profile remain local. There is no compact flag and no raw
fallback.

| Key | Default | Meaning when `remote.enabled=true` |
|---|---|---|
| `remote.enabled` | `false` | Enable authenticated compact-source input. Disabled mode reads no other `remote.*` value and uses the unchanged local raw runner. |
| `remote.bucket_id` | `henryac/nano-gpt-pretrain-steer` | Canonical `owner/bucket` HF Storage Bucket ID. |
| `remote.local_features_root` | production L22 CI/no-BOS feature path | Local mapping root containing the logical `feature_dir`. |
| `remote.remote_features_root` | `l22_ci_nobos_raw` | Normalized raw bucket prefix mapped to `local_features_root`; the compact prefix is derived automatically. |
| `remote.staging_dir` | `null` | Required scratch directory, disjoint from `feature_dir`, its token overlay, and `data_dir`. |
| `remote.expected_shard_count` | `null` | Exact contiguous `0..N-1` raw and compact family count. Production must pin `1400`; null infers the observed raw family and warns. |
| `remote.retry_timeout_seconds` | `900.0` | Positive finite verified-download retry timeout. |
| `remote.retry_interval_seconds` | `5.0` | Positive finite retry interval. |

Startup seals raw metadata, derives
`<raw-prefix>/_token_boundary_source/v1/<raw-inventory-sha256>/`, authenticates
its manifest, and requires an exactly matching compact inventory before any
compact body or semantic output. A missing manifest reports the exact publisher
command. Every invocation streams the complete compact family, including
resume and completed runs; no raw feature body is downloaded by this runner.
`shard_start`/`shard_end` select overlay computation only; their defaults are
`0`/`-1` (all), and remote `force=true` requires the complete range. Staging
admission uses the two largest compact bodies plus 1 GiB. Candidate overlays
remain private until the raw inventory, compact inventory, and compact manifest
identity all pass their final fences. Parquets are independent required inputs:
they remain local, must exactly pair with the raw feature family, and are not
contained in or replaced by the compact source.

The one-time publisher uses
`configs/extraction/publish_token_boundary_source.yaml`:

```bash
uv run python -m scripts.preprocess.publish_token_boundary_source \
  raw_prefix=<raw-hf-prefix> \
  staging_dir=<existing-dedicated-scratch> \
  expected_shard_count=1400
```

It has no resume, force, dry-run, range, worker, or compression knobs. Each run
reprocesses the sealed raw family, verifies same-path immutable sidecars, and
publishes `manifest.json` last. Partial sidecars are not consumable before that
commit and make a rerun naturally resumable. Publishing uploads to HF and
therefore requires separate operational authorization.

### Packed-row percentile representation (`precompute_packed_rows.yaml`)

Packed-row completeness is an executable producer invariant rather than a
manifest-version contract. The producer rejects empty input, consumes every
loaded document, and drains the final lookahead buffer; resume replay follows
the same path. Completed manifests may contain unrelated metadata extensions,
while inventory publication continues to bind their exact immutable bytes.

Packed-row preprocessing keeps the percentile overlay's already-quantized
`uint8` tensors encoded through row assembly. This is the only packed-row code
path and has no configuration knob. The live encoder-decoder dataloader retains
its float32 `SegFeatures.percentiles` contract. Packed preprocessing disables
live-only bucket collection in the shared segment builder, and its private
segment carrier has no `buckets` field. The former
`preserve_percentile_uint8` key has been removed; delete that override from old
commands or configs because strict config loading rejects it.

### Remote packed-row raw and percentile input (`precompute_packed_rows.yaml`)

`precompute_packed_rows` has its own disabled-by-default `remote` block. It
streams raw HF feature bodies and their checksum-addressed percentile subfamily;
statistics, token overlays, parquets, packed outputs, provenance, certificates,
mirrors, and profiles remain local unless output offload is enabled. The
percentile manifest is verified directly from the bucket snapshot. No source
toggle exists: remote mode requires both families and never reads percentile
bodies through a mount.

| Key | Default | Meaning when `remote.enabled=true` |
|---|---|---|
| `remote.enabled` | `false` | Enable identity-sealed HF raw and percentile input. Disabled mode reads no other `remote.*` value and dispatches to the literal local runner. |
| `remote.bucket_id` | `henryac/nano-gpt-pretrain-steer` | Canonical `owner/bucket` HF Storage Bucket ID. |
| `remote.local_features_root` | production L22 CI/no-BOS feature path | Symlink-resolved local mapping root containing the logical `feature_dir`. |
| `remote.remote_features_root` | `l22_ci_nobos_raw` | Normalized relative bucket prefix mapped to `local_features_root`. |
| `remote.staging_dir` | `null` | Required dedicated scratch, disjoint in both containment directions from `feature_dir`, `data_dir`, and the enabled profile output. |
| `remote.expected_shard_count` | `null` | Exact contiguous `0..N-1` family count. Production pins `1400`; null seals the observed contiguous family and warns. |
| `remote.retry_timeout_seconds` | `900.0` | Positive finite verified-download retry timeout. |
| `remote.retry_interval_seconds` | `5.0` | Positive finite retry interval. |
| `remote.offload.enabled` | `false` | Publish each completed packed body and provenance sidecar to the bucket, verify its size + Xet identity, receipt it under `_offload/`, then evict the local bodies. Required for families larger than their filesystem (c2048 is ~2.4 TB against a 1 TB quota). |
| `remote.offload.remote_packed_root` | `null` | Required when offload is enabled. Normalized relative bucket root the packed output tree is published under, mapped from `remote.local_features_root` exactly like the input roots. Must be disjoint from `remote.remote_features_root` in both containment directions. |

Remote mode requires local statistics, a complete schema-v2 percentile manifest
in `<raw-prefix>/pct/<pct16>/`, a complete local token manifest, a sealed
percentile leaf family positionally equal to the sealed raw family, token leaves
that are a subset, and at least two selected local parquets. It applies
`max_shards` before the train/validation split and downloads that selected prefix
from both streams. Staging is admitted for the two largest raw plus two largest
percentile bodies plus 1 GiB; each stream has fixed one-body lookahead. It never
deletes remote objects, retains source bodies, or writes the local base-hash
cache, and it uploads only its own output, only under
`remote.offload.remote_packed_root`, and only with offload enabled.

With offload enabled a shard is resume-complete from its receipt alone, so local
growth stays bounded at `compress_workers + 2` awaiting publication and a resume
first reconciles whatever an earlier window left on disk. Offload is
deliberately outside the resume-state identity — it changes where bodies are
stored, not their bytes — so it can be turned on mid-family. Mirror symlinks are
not published for an offloaded family, and the FP-target certificate is folded
from per-shard scans taken before eviction rather than from a whole-family
rescan that an evicted family could not satisfy.

`resume=true` is authenticated by the atomic
`.sN.remote-packing-state.json`/`.sall.remote-packing-state.json` marker until
both manifests, FP certificates, and mirrors validate. Source identities,
packing policy, context, `max_shards`, buffer/row sizes, `zstd_level`, and SAE
width enter that schema-v2 semantic-identity-only state. Its source identity retains the bucket/prefix,
sealed raw inventory count/SHA-256, the percentile inventory SHA-256, expected
base hash, ordered stems, segment size, overlay checksums, and selected parquet
`{stem, n_rows}` entries; filesystem
signatures are excluded. A stale schema version is refused with instructions to
remove the state and both partial train/validation output directories before
starting fresh. Staging/retry paths, resolved `compress_workers`, and
profile settings are operational and excluded. `resume_from_shard` is rejected
in remote mode. Canonical bodies can be visible while the marker exists; they
must not be consumed until the marker is absent and both certified manifests
validate. Every successful invocation authenticates the verified remote
percentile manifest and local token overlay against their agreed hash and passes
final exact HF inventory fences for both remote families. A completed no-op
downloads zero shard bodies, fetches the percentile manifest once, and
republishes mirrors idempotently.

### `allow_degenerate_statistics` — Statistics Health Guard

Applies to `compute_statistics` (`configs/extraction/compute_statistics.yaml`).
Default: **`false`**.

When `false` (the default), `compute_statistics` hard-fails with
`DegenerateFeatureStatisticsError` if the aggregated features show the
**random-weight degenerate signature** (only when ≥ 1000 segments have been
processed). The gate fires if **either** of the following is true:

- The median per-feature firing frequency `> 0.5` (the median is the
  interpolated true median, `torch.quantile(freq, 0.5)`, not `torch.median`,
  which returns the lower middle value for even-length inputs); or
- At least 50% of ever-seen features fire in > 99% of segments
  (`n_features_firing_gt_99pct ≥ 0.5 × n_features_seen`).

This guards against producing a stats artifact from a model whose weights were
silently random-initialised — the exact failure mode of the issue #549 Gemma-3
VLM loader bug, where a plain `Gemma3ForCausalLM.from_pretrained` on a
4b/12b/27b checkpoint would silently random-init the text decoder and produce
near-universally-firing SAE features.

Set `allow_degenerate_statistics: true` to bypass the gate. This is the
**intended recovery path for a genuinely low-diversity-but-healthy corpus**
that trips the gate (e.g. a domain-specific corpus where most features really
do fire very often) — not only for intentional degenerate-model testing.

Two new metrics are present in every stats file that is successfully produced,
regardless of the `allow_degenerate_statistics` gate (when the gate fires the
run aborts before the stats file is written, so no file is produced at all):

| Metric | Description |
|--------|-------------|
| `health.median_feature_firing_frequency` | Median per-feature fraction of segments the feature was active; expected ≈ 0.005 for healthy weights, ≈ 1.0 for random-init |
| `health.n_features_firing_gt_99pct` | Number of features active in > 99 % of segments; should be near zero for healthy weights |

These metrics are written to the safetensors metadata under a `health.*`
namespace (`metadata["health.<key>"]`) and the prefix is stripped on read by
`load_statistics`, giving a lossless round-trip.

### Artifact identity caches

Overlay writers produce or consult a **base-shard hash cache** dotfile — a stat-validated (name/size/mtime_ns/ctime_ns/inode) cache of the streaming SHA-256 over base shards. There is no config knob; the cache is always active and self-invalidating.

| Artifact | Location | Writers |
|----------|----------|---------|
| `<feature_dir>/.base_shard_hash_cache.json` | Feature directory | `precompute_percentiles`, `precompute_packed_rows` (train+val), `discover_overlay` |
| `<packed_row_dir>/.base_shard_hash_cache.json` | Packed-row directory | `precompute_selected_features` (hashes packed-row shards, not base shards) |

The cache value is exactly the streaming SHA-256; no overlay tensor, manifest hash, or packed-row content changes. See [`docs/pipelines/feature-extraction.md`](../pipelines/feature-extraction.md) § Base-shard hash cache.

Splice preselection, packed loading, and training also keep additive caches in
the packed-row directory for packed-row, packed-provenance, derived-feature,
and splice-sidecar content fingerprints, plus a packed/provenance
semantic-validation certificate. Each entry records the exact canonical
content digest together with both `lstat` and target `stat` identity (`mode`,
device, inode, size, mtime_ns, ctime_ns). A hit reads no tensor bodies. A
missing, stale, old-version, or malformed cache repeats the existing full
content hash or semantic tensor validation; a file that changes during a cold
validation fails loudly. Writes are atomic and best-effort, so a read-only
cache directory preserves correctness and merely loses the next warm hit.
These dotfiles are not artifacts, never enter selection identity, and have no
config knobs.

### Validate Features Config (`configs/extraction/validate_features.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `feature_dir` | **required** | Path to feature directory |
| `parquet_dir` | null | Source parquet directory for cross-referencing |
| `resolve_parquet_default` | false | When `true` and `parquet_dir` is null, `parquet_dir` is resolved to the nanochat default parquet dir (the same dir extraction uses). Opt-in — default `false` keeps all existing callers byte-identical. |
| `model` | null | Model name (for tokenizer + GPU validation) |
| `n_spot_check` | 5 | 0=CPU-only, 1=first doc, 2=first+last, 3+=length-biased sampling |
| `sae_release` | null | SAE release string (required for GPU spot-checks) |
| `sae_id` | null | SAE id string (required for GPU spot-checks) |
| `backend` | null | Use recorded `extraction_backend`; all-legacy directories fall back to `auto` with a warning. Mixed known/unknown provenance requires an explicit backend for spot-checks. `auto` (nnterp), `nnterp`, or `transformerlens` overrides may disagree with known provenance but warn at startup and completion. Native replay is unsupported; CPU-only validation ignores this key. |

Extraction stamps concrete backend provenance on every new shard. Resume preserves
legacy shards without backfilling metadata; missing provenance warns, while
conflicting known values fail. See [backend provenance and legacy resume](../pipelines/feature-extraction.md#shard-format-csr-sparse)
and [validation policy](../pipelines/feature-extraction.md#validation-config-configsextractionvalidate_featuresyaml).

`validate_features_gemma3_l22_f65k.yaml` inherits from `validate_features.yaml`
and sets `resolve_parquet_default: true` so the full 370-shard source validation
runs without specifying the parquet path explicitly. It is used in Step 0b of
`runs/gemma3_l22_f65k_pipeline.sh`.

### Compact FP Source Validator (`configs/extraction/validate_fp_target_source.yaml`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `packed_row_dir` | **required** | One immutable packed-row shard family whose raw CSR features will supply compact FP targets. |
| `n_sae_features` | **required** | Original SAE dictionary width, not `V_effective`. |

The command streams each shard, validates packed dtypes, CSR structure, bounds,
and exact original-ID uniqueness within every segment, then atomically adds a
versioned `fp_target_source_integrity` certificate to the existing packed
manifest. Any failure leaves that manifest unchanged.

### `feature_mask` — Optional Feature Blacklist/Whitelist

Applies to `configs/training/conditional_encdec_train.yaml` and `conditional_encdec_sft.yaml`. Default: `null` (disabled).

```yaml
feature_mask: null
# feature_mask:
#   mode: blacklist         # "blacklist" | "whitelist"
#   ids: [123, 456]         # inline int list — mutually exclusive with `path`
#   # path: /abs/path/feature_ids.txt
```

**`mode`** (required): `blacklist` forbids the listed IDs; `whitelist` restricts to only those IDs. Both positive selection, negative-sample universe, FP-prediction targets, and self-conditioned feedback are restricted identically.

**`ids` vs `path`** (exactly one required):
- `ids`: inline list of integer feature IDs.
- `path`: absolute path to either a `.txt` file (one integer per line; `#` comments and blank lines are ignored) or a `.yaml`/`.yml` file containing a flat list of strict integers or quoted decimal IDs. Quoted zero-padded IDs from `manual_feature_filter.yaml` are accepted; unquoted zero-padded YAML IDs fail loudly to avoid octal parsing.

**Validation** (all errors are `FeatureMaskError`):
- Both `ids` and `path` present, or neither → error.
- Duplicate IDs → error.
- Any ID outside `[0, n_sae_features)` → error. `n_sae_features` is the **ORIGINAL** SAE dictionary width (16,384 / 65,536 / …), **not** the encoder's compacted table height `V` (`model.feature_vocab_size`) — since #695 those are different numbers. Every id crossing a config or API boundary is an original SAE id.
- Empty resolved set → error.

**No-clobber invariants:**
- `null`/absent mask → byte-identical `sel/<hash>/` (I1), byte-identical checkpoint path (I2), bit-identical RNG/outputs (I5).
- A distinct mask → distinct `sel/<hash>/` and checkpoint leaf `bl_<16hex>` or `wl_<16hex>`.
- Base extraction shards, statistics, pct overlays, token boundaries, and packed rows are NEVER recomputed (I3).

**The mask and resolved clusters are also STRUCTURAL (issues #695/#714).** The encoder's feature-id-indexed tables (`encoder.feature_embed` and every `encoder.value_embeds[i]`) have one row per **effective feature**: mask-allowed ids minus members consolidated into another representative. Compaction is always on whenever that derived universe is smaller than the SAE dictionary; there is no knob.

- Compact feature prediction is supported for precomputed pretraining: the head and dense target use `V_effective`, while all artifact/log/report IDs remain original representatives. Existing packed families must first be certified with `scripts.preprocess.validate_fp_target_source`; live compact-FP loading is rejected.
- A compact-FP checkpoint supports **inference-time self-conditioned generation**: head row `j` is translated to original id `feature_vocab.original_ids[j]` before selection, and the encoder remaps selected original ids at its boundary. Direct `return_fp_logits` and `fp_logits_only` return `V_effective` columns. The packed-pretraining post-training source remains uncompacted-only; source-aware SFT supports exact compact mask/cluster identity and compact-row STE. Universal `feature_gcg` remains supported for unclustered compact checkpoints and accepts static-cluster checkpoints when its entry point receives a validated cluster feature contract: proposals remain original IDs restricted to the effective `V`, compact-row translation is internal, and returned bags retain original representative IDs.
- Pretraining and SFT resume require the same mask **and** canonical cluster content; both are checked before weights load. SFT resolves the explicit base-checkpoint contract and folds cluster siblings inline before its live per-conversation selection. Clustered SFT supports `weighted` / `top_k_percentile` sampling and percentile-domain thresholds only; raw-magnitude threshold types fail at startup because the representative fold has percentiles, not a representative raw magnitude. Both clustered selection arms use full-distribution ranks from the statistics percentile table; thresholding no longer derives a second tail-relative coordinate from bucket boundaries. The chained `run_encdec_sft_pipeline` driver rejects clusters; run clustered phases individually with explicit `base_checkpoint_dir`. Generic checkpoint loaders that still declare a deferred compaction stance can reject clustered checkpoints before any generation adapter is built. Once a compact model is loaded under a valid eval stance, inference-time self-conditioning needs no runtime cluster reduction: absent members have no head row, and the provider emits only original representative ids from the checkpoint vocabulary. Clustered-SFT sampling and thresholding are self-conditioning-compatible because their consolidated full-percentile scores match the FP targets. Source-aware SFT post-training supports the same exact compact cluster/vocabulary contract and compact-row STE; the packed-pretraining source remains uncompacted-only. Cluster-aware evaluation exceptions are AxBench-Base's stage-limited source/Stage-4 orchestration, BPB-Adv's contract-aware dispatcher, and clustered BPB-Adv sample replay. Sample replay retains one source validation, candidate snapshot, and in-memory aggregate result across every requested target instead of reopening those paths after validation. Feature-GCG likewise requires a validated cluster feature contract. BPB-Adv's advertised clustered score set is **end-to-end runnable**: #1008 removed the stale precomputed-bag intervention guard and #1050 made a resumed run reuse committed bags by identity rather than byte-comparing a nondeterministic derive; DiffMean/linear-probe directions now consume exact source-certified bytes. For SFT, the configured effective mask/cluster universe must equal the base checkpoint vocabulary. Recover a mask with `uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>`; on clustered checkpoints the tool also verifies canonical cluster metadata and refuses ambiguous thin metadata.
- Existing legacy **masked but uncompacted** checkpoints cannot resume into a compacted run. A pre-effective clustered checkpoint is a second migration shape: it has cluster metadata but a mask-only vocabulary with rows for consolidated-away members. Current resume detects it before the generic size guard and requires a fresh run; no offline row/optimizer-state migration tool exists.
- I1/I2/I3/I5 all still hold: the vocabulary is a pure derived function of `(feature_mask, canonical feature_clusters, n_sae_features)`. Those inputs already condition the selection hash and the `bl_`/`wl_` plus `fc_` checkpoint leaves, so **nothing new is added to identity**. With no clusters, derivation literally delegates to the old mask-only constructor.

See [encoder-decoder-training.md § Feature Mask](../pipelines/encoder-decoder-training.md#feature-mask-blacklistwhitelist) and [§ Encoder Feature-Dictionary Compaction](../pipelines/encoder-decoder-training.md#encoder-feature-dictionary-compaction) for the full enforcement details.

### `feature_clusters` — Static Preselection-Only Clusters

Default: `null` (disabled). V1 reads a static YAML/JSON definition; it does not
discover clusters and does not restore the removed coactivation/Ising producer
or runtime machinery.

```yaml
feature_clusters:
  path: /abs/or/cwd-relative/clusters.yaml
```

The referenced file must contain exactly these two keys:

```yaml
schema_version: 1
clusters:
  - [12, 43, 91]
  - [7, 104]
```

`clusters` must be a non-empty list. Every cluster is a list of at least two
strict integer IDs in the original SAE range; members may not repeat within a
cluster or overlap between clusters. Duplicate YAML/JSON mapping keys, unknown
keys, booleans/floats/strings as IDs, empty clusters, and unsupported file
extensions fail with `FeatureClustersError`. Member and cluster order are
canonicalized, so equivalent files have the same content identity even when
their paths or source ordering differ.

**Ordering and representative rule.** First resolve the complete explicit or
frequency-produced `feature_mask`. For each source cluster `C`, let `S` be the
members allowed by that mask. If `S` is non-empty, `min(S)` is the run-wide
representative; all other members of `S` are consolidated away. The
representative is not re-chosen per segment. Ordinary preselection applies the
mask and then max-reduces each cluster's normalized percentile. Under
`splice_counterfactual`, apply the mask independently to the current and
would-have-followed segments, map both sides through the same fixed
representatives, max-reduce normalized percentiles separately on each side,
then compute `current_max_pct - reference_max_pct` per representative. The
clustered current percentile supplies buckets/absolute-percentile ranking and
that representative delta supplies `percentile_delta` ranking. A missing
representative is zero/absent exactly as in the unclustered counterfactual
implementation. The generic cluster module has no delta-mode dependency; the
two-side orchestration lives in `delta.py`.

Clustered splice selection adds
`cluster_delta_semantics=mask_then_cluster_each_side_max_percentile_then_delta_v1`
to its selection hash and schema-v2 splice manifest. The signed splice-resume
certificate advances to schema v3 and binds the same key. That identity is
strictly conditional: it is absent for `feature_clusters: null` and for the
delta-free clustered base-validation twin, preserving their selection hashes
and selected-shard bytes exactly. Remote-loader certificates use one current
schema v2 for every clustered family: the delta-free certificate omits the
conditional key, and a former schema-v1 certificate requires one producer
revalidation and certificate rewrite without recomputing selection. A
clustered splice manifest from the former raw-delta-then-max ordering is stale
and is rejected rather than resumed.

Prompt-context caches have no current/reference segment pair and therefore use
an explicitly **delta-free** policy. Copy the training positive sampling policy
but **omit the `delta_weights` key entirely** from the cache config, keep the
training negative selection, and set `feature_delta: null`. The canonical
identity resolver internally reintroduces `delta_weights: null` only in the
sampling-shaped hash payload so it matches the producer's delta-free
base-validation identity; putting that key in the user-facing cache config is
rejected before conversion. Clustered prompt caches support sampling only. Each
raw member magnitude is converted through that member's own statistics row,
sibling normalized percentiles are max-reduced at the post-mask representative,
and the existing sampler/bucketer consumes those representative percentiles
using one ordered real-seed (`conditioning.seed`, the checkpoint's actual
training-preselection seed) RNG stream. Threshold-based clustered prompt caches
fail at startup.

All artifacts and APIs continue to carry **original SAE IDs**. Only
`FeatureEncoder._to_compact` maps a representative to its effective-vocabulary
row. `effective_feature_vocab(feature_universe, n_sae_features)` — handed the
consolidation itself, which carries its resolved mask — contains exactly
the mask-allowed universe minus `consolidated_away`: each surviving cluster's
minimum allowed member remains, every away member is physically absent, and an
away id fails the ordinary out-of-vocabulary guard. With no clusters the function
literally delegates to `FeatureVocab.from_mask`; an identity result normalizes to
the full-width `None` path. Thus clustering changes no upstream id space and adds
no second remap boundary.

#### Cluster-vector SAE initialization

`initialize_encoder_embeddings_cluster_vectors` chooses how SAE-backed init fills
the physical row retained for each static cluster:

- `mean` (default; also selected by an absent key or `null`) averages in fp32 over
  each cluster's ascending mask-surviving members. Mask-excluded members never
  contribute, only the retained representative row is replaced, and a singleton
  survivor is a bitwise no-op. The stage deliberately follows each projection
  mode's existing row-subset boundary:
  - `random`: on the normalized full-dictionary projection, after the global
    rescale and before the final `FeatureVocab.select_rows` gather;
  - `svd`: on oriented full-dictionary `W_dec`, before the initial
    `FeatureVocab.select_rows` subset, covariance, eigendecomposition, projection,
    and rescale. The SVD basis therefore describes the averaged effective
    dictionary, not the representative-only dictionary.
- `representative` is the explicit opt-out and keeps the existing direction for
  the minimum mask-surviving member. It adds no averaging operation to either
  projection path.

The default is removed from checkpoint `user_config` so adding this key does not
change vanilla checkpoint metadata; the resolved W&B config keeps it. All declared
mode/policy rows are legal: with `null` init no SAE table exists, and without
clusters there is nothing to average, so both are inert bitwise no-ops. Before an
active mean path touches `W_dec`, the initializer requires the cluster state to use
the original SAE width and the model's live vocabulary to equal
`effective_feature_vocab(feature_clusters, n_sae_features)`. These shared guards
prevent a mask-only-shaped model from accepting random or SVD cluster means under
the wrong row mapping. Multiple surviving members necessarily compact that
vocabulary, so an uncompacted model cannot coexist with active averaging.

**Supported producer paths:** ordinary `weighted` or `top_k_percentile`
sampling; ordinary `feature_percentile` or
`feature_percentile_segment_topk` thresholds; and splice-counterfactual
sampling with `absolute_percentile` or `percentile_delta` weights. The
completion TUI is the downstream interactive exception: it reads the resolved
mask and static clusters from checkpoint metadata, then reproduces the ordinary
percentile-based selection semantics over base feature shards. It does not read
a `feature_clusters` launch/config key. Sampling-based prompt caches are the
other downstream exception and support in-training CORE/sample generation.
Raw-magnitude thresholds in clustered pretraining or SFT, splice thresholds,
gradient-attribution ranking, live/fused pretraining selection, next-segment conditioning, serving,
pretraining-source on-policy self-conditioning post-training/STE, and feature-GCG without a validated cluster feature contract
fail loudly. Low-level Feature-GCG entry points accept a validated contract, and
AxBench-Base and BPB-Adv resolve and thread that same contract through their
narrow clustered dispatchers. AxBench-Base supports stages `0a`, `0b`, `0c`,
`1b`, `1c`, `2`, `3a`, `3b`, `3e`, and `4`, one at a time, with an explicit
stage-appropriate method allowlist; prompting/LoRA training and stages `5`/`6`
remain unsupported. BPB-Adv's cluster-aware artifact stages are present, and
source-certified clustered direction scoring is enabled, but one stale
precomputed-bag scoring is runnable (#1008, #1050). Ordinary/threshold clustered inline steering and
generic standalone eval paths remain unsupported. Clustered pretraining
requires:

Clustered evaluation uses a stricter config schema than training. The mask and
cluster blocks must be supplied together and must be path-backed:

```yaml
feature_mask:
  mode: whitelist
  path: /abs/path/accepted_feature_ids.txt
feature_clusters:
  path: /abs/path/feature_clusters.yaml
```

AxBench-Base nests these under `conditioning`; BPB-Adv uses the top-level
fields. Both require an explicit non-empty method list. On the shipped main
config, AxBench-Base also requires adding the absent nested cluster key (for a
Hydra CLI override, `+conditioning.feature_clusters.path=...`), selecting one
supported `stage`, and replacing the default `methods=all`. BPB-Adv already
declares both top-level keys as `null`, but its default ten-method list is only a
subset of the fourteen-method clustered allowlist. There are no dedicated
clustered AxBench Hydra presets on main; copy every identity-bearing override
across separately invoked stages.

```yaml
use_precomputed_batches: true
feature_delta: null  # or {mode: splice_counterfactual, neg_delta_ratio: ...}
```

Clustered in-training steering is additionally supported when
`feature_delta.mode=splice_counterfactual`, the derived validation packed rows
and exact clustered `sel/<hash>/` selection/provenance family are complete, and
`steering_eval_every>0`. Clustered threshold steering and ordinary delta-free
clustered steering remain startup errors. The generic encoder-decoder config
defaults `feature_delta` to `prev_segment`, so setting only
`feature_clusters.path` is intentionally insufficient. Conditioned-BPB
validation (`eval_every`) remains supported, but every evaluated split must
have complete clustered packed-row and `sel/<hash>/` artifacts. Positive
`core_metric_every` or `sample_every` additionally requires
`conditioning.context_cache_dir` and an identity hash exactly derived from the
top-level training selection (with `delta_weights: null` retained only in the
internal canonical hash payload, not the cache's input config),
negative selection, the real training-preselection seed, bucket count,
resolved mask, and canonical cluster descriptor -- statistics file SHA-256 is
**not** part of this hash; it is independently pinned by the clustered
feature contract and cache manifest instead. The cache manifest binds that
identity plus the full cluster descriptor, effective-vocabulary fingerprint,
`cluster_selection_semantics=post_mask_representative_max_percentile_v1`, and
`feature_delta=null`. Cached rows remain original representative IDs; provider
startup rejects any masked or consolidated-away ID before building entries.

The clustered inline evaluator does not create a persistent steering cache. At
training startup it validates the derived/spliced validation manifest and
content aggregate, packed fingerprint, provenance, splice sidecars, derived
features, token boundaries, mask, cluster descriptor, and effective vocabulary.
It then performs a bounded two-pass scan: pass 1 deterministically selects
positive-bearing target documents/segments and fixed different-document donors;
pass 2 joins only their required prefixes to the derived `FeatureIndex`. All
cached conditioning IDs remain original representatives. During probing,
representative raw activations retain their single-feature scale, while the
primary percentile score is the maximum percentile over every mask-surviving
member of that representative. DDP ranks all-gather the startup digest and use
a status gather before score gathering so ordinary evaluation failures raise
symmetrically.

The path is provenance, not identity. Selection hashes conditionally include a
strict descriptor containing schema version, original SAE width, canonical
content SHA-256, and the versioned consolidation semantics. Clustered
checkpoints append `fc_<full-64-hex-descriptor-sha>` and persist both descriptor
and canonical content. Selection manifests persist those values, mask-derived
counts, source-path provenance, an `in_progress`/`complete` state, and—when
complete—the exact ordered compressed selection-shard names, lengths, and
SHA-256 fingerprints. Producers and loaders validate exact packed-shard
coverage and output fingerprints plus CSR/dtype/channel/bucket integrity and
representative-only positive/negative IDs before accepting
`complete`. `null`/absent clustering adds no hash key, path leaf, checkpoint
metadata, tensor operation, or RNG draw.

The `fc_` leaf and metadata already pin the model-shaping cluster content. A
checkpoint written before effective-feature compaction used the same leaf but a
mask-only table; resume identifies that exact shape and raises a dedicated
"pre-effective-compaction" diagnostic. Start fresh: reshaping every feature-indexed
parameter and optimizer state is possible in principle but no repository tool does it.

See [encoder-decoder-training.md § Static Feature Clusters](../pipelines/encoder-decoder-training.md#static-feature-clusters-preselection-only) for the dataflow, negative-sampling contract, and operational sequence.

### `force_cluster_shard_revalidation` — Clustered Evaluation Proof Path

Default: `false`. Available on AxBench-Base and AxBench-BPB-Adv, including
BPB-Adv sample generation through its shared preflight. `true` is rejected
unless `feature_clusters` is active. The false default is omitted from the
legacy config serializers, so existing run-tag and profiler-config bytes remain
unchanged; an explicit true value is recorded.

Normal clustered startup requires either the schema-v1
`clustered_selection_validation` producer certificate or the exact complete
legacy clustered-splice schema-v3 proof. It always hashes each compressed
selection file against the certified inventory, but a valid stat-certified
`.packed-row-fingerprint-cache.json` skips the packed body hash and redundant
packed/selection tensor decompression. Missing, stale, malformed, or unreadable
cache data is recomputed from packed bytes. Cache writes are best effort: an
unwritable cache does not prevent validation. A cache records packed content
identity independently of selection validation, so it may remain after an
unrelated manifest or selection check fails. Actual packed source changes and
selection validation errors still fail preflight.

`force_cluster_shard_revalidation=true` ignores the cache, hashes and executes
every canonical packed/selection body validator from one exact compressed read
per packed shard, and optionally refreshes the same cache.

Under DDP, rank 0 owns cold/forced content validation and is the only shared
cache writer. Peers resolve manifests, selection bytes, signatures, cache, and
the final contract independently after rank 0 publishes; a peer on a stale
mount may hash locally but never writes the shared cache. All ranks gather
errors and fingerprints through the same collective sequence.

The resulting `CheckpointSelectionContract`, full feature-contract document,
semantic/provenance JSON, and fingerprint are identical on certified-fast and
forced paths. `feature_contract.json` remains the immutable run-ownership
marker, and `expected_feature_contract_*` remains source/target binding; neither
is a validation cache.

### `feature_selection.gradient` — Gradient-Attribution Ranking (offline-only)

Optional block on `feature_selection` (training + selection-precompute
configs). Its PRESENCE switches the segment ranking key from activation
percentile to precomputed gradient-attribution scores; absent/null keeps the
legacy behavior and a byte-identical selection hash (I1).

| Key | Default | Description |
|-----|---------|-------------|
| `mode` | required | `future_half_split` \| `future_self_full` \| `self` — which loss window each segment attributes (and how the doc-final unit is treated) |
| `model_dtype` | `bfloat16` | dtype of the overlay producer run (`bfloat16` \| `float32`); part of the overlay identity |
| `min_final_segment_tokens` | `8` | doc-final segments shorter than this merge with the second-to-last segment into one attribution unit; **must be >= 2** (a 1-token final is unscoreable, so it always merges) |

Validity: requires `mode: top_k_percentile`; incompatible with
`delta_weights`, `feature_delta`, and threshold-type selections; requires
precomputed packed rows + sel shards for train AND val (live selection paths
fail fast). Producer configs:
`configs/extraction/precompute_gradient_attribution.yaml` (GPU overlay; adds
`attribution_mode`, `batch_size`, `device`, `resume`/`force`/`dry_run`).
Packed-segment provenance is written natively by `precompute_packed_rows`
(no separate config). Full pipeline:
[gradient-attribution-selection.md](../pipelines/gradient-attribution-selection.md).

### `feature_mask_freq_min` / `feature_mask_freq_max` — Auto Activation-Frequency Blacklist (producer)

Applies only to `configs/training/conditional_encdec_train.yaml`. Defaults: **`freq_min: 1e-5`** (kills dead features and other features too rare for the autointerpretability judge to score — `count <= 1e-5 * n_segments`) and **`freq_max: 0.5`** (kills features active in >50% of segments). Siblings of `feature_mask`. Encoder-decoder SFT is a checkpoint-universe consumer and deliberately has no frequency producer; pass the exact checkpoint mask explicitly.

> **The metric is fraction of SEGMENTS active, not fraction of tokens.** Extraction pools the activations of `segment_size` tokens into a single vector per segment (`max_pool` / `mean_pool` / `last_token`) before SAE encoding, so what the statistics artifact records — and what these bounds consult — is "pooled-active in this segment yes/no", not per-token firings. A per-token interpretation is not recoverable from the existing artifacts; if you need it you must re-run extraction without pooling. See [feature-extraction.md](../pipelines/feature-extraction.md).

```yaml
feature_mask_freq_min: 1e-5  # null or float in [0, 1); blacklist count <= min*n_segments
feature_mask_freq_max: 0.5   # null or float in (0, 1]; blacklist count > max*n_segments
feature_mask: null
```

> **Encoder compaction (#695/#714) is not optional.** Whenever these bounds resolve a `feature_mask`, the encoder and enabled FP head are sized to that allowed universe, with any static-cluster away members removed as well. There is no knob that turns this off, and **nulling these bounds is not a workaround for it**. Inference-time self-conditioned generation supports the compact head by translating candidate rows back to original ids. Source-aware SFT post-training supports exact compact mask/cluster identity and compact-row STE; the packed-pretraining source and other explicitly unsupported consumers still fail loudly. Universal `feature_gcg` supports unclustered compacted checkpoints in original-ID space with proposals restricted to `V`; a static-cluster checkpoint additionally requires a validated cluster feature contract. AxBench-Base and BPB-Adv can resolve that contract, subject to their documented stage/method boundaries and BPB-Adv's current intervention blockers. Nulling the band or dropping clusters for another unsupported surface does not fix it: it silently changes the experiment. An honest crash beats an invisible regression.

**KEEP iff `freq_min < freq <= freq_max`** (open below, closed above). The producer for `feature_mask` reads the precomputed **`feature_statistics.safetensors`** and blacklists every SAE feature whose per-segment activation frequency falls *outside* the kept interval, using the **exact integer** `count` and `n_segments` from the stats (never the lossily-rounded `pct_segments_active`). The generated id list is fed into `feature_mask` as `mode: blacklist, path: <generated file>`, so all `feature_mask` enforcement/no-clobber semantics above apply unchanged.

- **Source = the statistics artifact, not a fresh shard scan.** `compute_statistics` already aggregates per-feature segment-fire counts and persists `n_segments`. **The blacklist's train/val split is therefore exactly the one `compute_statistics` was run with** (`exclude_last_shard: true` by default ⇒ validation shard excluded). Run statistics computation first; the `meta.json` sidecar records `stats_generated_at` for audit.
- **Bounds semantics.** `freq_min` is INCLUSIVE (count ≤ min·n_segs ⇒ killed); `freq_max` is EXCLUSIVE (count > max·n_segs ⇒ killed). Even at `freq_min = 0.0`, dead features (`count == 0`) get killed: `0 ≤ 0·n_segs`. Either bound can be `null` to skip; **both `null` ⇒ feature fully disabled** (byte-identical legacy behavior, I1/I2/I5 — and no encoder compaction).
- **Validation.** `freq_min ∈ [0, 1)`; `freq_max ∈ (0, 1]`; if both set, `freq_min < freq_max` strictly (else the kept interval is empty). Out-of-range / non-numeric / `NaN` / `bool` ⇒ `FeatureMaskError`. The old single key `feature_mask_freq_threshold` is renamed to these two and is no longer recognized.
- **Resolution & precedence:** an explicit `feature_mask` always wins — the auto-blacklist is skipped with a warning. An *empty* computed blacklist is a no-op (`feature_mask` left unset; byte-identical).
- **Artifact:** `{feature_dir}/blacklist/freq_min{m}_max{M}.txt` + `.meta.json` (bound-keyed; the statistics file is the single immutable input). Generated DDP rank-0 + atomically; (re)written only when missing or the deterministic content would change (stats recompute / bound change / generator-version bump auto-corrects; an identical re-run never rewrites).
- **The bounds never enter any selection hash** — only the resolved id set does (via the existing `feature_mask` payload). Two bound-pairs that yield the same id set produce an identical `sel/<hash>/` and checkpoint leaf.
- **Resume / eval:** keep both bounds + the statistics constant to resume the same checkpoint. To evaluate a freq-blacklist-trained model, point `feature_mask.path` at the generated `freq_min{m}_max{M}.txt`, or re-run `scripts/preprocess/precompute_feature_blacklist.py` with the same bounds (the file is deterministic). **Since #695 the generated `.txt` also defines the encoder's embedding-table shape**, so a `feature_dir` move or a statistics recompute that changes it makes the checkpoint's mask unrecoverable from the config alone. The escape hatch is `uv run python -m scripts.analysis.dump_feature_mask <checkpoint_dir>`, which recovers the exact mask from the checkpoint itself.
- **Compaction is not optional.** A non-empty computed blacklist compacts the encoder and any enabled FP head (#695/#768). Compact FP pretraining is supported from certified packed rows, and inference-time self-conditioned generation consumes compact head rows in original-ID space. Source-aware SFT post-training supports exact compact mask/cluster identity and compact-row STE; the packed-pretraining source and legacy-masked resume remain unsupported. Universal `feature_gcg` is supported for N=1 and larger groups; it keeps public bags in original-ID space and restricts compacted proposals to `V`. Static-cluster checkpoints additionally require a validated cluster feature contract at the low-level Feature-GCG entry points. AxBench-Base and BPB-Adv resolve that contract only within their documented clustered subsets; BPB-Adv scoring still has the two intervention blockers described above. Do **not** null the band to make another unsupported surface run — it does not fix the gap, it silently forfeits the saving. Take the crash.
- **Reproducing pre-`1e-5` runs.** `freq_min`'s default moved `0.0 → 1e-5` in 2026-07. Any run that relied on the implicit default before that date used `0.0`; to reproduce one exactly, set `feature_mask_freq_min: 0.0` explicitly (still fully supported — kills only `count == 0` dead features) rather than relying on the current default.

> #### MIGRATION: the freq default changes re-key your artifacts, and a pre-#695 run cannot be resumed
>
> Both defaults moved (#702/#695): `feature_mask_freq_min` `0.0 → 1e-5` and `feature_mask_freq_max` `0.99 → 0.5`. These are deliberate changes to the selection policy, not bugs. But per the bullet above, the bounds never enter a hash — **the resolved id set does**, and the old and new bands do not resolve the same one. On the real 65k corpus `(1e-5, 0.5)` blacklists 3,701 of 65,536 (`V = 61,835`); a `0.99` upper bound blacklists a strictly smaller, **different** set. A different id set means a different canonical `feature_mask` payload, and therefore:
>
> - a different **`bl_<16hex>` checkpoint leaf** (`feature_mask_dirname` = sha256 of the canonical mask), and
> - a different **`sel/<hash>/`** (the mask payload is an input to `compute_selection_hash`).
>
> **A run started on the old defaults does not resume on the new ones** — it looks for its checkpoint under a leaf that holds none — **and re-pinning the old band does not rescue it either.**
>
> The leaf is not the obstacle. Both the leaf and the `sel/<hash>/` are pure functions of the resolved id set, so pinning the old bounds (**both** of them — `freq_min` moved too, so `feature_mask_freq_max: 0.99` alone reproduces neither the old band nor the old id set) does put you back on the old leaf and the old selection artifacts. **The encoder shape is the obstacle.** Compaction arrived *with* #695, so any checkpoint trained on the old defaults predates it and is physically uncompacted: full-width `encoder.feature_embed`, and no `encoder.feature_vocab_ids` / `feature_vocab_n_sae` buffers. The current pretraining code compacts the encoder for **any** mask that resolves, so `assert_checkpoint_vocab_compatible` raises before its `load_state_dict`. SFT rejects the same semantic mismatch in its effective-universe equality guard before model construction. There is no reshape tool.
>
> Do **not** null the band to get past that error. It is not a fix: a no-mask run resolves a *different* `sel/<hash>/` (the unmasked hash) and an *un-leafed* checkpoint path, so it does not continue that run at all — it conditions on the full unrestricted universe, over encoder rows the blacklisted run never trained. That is a different experiment wearing the old run's name, and it forfeits compaction on top.
>
> **What you can actually do: start the run fresh on the current defaults.** Nothing is deleted — the old leaf's checkpoints and the old `sel/<hash>/` artifacts are **orphaned**, not removed, and a fresh run simply builds its own alongside them (at full precompute cost, and without telling you the old ones are there). `scripts/analysis/dump_feature_mask.py <checkpoint_dir>` recovers the exact mask a given checkpoint was trained with, if you need to identify which band produced it.

Pre-generate explicitly with `uv run -m scripts.preprocess.precompute_feature_blacklist feature_dir=... feature_mask_freq_min=1e-5 feature_mask_freq_max=0.5` (config `configs/extraction/precompute_feature_blacklist.yaml`).

### BPB Representative Coefficient

BPB and BPB-Adv collapse coefficient sweeps to one representative coefficient
per `(mode, method, target_model, adversary, context_signal_mode)` for aggregate
tables, plots, and sample generation. This is not configurable: the
representative coefficient is always the one whose no-adversary steerability is
closest to the scored fair encoder-decoder Feature-DiffMean conditioning-bag
baseline (`feature_diffmean_encdec` / `neg_feature_diffmean_encdec`) for the
same mode/adversary/context tuple. Exact closest-score ties choose the higher
steering score, then smaller `abs(coeff)`, then lexicographic coefficient
string. Missing or null fair Feature-DiffMean target rows are hard errors.
In BPB-Adv, `representative_target_methods` can list fair Feature-DiffMean
methods that should be scored only at no-adversary control rows for selector
targets, without adding those methods to the full optimized `base_methods` grid.
When the adversary grid has no explicit zero-strength entry, BPB-Adv raises
instead of synthesizing one for these selector targets.

For a fresh BPB-Adv `stage=all` run, configuration preflight rejects any
selected mode whose fair target method is absent, before model initialization or
expensive scoring. Add the exact fair method(s) to `base_methods` or
`representative_target_methods` and include `encdec` in `target_models`. A
decoder-only run may still stop at `score` or `aggregate`. Direct `stage=plot`
is validated against the actual macro artifact instead of being rejected from
the replay config alone; valid hidden fair target rows therefore remain
replot-compatible, while missing or null rows fail before prior plot output is
cleared. The standalone sample-generation consumer is intentionally not subject
to the fresh-run `stage=all` preflight because it replays a completed run.

BPB-Adv feature-DiffMean bag size and held-out split reuse the `ours_gcg`
top-k/split knobs so deterministic conditioning-bag baselines, fair feature-GCG,
and roofline oracle feature-bag methods are evaluated on the same held-out prefix
split.
The held-out split is also the universal BPB-Adv class-margin evaluation
population: `prepare_attack_rows` emits rows only for held-out prefixes and
records the exact held-out rich/avoid pair keys in `evaluation_population.json`;
candidate, score, and general-val score stages enforce that artifact for every
method, including decoder-only baselines.
Plain BPB has no held-out adversarial prefix axis, so it uses the separate
`feature_diffmean.top_k_pos` / `feature_diffmean.top_k_neg` overrides and derives
from all accepted source rows for each `(concept, mode)` with both
`concept_rich` and `concept_avoiding` rows. One-sided groups cannot define a
DiffMean contrast, so plain BPB raises before scoring rather than silently
hiding incomplete Feature-DiffMean inputs.

### Feature-GCG typed core domains

The reusable Feature-GCG core declares unary value domains with
`phantom-types` annotations and enforces them at public and execution helper
boundaries with `beartype`. Builtin values are retained; the annotations do not
coerce or wrap caller data. Strict integer domains exclude `bool`, and numeric
domains reject non-finite values.

| Surface | Domain |
|---|---|
| `FeatureGcgConfig.max_steps`, `candidate_batch_size`, `top_k`, `minibatch_size`, `max_forward_batch`, `max_k_pos`, `max_k_neg` | Strict integer `> 0` |
| `FeatureGcgConfig.min_k_pos`, `min_k_neg` | Strict integer `>= 0` |
| `FeatureGcgConfig.temperature` | Finite non-boolean number `> 0` |
| `FeatureGcgConfig.reg_lambda` | Finite non-boolean number `>= 0` |
| `FeatureGcgConfig.master_seed` | Strict integer, with no range restriction |
| `FeatureConditioning.entries` | Tuple of unique `(feature_id, bucket)` pairs; every element is a strict integer `>= 0`; the tuple may be empty |
| Conditioning mutation `fid` / `bucket` arguments | Strict integer `>= 0` |
| Candidate ranking `top_k` / `distance_weight` | Strict integer `> 0` / finite non-boolean number `>= 0` |
| Candidate and edit-space sampling `batch_size` / `temperature` | Strict integer `> 0` / finite non-boolean number `> 0` |
| Batched masked-loss `max_forward_batch` | Strict integer `> 0` |

Direct calls that violate these unary domains raise the corresponding
`beartype` parameter violation. Cross-value and structural contracts remain
explicit Feature-GCG checks: `max_k_{pos,neg} > min_k_{pos,neg}`, mutation slot
upper bounds, duplicate IDs introduced relative to an existing bag, nonempty
candidate pools, tensor dimensions, and aligned collection lengths. The
conditioning-entry phantom predicate checks the complete tuple rather than a
sampled generic-container element. Other generic containers retain their exact
static element types but use non-sampling runtime bases, because beartype's
default element sampling consumes Python's global RNG and Feature-GCG must not
perturb that state.

### AxBench-BPB typed configuration domains

Plain AxBench-BPB validates unary configuration domains at dataclass
construction with reusable `phantom-types` predicates enforced by `beartype`.
The Hydra loader wraps constructor violations in `AxBenchBpbError` naming the
offending nested block. Direct library construction raises the corresponding
`beartype` violation.

| Config surface | Domain |
|---|---|
| `context_cache.extraction_batch_size`, `segment_size` | Strict integer `> 0` |
| `context_cache.num_buckets` | Strict integer `>= 2` |
| `aggregation.min_rows_per_side` | Strict integer `> 0` |
| `aggregation.invalid_warning_threshold`, `invalid_drop_threshold` | Finite non-boolean number in `[0, 1]` |
| `aggregation.primary_metric` | One of the four `LeaderboardMetric` values |
| `feature_diffmean.top_k_pos`, `top_k_neg` | `null` or strict integer `> 0` |
| top-level `device_batch_size` | Strict integer `> 0` |
| top-level `diffmean_and_probe_layer` | Strict integer `>= 0` |
| top-level `feature_extraction_model_id` | `null` or non-blank string |
| top-level `stage` | `prepare_context_cache`, `bpb_score`, `bpb_aggregate`, or `all` |

The aggregation relationship remains explicit:
`invalid_warning_threshold <= invalid_drop_threshold`. Stage validation still
normalizes the selected leaderboard metric to its enum, rejects the currently
unsupported next-segment conditioning mode, and validates the Stage 0-3 source
artifacts before loading models.

### AxBench-BPB-Adv typed configuration domains

AxBench-BPB-Adv validates unary configuration domains at dataclass construction
with reusable `phantom-types` predicates enforced by `beartype`. Hydra sequence
containers are normalized from lists to tuples at the loader boundary, but
their elements are never converted with `int()`, `float()`, or `str()`. A value
with the wrong runtime type therefore fails loudly instead of being truncated
or otherwise reinterpreted. In particular, `bool` is not accepted as an
integer.

| Config surface | Domain |
|---|---|
| `faster_gcg.optimized_token_counts` | Non-empty tuple of unique strict integers `>= 0` |
| `faster_gcg.max_steps` | Strict integer `>= 0` |
| `faster_gcg.candidate_batch_size`, `top_k`, `universal_minibatch_size` | Strict integer `> 0` |
| `faster_gcg.candidate_eval_device_batch_size` | `null` or strict integer `> 0`; `null` inherits top-level `device_batch_size` |
| `faster_gcg.distance_weight` | Finite non-boolean number `>= 0` |
| `faster_gcg.temperature` | Finite non-boolean number `> 0` |
| `gepa.metric_call_budgets` | Non-empty tuple of unique strict integers `> 0` |
| `gepa.k_editable_tokens` | Strict integer `> 0`; `reflection_model` must be non-blank when GEPA is enabled |
| `additive_direction.families` | Non-empty unique tuple drawn from `diffmean`, `linear_probe`, `lora`, `ours` |
| `additive_direction.magnitudes` | Non-empty tuple of unique finite non-boolean numbers `>= 0` |
| `additive_direction.counter_direction_family` | `diffmean` or `linear_probe` |
| `ours_gcg.reg_lambda_grid` | Non-empty tuple of unique finite non-boolean numbers `>= 0` |
| `ours_gcg.objective_alpha_grid` | Non-empty tuple of unique finite non-boolean numbers `>= 0` (default `[1.0]`). Selecting an `ours_gcg_alpha*` method additionally requires `reg_lambda_grid == [0.0]` |
| `ours_gcg.top_k_pos`, `top_k_neg` | `null` or strict integer `> 0` |
| `ours_gcg.split_ratio` | Finite non-boolean number strictly between `0` and `1` |
| `ours_gcg.max_steps` | Strict integer `>= 0` |
| `ours_gcg.minibatch_size`, `mean_g_steer_batch`, `max_forward_batch` | Strict integer `> 0` |
| `roundtrip_generation.samples_per_candidate`, `top_k_sampling` | Strict integer `> 0` |
| `roundtrip_generation.temperature` | Finite non-boolean number `>= 0` |
| `general_val_bpb.max_interventions`, `seed`; top-level `pair_subsample` | Strict integer `>= 0` |
| top-level `diffmean_and_probe_layer` | Strict integer `>= 0`; target residual layer, never the Gemma SAE layer |
| `selection_splice_batch_size`, `feature_diffmean_extraction_batch_size` | `null` or strict integer `> 0` |
| `selection_manifest_path`, `feature_extraction_model_id` | `null` or non-blank string |

Seeds remain strict integers without a sign restriction unless the table says
otherwise. The conditional general-validation rule is unchanged:
`n_documents`, `prefix_tokens`, and `output_tokens` must be strict integers and
must be positive only while `general_val_bpb.enabled=true`. Cross-field rules
such as the Faster-GCG token-count/prefix-length relationship remain explicit
in `validate_config`.

Faster-GCG and GEPA objective loss evaluation use exact compatibility-grouped,
call-local shared-prefix KV reuse. The full-sequence implementation is retained
only as a test oracle, not as a production configuration surface. Gradients
remain uncached. The scoped batch override affects only Faster-GCG candidate
loss evaluation, so it cannot silently change batch composition in gradients,
GEPA, final scoring, or general validation. Select a fresh `run_tag` for the
shared-prefix implementation and whenever changing candidate batch shape;
objective floats, candidate rankings, and retained triggers may differ across
attention backends and batch shapes. The fixed `shared_prefix_kv` algorithm
recorded in run metadata is therefore a search-semantics version, not an opt-in
mode or a claim of legacy full-sequence trajectory identity.

Faster-GCG group checkpointing is intentionally not a configuration surface.
On the production model-backed `build_candidates` path it is always enabled
when at least one non-control Faster-GCG group exists; the model-free
`embedding_weight` test path and runs without such groups create no manifest.
Semantic invalidators include source/attack/population/cluster identity, the
complete `faster_gcg` block, objective-owned batch/layer/conditioning/LoRA
settings, resolved checkpoint bytes and effective loaded metadata, tokenizer and
statistics contents, applicable context/bag trees, and expected group keys.
Operational settings such as device/offload policy, profiling, logging cadence,
plotting, GEPA/general-validation settings, output path, and `run_tag` are not
fingerprint fields. The numeric runtime fingerprint is recorded per shard and a
change warns without invalidating; exact resumed parity is promised only on the
same numeric runtime.

### DiffMean / linear-probe direction identity

AxBench-Base, plain BPB, and BPB-Adv share one direction-artifact validator.
`layer_hooks.diffmean_and_probe_layer` in Base and
`diffmean_and_probe_layer` in BPB/BPB-Adv always name a block in the target
nanochat or encoder-decoder residual stack. They never name the Gemma SAE
extraction site. Thus the Gemma-3 L22 source presets use target layer 6.

Each direction's canonical `concept_id`, `mode`, `target_model`, and `layer`
metadata must agree with its family/path coordinates and the requested config;
the `direction` must be a non-empty finite one-dimensional floating tensor.
Cluster-stamped artifacts also require the family-specific `artifact_kind`.
The three evaluators resolve every selected direction before the first
generated/scored output and repeat validation at the loader boundary. Plain
BPB and BPB-Adv complete this preflight before target-model loading; Base Stage
4 completes it before its first forward/output after the stage model is ready.
LoRA stays outside this contract because its layer-specific tensor keys already
fail loudly on a layer mismatch.

BPB-Adv source contracts are schema v6 when unclustered and v7 when clustered.
Both hash `direction_training_layer`, which is the single layer shared by every
present DiffMean/probe artifact or `null` for a directionless source. A runtime
that selects a scalar baseline or nonzero additive counter-direction rejects
`null` and proves each required coordinate exists. Current contracts are
immutable within a run. Production paths reject v4/v5; use a fresh run tag for
new output.

### `profile` — AxBench Production Profiling

Applies to AxBench-Base and AxBench-BPB-Adv eval configs. The production root
YAML files intentionally do not list this block, so profiling stays absent from
the default config surface. To enable it from the CLI, use Hydra add syntax such
as `+profile.enabled=true`, or keep this block in a local-only YAML edit:

```yaml
profile:
  enabled: false
  output_dir: null
  synchronize_cuda: true
  gpu_utilization: true
  gpu_poll_interval_ms: 50
  gpu_idle_threshold: 5.0
```

When `enabled=true`, rank 0 binds `AxBenchProfiler` inside the real eval CLI
after `compute_init`; nonzero ranks still participate in distributed stages but
do not write profile files. `output_dir=null` writes to
`<output_root>/<run_tag>/profile/`. The profiler writes `events.jsonl`,
`summary.yaml`, and `summary.csv` with region timing, CPU RSS, CUDA memory, GPU
utilization, and stage counters. Benchmark wrappers still exist for lightweight
smoke tests, but production performance numbers should come from this block.

### Target-aware CORE batching

`core_example_batch_size` applies to `configs/eval/base_eval.yaml`,
`configs/eval/dclm_eval.yaml`, and the CORE branch of
`configs/eval/training_progression.yaml`; the default is `1`. It must be a
strict positive integer. The evaluator renders examples independently, then
flattens their completion rows into one right-padded forward while preserving
each row's exact target mask and context-cache key. Decoder-only and Hugging
Face models continue to use nanochat's per-example CORE evaluator.

Batch 1 preserves the pre-batching BF16 logits, losses, task scores, and active
decoder-attention implementation. Values greater than 1 are explicit
throughput-only opt-ins: changing the GEMM and attention batch shapes can alter
BF16 logits and task decisions even when every row, target mask, and context key
is unchanged. Do not compare scores across different values as if they shared a
numerical scoring contract.

### Harmfulness judge settings

Applies to `configs/eval/harmfulness_eval.yaml` and the corresponding
`harm_*` keys in `configs/eval/training_progression.yaml`.

| Key | Default | Description |
|-----|---------|-------------|
| `source` | `null` | Native nanochat checkpoint source. Mutually exclusive with `hf_checkpoint_dir`; either selector may be omitted when an explicit `responses_file` is judged. |
| `hf_checkpoint_dir` | `null` | One concrete local repository SFT export named `step_NNNNNN`. The evaluator does not accept Hub IDs, URLs, mutable parent directories, or latest-step lookup. HF mode rejects native checkpoint, feature-selection/mask, context-cache, static-feature, and non-default next-segment settings before weights or the judge client are constructed. |
| `judge_model` | `openai/gpt-5.6-luna` | Standalone OpenRouter model identifier for both isolated harmfulness and compliance requests. Explicit overrides remain supported and invalidate mismatched result/cache provenance. |
| `harm_judge_model` | `openai/gpt-5.6-luna` | Training-progression equivalent of `judge_model`. |
| `requests_per_minute` | `2000` | Standalone shared rate limit across both request kinds. Progression uses `JudgeClient`'s `500` default. |
| `cache_path` | `null` | Optional SQLite path. Harmfulness retains its legacy model/pair cache key; compliance uses a disjoint versioned request-complete namespace. |
| `harm_num_harmful` / `harm_num_benign` | `50` / `50` in progression | Per-seed pool sizes. They are part of progression resume identity. |
| `batch_size` | `8` | Native decoder-only outer prompt batch. Encoder-decoder self-conditioning requires `1`. |
| `repository_hf_batch_size` | `64` | Repository-HF outer prompt batch, separated from `batch_size` so its measured-safe higher occupancy does not change native/self-conditioned memory behavior. Repository-HF prompts are left-padded and decoded in one Transformers call; a matrix-wide multinomial uses one generator seeded with `seed + batch_start_index`. The resolved backend batch size and HF batched-algorithm seed descriptor are response-cache identity because both BF16 batch shape and RNG consumption can change sampled tokens. |

Standalone `harmfulness_eval.feature_configs` must be `null`; hand-supplied
sparse maps, including empty maps, are rejected. Encoder-decoder generation
requires `batch_size: 1` and `conditioning.context_cache_dir`, and uses the shared
self-conditioning runtime. `self_conditioned.safety_steering: null` selects pure
FP feedback; `{targets: PATH, strength: NUMBER}` selects a validated dense safety
bundle. Strength is finite in `[0, 6]`, with interpolation weight `strength / 6`.
Fixed and turn checkpoints use their authenticated cadence and cache contracts;
`self_cond_segment_size` is only an optional fixed-cadence equality assertion.
Decoder-only evaluation retains its existing generation path. Other evaluation
surfaces keep their existing bucket-valued `feature_configs` behavior. See
[harmfulness interpolation](../pipelines/harmfulness-interpolation.md) for fitting
and checkpoint-specific progression overrides.

`conditioning.identity` fields are optional exact assertions against the
checkpoint-derived cache identity, for both fixed and turn checkpoints. Null
fields derive from the checkpoint; supplied model/SAE/layer/pooling/hash values
must match before generation.

Standalone response schema v2 binds the exact model/tokenizer content digests,
resolved native checkpoint or HF export step, condition label, effective
device/dtype, dense safety target digest and strength when configured, resolved target-conditioning mode,
normalized generation scalars, chat rendering, and the absolute-prompt seed
schedule for native checkpoints. Repository-HF generation uses the distinct
`base_plus_batch_start_hf_batch_v2` descriptor, so earlier HF response caches
fail identity comparison and regenerate. Schema-v1 response
files are rejected and must be regenerated.
Result schema v4 copies that same generation block before it can be reused.

Benign compliance is derived solely as `not is_refusal`. Helpfulness remains a
required 1--5 diagnostic, but there is no helpfulness threshold and no runtime
threshold knob. Compliance uses a versioned, request-complete cache identity;
changes to its prompt or result semantics require result-schema and cache-
namespace version bumps so old judgments cannot be silently reinterpreted.

### `roundtrip_steering` — AxBench-BPB-Adv SAE Probe

This block is specific to AxBench-BPB-Adv round-trip scoring. The generic
harmfulness, chat, response-dump, and progression round-trip paths were
removed; those eval surfaces no longer accept this block.

```yaml
roundtrip_steering:
  enabled: true
  feature_statistics_path: null
  feature_chunk_size: 512
  probe:
    model_name: gemma-2-2b
    sae_release: gemmascope
    activation_type: residual
    layer: 8
    sae_width: 16k
    l0: null
    device: auto
```

The block selects the probe model/SAE, optional percentile-statistics
artifact, and SAE matmul chunk size. With `steering_metric=roundtrip`,
AxBench-BPB-Adv requires `enabled=true` and probes each candidate's
`Concept.feature_index`. `device=auto` chooses CUDA when available,
otherwise CPU. The probe identity must match the extraction identity;
the evaluator validates the model, layer, SAE width, and supported L0
tier before loading target models. If `feature_statistics_path` is null,
the evaluator falls back to the top-level statistics path for percentile
scoring only.

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Required when `steering_metric=roundtrip`; disable it only for the legacy BPB-margin metric. |
| `feature_statistics_path` | `null` | Optional statistics override used to build percentile lookup rows. Null falls back to the top-level statistics path. |
| `feature_chunk_size` | `512` | Number of target features per SAE matmul chunk. AxBench currently probes one concept feature per row. |
| `probe.*` | see YAML | Probe model and SAE identity passed to `steering.eval.probe_model.load_probe_model`. |

AxBench-BPB-Adv adds two sibling settings:

```yaml
steering_metric: roundtrip        # roundtrip | bpb_class_margin
roundtrip_generation:
  samples_per_candidate: 4
  temperature: 0.7
  top_k_sampling: 50
  master_seed: 42
```

`steering_metric=roundtrip` is the default y-axis. It generates steered
completions for each fixed adversarial candidate and a paired per-strength
control completion, then probes the original concept feature in both. Each
sampled completion uses the generation budget computed before sampling. The
mode-appropriate continuation (`concept_rich` for positive rows,
`concept_avoiding` for negative rows) is decoded in the context of the source
prompt tokens, segmented in extraction-token space with
`context_cache.segment_size`, and mapped through character offsets back into
target-model output tokens. The score artifact records that final budget as
`reference_segment_token_count`. `steering_metric=bpb_class_margin` preserves
the legacy signed BPB class margin; candidate construction and Faster-GCG/GEPA
optimization remain BPB-based in both modes.

Residual-direction scaling is not configurable and never reads either
statistics path. Each applied DiffMean/LinearProbe artifact supplies its own
`max_activation`, defined at Stage 3 as the maximum signed continuation-token
`h_token · d` for unit direction `d` across rich and avoiding fit rows at the
target hook layer (float32 CPU, no ReLU or absolute value). Missing, invalid,
negative, zero-inconsistent, or wrongly bound metadata raises before scoring;
there is no statistics-based or constant fallback. Counter directions use
their own artifacts' values.

Additional BPB-Adv plotting knobs:

- `plot_steering_strength_families` (default `false`) adds
  `steering_strength_plots/<mode>/steering_strength_families.png` during the
  plot stage, using only no-adversary control rows.
- `axbench_bpb_adv_strength_sweep.yaml` enables that plot and uses zero-strength
  additive rows (`magnitudes: [0.0]`) without counter-steering pressure.

### Autointerp Intruder Filter — Whitelist Producer

Config: `configs/extraction/precompute_autointerp_filter.yaml`.

See [`docs/pipelines/autointerp-filter.md`](../pipelines/autointerp-filter.md)
for exact commands, artifacts, and the serial/concurrent parity ladder.

This is a second-stage producer for `feature_mask`: it scores local extracted singleton features with an intruder-detection judge and writes a whitelist file at `${output_dir}/accepted_feature_ids.txt`.

```yaml
feature_dir: null            # required; extracted feature shard directory
output_dir: out/autointerp_filter
data_dir: null               # null -> infer source parquet dir from feature_dir phase
n_sae_features: null         # null -> stats file, else feature_dir width
feature_ids: null            # optional candidate subset before mask
feature_mask: null           # optional pre-filter; same schema as training feature_mask
document_blacklist_path: null  # optional "<shard_stem>:<doc_id>"-per-line exclusion file

n_trials_per_target: 100
min_trials_for_score: 10     # below this many built trials -> score=null
n_positive_examples: 3       # + one intruder = four presented options
n_quantiles: 5               # Delphi-style activation quantile buckets
max_judge_calls: null        # per-process admitted-uncached-trial cap
score_threshold: 0.6         # accepted iff score >= threshold
top_k: null                  # optional cap after thresholding; counts targets
fail_on_empty: true          # final serial/combine CLI exits nonzero on no accepted IDs
audit_sample_limit: 20       # trial_audit jsonl/yaml rows; 0 disables audit samples

judge:
  model: openai/gpt-5-nano
  requests_per_minute: 60     # complete() admissions per minute; 0 = unlimited
  max_concurrent: 8           # in-flight uncached intruder judge calls; 1 = serial fallback
  max_prompt_chars: 32000
  parse_max_retries: 2        # retries after an unparseable response
  cot: false
  prompt_variant: highlight_context_token_first_v1  # delphi | calibrated_v1 | highlight_context_v1 | highlight_context_token_first_v1
  few_shot_set: sae_realistic_v1                    # delphi | sae_highlighted_v1 | sae_balanced_v1 | sae_realistic_v1

token_examples:
  device: cpu
  sae_release: gemmascope
  sae_l0: null
  activation_backend: auto
  model_dtype: bfloat16
  example_ctx_len: 32
  activation_threshold_ratio: 0.3
  max_positive_source_docs_per_target: 128
  max_intruder_source_docs: 5000
  max_windows_per_source_doc: 8
  max_positive_windows_per_target: 512
  max_non_activating_windows_per_target: 512
  materialization_batch_size: 128  # fixed slot-block size; 1 = serial reference

source_index:
  enabled: false
  path: ${output_dir}/source_doc_index.sqlite
  manifest_path: ${output_dir}/source_doc_index.manifest.json
  shared_cache_dir: null       # opt-in cache directory shared across repeated runs
  force_rebuild: false
  chunk_size: 5000000
  scan_workers: 1

neuronpedia:
  enabled: false
  model_id: null
  sae_source: null
  timeout_seconds: 30.0
  max_retries: 3
  backoff_base_seconds: 1.0
  backoff_max_seconds: 30.0
```

Target resolution order is candidate IDs → optional `feature_mask` → sorted
singleton feature targets → optional `max_targets` prefix. `top_k` counts
feature targets.
`fail_on_empty` must be a boolean and `audit_sample_limit` a strict nonnegative
integer. Neuronpedia `max_retries` must be a strict nonnegative integer;
`timeout_seconds`, `backoff_base_seconds`, and `backoff_max_seconds` must be
finite positive numbers, with `backoff_max_seconds >= backoff_base_seconds`.
These checks run even when Neuronpedia is disabled so an invalid resolved config
fails before token preparation or external calls rather than becoming a latent
mode-dependent error. `document_blacklist_path` excludes specific source
documents (identified by `<shard_stem>:<doc_id>`, not a per-target token-window
key) from ever being sampled as a positive or intruder candidate, anywhere in
the run — filtering happens in the sampler after any cache read, so it also
applies to windows cached before the blacklist existed and needs no cache-key
change. Each runner snapshots the parsed unique document-key set; its canonical
SHA-256 is stored as `document_blacklist_sha256` alongside `document_blacklist_path`
in the runner manifest, so both participate in the manifest-compatibility identity
check. Comments, ordering, and duplicate lines do not change that digest, but any
effective exclusion-set change invalidates a prior compatible runner manifest.
The digest is never null: when
`document_blacklist_path` is `null`, it is the canonical empty-set SHA-256.
Pre-blacklist manifests missing these fields normalize to that disabled identity
during compatibility checks, while a non-null legacy path without a digest remains
incompatible. `null` (the default) preserves the no-blacklist sampling behavior. See
[autointerp-filter.md § Document blacklist](../pipelines/autointerp-filter.md#document-blacklist).

Runner-manifest compatibility remains legacy-preserving: creation time,
schema-v2/unclustered shape, judge concurrency (`judge.max_concurrent`), and
source scan/shared-placement fields are normalized before comparison, while
judge RPM remains exact identity and compatible existing bytes are retained.
This is what lets a compatible existing `runner_manifest.json` from an older
run, or from a different `max_concurrent`, be reused rather than rejected.
When materialization actually runs serially (batch size `1`, or a custom
provider without the optional bulk extension), the batch field is
conditionally omitted as well. A real bulk run records it.

Feature shards are used to find candidate source documents, then those documents are re-forwarded through the SAE to build token-level windows. Autointerp scoring requires feature shards extracted with `ignore_bos_features=true`; legacy/no-BOS-off artifacts are rejected. `source_index.enabled=true` builds/reuses a SQLite source-doc index so full-width runs scan shards once and then read bounded top activating segment refs per feature plus deterministic intruder refs. Its manifest pins feature/data fingerprints, the full resolved target-ID universe, shard domain, top-k limits, `context_independent_features`, `ignore_bos_features`, and schema version; mismatch aborts rather than reusing stale refs. Building or reusing that index never requires loading the model, and every target/trial in the run reads the same in-process index through a query-time feature projection that never changes manifest/cache identity. Context-independent dirs are supported: token examples slice `segment_token_start:end`, inject BOS only when extraction would, drop injected BOS from rendered examples, and zero ignored natural BOS positions before active/highlight decisions. SAE projection is target-feature scoped. `token_examples.activation_backend=auto` uses TransformerLens, the path that matches SAELens/Neuronpedia examples; set `native` only for an explicitly validated fast-path run, or `transformerlens` to require the reference path. `token_examples.model_dtype` controls the native model/SAE dtype. TransformerLens token examples load only layers `0..L` for target SAE layer `L`, matching the existing `stop_at_layer=L+1` forward and avoiding unused later blocks during cold token-example materialization. Positive prompt rows highlight target-active tokens above `activation_threshold_ratio * max_activation`, and intruder rows are target-non-activating windows with Delphi-style false-positive highlighted tokens. The default judge prompt is the calibrated `highlight_context_token_first_v1` variant with `sae_realistic_v1` few-shot examples; set both fields to `delphi` only when you need Delphi-compatible scoring. Changing prompt/few-shot content changes its 16-hex SHA-256-derived version fingerprint. Candidate windows are deduped by rendered source-window identity before trial sampling, so activation differences from alternate focused forwards do not create duplicate options in a prompt. Trial sampling also rejects overlapping same-document token windows across the presented options. The token-example cache in `${output_dir}/token_example_cache.sqlite` is keyed by source fingerprints, model/SAE/backend/token-window settings, source-index settings, extraction flags, and canonical target identity. Parsed trial-result caching is in `${output_dir}/intruder_trial_cache.sqlite`: identity combines target/trial/sample settings with `token_window_v1` digests of sampled refs, plus judge model/CoT, prompt/few-shot names and content-version fingerprint, and provider render context. Window digests include token IDs and activations but exclude `token_strings`; the rendered final prompt is not directly hashed, so renderer/token-string changes require an explicit method/version or context bump. The configured prompt-length cap, parse retries, RPM, and concurrency also do not enter parsed-result identity. The cap is enforced only on a cache miss, while the cap/retries remain semantic manifest fields and RPM/concurrency are operational provenance. Successful model responses are retained for auditability. The generic LLM transport cache is bypassed for intruder prompts. Transport failures and parse exhaustion fail the run loudly and are retried on future runs rather than persisted. An enabled Neuronpedia fetch failure is non-fatal: the runner logs a warning, records a null description for that feature, and scoring continues, since the description is decorative metadata for the ranked/manual-review YAML rather than part of scoring identity.

`token_examples.materialization_batch_size` defaults to the fixed production
slot-block size `128`; it is a strict integer `>=1` and has no adaptive/`auto`
value. This is a feature-addressing/cache block size, not a model-forward batch
dimension. `1`
uses the unchanged serial provider and legacy v6 cache/manifest identity.
Values above one require the validated native-CUDA Gemma residual-post,
context-independent/no-BOS, fixed-token, singleton-target contract; unsupported
real-provider configurations fail with an instruction to set `1`. In
particular, the generic CPU/`activation_backend=auto` settings shown above must
override the batch size to `1`. The production Gemma 3 4B preset satisfies the
batch contract and uses `128`.

For block mode, feature IDs map to immutable quotient/remainder blocks, partial
cache blocks rebuild in full, and each completed block is one SQLite
transaction. Batch identity additionally includes the ordered target-plan hash,
block/resume, true-length singleton, fixed two-stream concurrency, and
non-vacuous companion-probe policy, all
source/window caps, numeric validation profile, and runtime fingerprint. A
normal run requires both the profile's exact runtime hash and its calibration
observation digest; only the internal calibration harness may inject an
unvalidated profile. CPU prefetch is scheduling-only and excluded. Startup,
periodic, and final serial shadows enforce finite tensors, exact active support,
and independent selected-SAE and raw-residual max/p99/mean envelopes. The
frozen calibration and held-out gates separately require exact window,
highlight, quantile, sampled-trial, and rendered-prompt surfaces. No failed
gate or OOM falls back. See the
[Autointerp Intruder Filter pipeline](../pipelines/autointerp-filter.md) for the
operator workflow and validation ladder.

Here prompt/few-shot content is represented by the truncated version
fingerprint rather than literal prompt bytes; renderer/token-string evolution
is safe only when the enumerated identity is deliberately bumped, and all such
digest identities assume no collision.

Every source-index load requires a rollback-format main database (SQLite
header write/read versions `1/1`) and rejects live `-journal`, `-wal`, or
`-shm` sidecars. The builder publishes DELETE mode; recovering nonpersistent
historical PRAGMA modes is not claimed.

The full-corpus preset uses `token_examples.model_dtype=bfloat16`, whose token
examples follow the structural tolerance contract in
[`autointerp-filter.md`](../pipelines/autointerp-filter.md#native-bf16-tolerance-contract)
rather than CPU-reference byte/trial equivalence. Use `model_dtype=float32`
(still native CUDA) or `activation_backend=transformerlens` when that reference
equivalence is required.

During source-index builds, each compressed shard is streamed to a temporary uncompressed safetensors spool and scanned in `source_index.chunk_size` entry chunks. The build path computes each shard's top source refs with a vectorized
scanner, then merges shard results in the parent with the same reducer:
max-dedup by `(feature_id, stable_key)`, followed by per-feature top-k ordered by
`magnitude DESC, stable_key ASC`. Tests retain the exact eager-equivalence
scanner and the prior survivor scanner as byte-equality oracles. The spool is
deleted after that shard; if a process is killed, a hidden
`.spool.*.safetensors` file in the output directory is safe to remove. This is
an implementation detail and does not change source-index or token-example
cache identity.
The intruder dedup set is scoped per shard rather than globally, so its RAM is
bounded by one shard's intruder candidates instead of growing with the whole
corpus; feature shard stems are unique, so cross-shard stable keys never collide
and the reservoir-sampled `intruder_refs` are byte-identical (duplicate shard
stems fail the build loudly).

`source_index.scan_workers` (default `1` = serial) parallelizes the per-shard
scan across that many worker processes. It is a scheduling knob only — excluded
from source-index cache identity, so an index built with any worker count loads
under any other with no rebuild. The parent merges worker results strictly in
shard-index order, so `positive_refs`, `intruder_refs`, and the manifest are
byte-identical for every worker count (default-suite `1` vs `2`; slow-suite
`{1,2,4}` plus forced out-of-order completion). The parent survivor set is
bounded by `target_count * max_positive_refs_per_feature` rows. Parallel builds
may also hold up to `scan_workers + 2` completed shard results in the rendezvous
window. Each concurrent worker holds one decompressed shard spool (~1.75x the
compressed shard, e.g. ~3.4 GB) plus working RAM, and only `scan_workers`
workers run at once; the build fails loudly at startup if the temp dir cannot
hold `scan_workers` concurrent spools. Any worker error aborts the whole build
with no partial index. Size `scan_workers` to the box's cores and free disk (the
32-core production box comfortably runs 4). In parallel mode, workers stream the
same shard-spooled and chunk progress events back to the parent while the parent
waits for strict in-order merges. `SHARD_DONE` log lines keep the total
parent-side elapsed time and also report `wait=` and `merge=` timings so a live
run can distinguish worker/rendezvous delay from parent merge cost without
changing index bytes or cache identity.
Do not treat larger `scan_workers` as automatically faster: short or
straggler-heavy runs can lose to process startup, concurrent spooling, and
strict-order rendezvous waits. Choose the value from a representative
`scripts.benchmark.benchmark_autointerp_source_index_build` matrix over
`scan_workers={1,2,4,8}` (or the machine's safe subset), and only compare timing
after `positive_refs_sha256` and `intruder_refs_sha256` match. The benchmark JSON
reports `wait_*`, `merge_*`, `max_spool_*`, and
`estimated_peak_concurrent_spool_*` fields so worker count can be sized against
both wall time and temp-disk headroom.

`source_index.shared_cache_dir` is an opt-in cache for repeated runs over the
same source-index contract. When set, the runner ignores the output-dir-scoped
`path` / `manifest_path` for source-index storage and instead writes
`source_doc_index_<sha256>.sqlite` plus a matching manifest under the shared
directory. The SHA-256 is computed from the canonical source-index manifest
payload, so it includes feature/data fingerprints, target IDs, shard domain,
top-k limits, seed, extraction flags, chunk size, schema version, and method,
while still excluding scheduling-only `scan_workers`. Mismatches fail through
the same manifest validation path; `force_rebuild=true` rebuilds the resolved
shared-cache entry. Leave this null for isolated one-off outputs, and set it for
benchmark reruns or prompt/judge experiments that should not pay the full
source-index build repeatedly.

Source-index construction publishes atomically: `_build_index` writes to a
temporary file and `os.replace`s it into place, so a concurrent reader never
observes a half-written index. Single-process runs take no cross-process
advisory lock and use no `$TMPDIR` lock namespace.
`source_index.force_rebuild=true` unconditionally deletes and rebuilds the
resolved index/manifest pair before scoring.

`judge.max_concurrent` is a throughput-only bound for uncached intruder trials (default `8`; `1` takes an equivalent serial fallback loop). `_resolve_all_trials` admits every target's missing trials in one canonical single-threaded pass first, in strict (target, trial) order, before any concurrency. `_evaluate_trial_wave` then judges that fixed set -- serially at `max_concurrent=1`, or via a bounded `ThreadPoolExecutor` above it. As the calling thread observes each successful completion, it commits that row to SQLite before progress logging; executor workers never access the cache. Outcomes are still sorted by their pre-assigned index before result assembly and error selection, so cache keys, output schemas, final result bytes, and lowest-index error precedence are unchanged regardless of `max_concurrent` (see `tests/steering/autointerp_filter/test_concurrency_parity.py`). A drained wave caches all successfully parsed siblings even when another trial fails; after an external kill, only completed commits are guaranteed. `judge.requests_per_minute` gates `complete()` admissions even when several calls are in flight; parse retries re-enter `complete()`, while provider transport retries may issue additional HTTP attempts within one admission.

Outputs:
- `target_scores.jsonl`: one persisted row per feature target.
- `autointerp_ranked.yaml`: concise human-readable targets ordered by score descending, including description, feature ID, score, and non-null errors.
- `accepted_feature_ids.txt`: sorted whitelist file for downstream `feature_mask: {mode: whitelist, path: ...}`.
- `manual_feature_filter.yaml`: commented flat YAML list for manual review; IDs are quoted and zero-padded to equal width, and all-commented/empty parses as `[]`.
- `manual_feature_filter.meta.json`: provenance for the manual YAML.
- `report.json`: aggregate counts and paths.
- `trial_audit.jsonl`: up to `audit_sample_limit` detailed rendered trials with final prompt text, token strings, token activations, highlights, raw model response, parsed prediction, correctness, and the execution-provenance `cached` flag. This flag can differ after finite-cap retries even when all scoring fields are equal.
- `trial_audit.yaml`: concise human-readable audit rows with final prompt text, rendered examples, score-relevant max activations, raw model response, parsed prediction, and correct index.
- `runner_manifest.json`: run identity for incompatible-rerun checks, written only after scoring succeeds.

Preset: `configs/extraction/precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos.yaml` targets `out/data/features/s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-pt/base/context_independent/no_bos`, sets `n_sae_features=65536`, enables the source index, uses Gemma-scope-2 `sae_l0=medium`, and writes Neuronpedia links with the matching published PT 65k coordinates `gemma-3-4b/22-gemmascope-2-res-65k`. This full-corpus preset defaults token examples to `device=cuda` and `activation_backend=native`; the generic base config remains the CPU/TransformerLens reference path unless a run opts into native explicitly.
That preset also sets `model_dtype=bfloat16`, a tolerance-based fast path; see
the pipeline guide's [Native bf16 tolerance
contract](../pipelines/autointerp-filter.md#native-bf16-tolerance-contract).

Manual coverage config: `configs/extraction/compute_manual_filter_coverage.yaml`.

```yaml
feature_dir: null
selection_yaml_path: null
exclude_last_shard: true
```

`scripts/preprocess.compute_manual_filter_coverage` reads the manual YAML through the shared safe YAML parser, validates a flat list of strict int IDs or quoted zero-padded decimal ID strings, rejects duplicates and out-of-range IDs against `feature_statistics.safetensors`, and prints a deterministic YAML report. Zero-padded IDs must be quoted because plain YAML can parse leading-zero numbers as octal. Empty/all-commented selections fast-path to `{0: n_segments}` from stats; non-empty selections stream the stats-domain shards and report a histogram plus nearest-rank `p0,p1,p5,p10,p50,p90,p95,p99,p100`.

### AxBench-Base output roots and shared Neuronpedia cache

Applies to `configs/eval/axbench_base/axbench_base.yaml`.

```yaml
output_root: null
run_tag: null
```

`output_root=null` preserves the historical effective root
`$NANOCHAT_BASE_DIR/eval/axbench_base`. A non-empty string is honored
literally, including relative-path semantics, and every run artifact resolves
under `{effective_output_root}/{sanitized_run_tag}`. Empty or whitespace-only
strings are invalid.

The cross-run Neuronpedia description cache lives under
`{effective_output_root}/_neuronpedia_cache/v1/`. Its filename is the full
SHA-256 of canonical JSON containing the exact Neuronpedia `model_id` and
`sae_source`; the file repeats and validates that identity. Same-identity
writers use an OS file lock and merge rather than overwrite stale snapshots.
Cache publication is transactional: fetched rows are not exposed through the
in-memory fast path until the temporary file is flushed and fsynced, atomically
replaces the target, and the parent directory is fsynced. Invalid feature-ID
or description types fail loud rather than being coerced.

Distributed invocations broadcast rank 0's autogenerated `run_tag` and require
every rank to resolve the same configured and effective output root. Explicit
rank-varying roots or tags are rejected before stage artifact writes.

Before issue #1001, a custom root was ignored for run artifacts. Startup now
rejects the ambiguous case where the configured run does not exist but the
same tag exists under the legacy default. Set `output_root=null` to resume the
historical run, or explicitly migrate and validate the complete directory.
This check runs before expensive work when the run tag is explicit or already
resolved; a clustered autogenerated tag necessarily follows checkpoint
identity resolution because that identity contributes to the tag.

### AxBench-Base seed extraction, BOS, and Stage 3e contexts

Applies to `configs/eval/axbench_base/axbench_base.yaml`.

```yaml
seed_prefix_max_nanochat_tokens: 512
seed_continuation_nanochat_tokens: 256
seed_pool_train_ratio: 0.75
seed_pool_seed: 42
seed_extraction_candidate_shard_size: 1024
seed_extraction_gpu_batch_size: 16
seed_extraction_cpu_batch_size: 64
```

Stage 0b raw-encodes source documents without BOS, requires content length
`N >= 2*M`, stores the exact continuation `[N-M:N]`, and stores the immediately
preceding prefix `[max(0, N-M-C):N-M]`. `C` must be at least `M`; both are
positive. The three extraction batch/shard controls must also be positive and
are semantic artifact identity because batch grouping may be numerically
observable.

The stored seed lists are BOS-free exact source slices. AxBench adds one owned
BOS only at a target-model boundary and never deduplicates an equal payload
token ID. Stage 1 synth rows store `n_prefix_nanochat_tokens` as the exact first
model token overlapping output text after a single combined-text encode,
including that owned BOS.

### `min_prefix_chars` / `max_prefix_chars` — shared AxBench decoded-text bounds

Applies to `configs/eval/axbench_base/axbench_base.yaml`
(`AxBenchBaseConfig` in `steering/eval/axbench_base/config.py`).

```yaml
min_prefix_chars: 100
max_prefix_chars: 2048
```

Both values are strict integers (booleans are rejected).
`min_prefix_chars >= 0`, `max_prefix_chars >= 1`, and the closed interval must
satisfy `min_prefix_chars <= max_prefix_chars`. The unit is Python decoded
Unicode code points: it is deliberately not UTF-8 bytes, grapheme clusters,
words, sentences, Gemma tokens, or nanochat tokens.

Stage 0c candidate extraction and Stage 1b positive synthesis both retain the
most recent `max_prefix_chars` characters. Stage 0c applies the window at the
activating segment end. Stage 1b applies it to the complete frozen
`SeedRow.prefix_text`, builds one tuple aligned to the original train indices,
and excludes entries shorter than the minimum. Stage 1b fails before
percentile-table construction, any Stage 1b SAE forward, or API/cache work when
none remain; the CLI dispatcher has already entered the shared base-model/SAE
contexts. It uses the bounded view for all semantic consumers but never mutates
the Stage 0b safetensors, seed fingerprint, stored tokens, or Stage 4
evaluation seeds.

The exact bounded text enters Stage 1b prompts and therefore its existing
prompt-derived synth cache key. Changing a formerly overlong prefix produces a
cache miss by construction; the cache schema and tag format do not change.
Each rich or LLM-avoid attempt also derives unique collision-checked response
markers from its stage/class/concept/feature/source/attempt identity and prefix
digest. The prompt ends at the exact local suffix (at most 96 characters).
Provider output must authenticate that suffix before continuation data; parsing
preserves leading data, validates the entire response, limits it to
`output_length` extraction-tokenizer tokens, and validates the decoded window
again. `max_retries` therefore bounds semantic/protocol retries as well as SAE
QC retries in Stage 0c/1c, while `positive_max_attempts_per_seed` provides the
independent rich/avoid cap in Stage 1b. `synth.transport_max_retries` remains
transport-only.
These fields already participate in the Stage 0c fingerprint. Stage 1b also
has a schema-v2 partial-checkpoint fingerprint that binds the target-independent
positive compatibility stream, ordered concepts and seed rows, statistics
content, model/SAE/tokenizers, prompts and continuation protocol, and cluster
contract. It excludes only `n_positives_per_concept` from selection semantics
and separately records a strict positive `highest_requested_target`. It also
deliberately excludes operational rate,
concurrency, transport-retry, profiling, output-root, run-tag, and the
prefix-percentile-store toggle. It includes the post-synthesis
`qc_fallback_on_exhaust` and population-drop thresholds because they determine
whether and how a checkpoint may publish. Resume still requires the same
explicit `run_tag` because that selects the checkpoint directory. There are no
checkpoint cadence or storage configuration knobs:
Stage 1b uses the fixed 64-terminal interval and the run-local
`stage1b_partial.parquet` plus `stage1b_partial.json` pair. Every accepted pair
is validated before checkpoint ownership transfers, and every emitted prefix
is also validated again immediately before final parquet publication.

Candidate scoring first fills toward `72` admissible candidates with the exact
historical target-72 scorer batch contents and widths. Physical exhaustion may
end that fill early when the configured target is already buffered. For targets
above 72, clean scoring continues in full `qc_batch_size` chunks; only physical
clean-pool exhaustion makes a partial chunk, after which dirty candidates are
scored once in stable percentile order. Surplus candidates remain buffered.
Current target-72 provenance is
`positive_target_refill_post_provider_qc_v5`; non-72 outputs use
`positive_target_compatibility_post_provider_qc_v6` with the compatibility
target, prefix-scoring rule, and effective post-provider batch identity. The
requested target is artifact metadata only; the effective post-provider cap
remains in resumable checkpoint identity.

Schema-v2 resume permits equality or growth. A higher target restores state,
enlarges tracker targets, and proves enough deterministic candidate capacity
before admitting deficits or committing the new highest target. Infeasible
growth and failed writes leave the previous JSON authoritative; a decrease
fails before prefix-cache or provider construction. Schema-v1 migration is
restricted to an exact Phase 1 target-72 fingerprint and a current target of at
least 72. Other v1 targets used target-shaped BF16 prefix-scoring batches and
fail with finish-or-restart guidance. Growth requires the same explicit
`run_tag` and an
authoritative `stage1b_partial.json`. Once a completed corpus exists it is
never extended in place, including when JSON remains after interrupted
checkpoint cleanup; that partial can resume only the completed target. Use a
fresh tag for post-publication target changes. Automatic tags normally differ,
so growth is an explicit operator action.

Stage 0b extracts exact-prefix SAE activity with production `process_shard`,
using the layer, activation, pooling, segment size, phase,
context-independence, BOS-feature masking, model/SAE/hook/L0 identity parsed or
resolved from these fields:

```yaml
feature_dir: ???
feature_extraction_model_id: null
neuronpedia_model_id: gemma-2-2b
neuronpedia_source: 8-gemmascope-res-16k
extraction_tokenizer_id: google/gemma-2-2b
extraction_max_tokens: 1024
validation_shards: [369]
```

Stage 3e has no separate knobs. It consumes `enabled_methods`,
`disabled_methods`, `run_negative_mode`, the mode-specific synth eligibility,
and `generation.eval_seeds_per_concept_{positive,negative}` to enumerate exact
encoder-decoder Stage 4 prompts. Its manifest also binds prompting-description
hashes, the nanochat tokenizer, seed fingerprint, and real checkpoint sequence
length.

The generic `configs/eval/build_eval_context_cache.yaml` adds:

```yaml
required_contexts_manifest: null
extraction_tokenizer_id: null
```

For AxBench, set both to the Stage 3e manifest and matching extraction
tokenizer. Conversion validates the committed JSONL/parquet hashes and
ordering, stamps `contexts_fingerprint` and the extraction-tokenizer checksum
into cache metadata, derives the extraction BOS decoded character length, and
asserts any supplied `source_bos_char_len` agrees. Stage 4 then requires that
fingerprint along with the existing cache identity, feature-selection hash,
bucket/width settings, and `model_max_seq_len`.

### Self-conditioned ChatCORE

`configs/eval/chat_eval_self_conditioned.yaml` extends `chat_eval.yaml` with
`evaluation_mode: self_conditioned`, `source: encdec`, empty `feature_configs`
and `target_real_configs`, batch size 1, one sample and temperature 0. Both
fixed and turn checkpoints support exact `problem_start`/`max_problems` ranges.
Omitting `evaluation_mode` preserves legacy dispatch; explicit contradictory
mode/source/config combinations fail before loading.

| Key | Default | Contract |
|---|---|---|
| `self_conditioned.cache_dir` | `null` | Required explicit cache root: schema 4 fixed or schema 5 turn |
| `self_conditioned.benchmark_steering` | `null` | Optional exact `{targets: path, strength: number}` block; requires self-conditioned mode |
| `self_conditioned.benchmark_steering.targets` | required when enabled | One immutable six-task safetensors bundle fitted for the checkpoint's source/statistics/vocabulary |
| `self_conditioned.benchmark_steering.strength` | `0.0` in steering preset | Finite continuous number in `[0,6]`; existing interpolation weight is strength divided by six |
| `self_conditioned.seed` | `1234` | Integer base seed; example identity determines independent token/selection seeds |
| `self_conditioned.fixed_segment_size_assertion` | `null` | Fixed-only equality assertion against measured metadata K; no override/fallback |
| `self_conditioned_output_dir` | `null` | Defaults to checkpoint `chat_eval`; nonblank explicit root writes `self_conditioned/result.yaml` and never overwrites |

The checkpoint selects segment mode and K. Turn requires
`turn_self_cond_segment_size_rustbpe`; fixed requires
`fixed_self_cond_segment_size_rustbpe` plus measured provenance. CLI cadence
keys are rejected even when equal. `next_segment_conditioning_offset: auto`
derives checkpoint timing; explicit values assert equality. Self-conditioning
requires current-segment real prompt routing, MAX_POOL FP and two encoder
buckets. `model_max_seq_len` must equal the effective model and cache context.

`configs/eval/record_fixed_cadence.yaml` requires `checkpoint_meta_path`,
`feature_dir`, `parquet_dir`, `tokenizer_dir`, and a fresh `output_dir`;
`max_rows: 4096` bounds evenly sampled TRAIN rows.
`fixed_segment_size_assertion: null` optionally checks the resulting measurement.
It writes an immutable metadata leaf linked to original weights.

`configs/eval/prepare_chat_self_conditioned_cache.yaml` requires measured
`checkpoint_meta_path` and fresh `output_dir`. `stage: plan` authenticates and
writes extraction inputs/command; `stage: publish` validates the extracted
cache and atomically installs schema 4. Run the emitted `extract.sh` on a GPU
between stages. Preparation uses `<output_dir>.prepare` and extracts into its
`extracted` subdirectory. Optional legacy inputs are
`legacy_cache_dir`, `legacy_contexts_file`, `legacy_feature_dir`, and
`legacy_parquet_dir`. `chat_tasks` defaults to all six, `chat_max_problems: 500`,
and `shard_size: 1024`. All geometry and selection authority comes from the
checkpoint, including the cache preselection seed (separate from inference seed).

The historical merger also accepts `kind: self_conditioned` with receipt v3.
It requires identical architecture/compute dtype, checkpoint/cache bytes,
recorded cadence/provenance, policy, seeds, tools and generation identity.
See [evaluation runbooks](../pipelines/evaluation.md#self-conditioned-chatcore).

`configs/eval/chat_eval_benchmark_steering.yaml` extends the self-conditioned
preset and requires `self_conditioned.benchmark_steering.targets`.
`configs/eval/build_chat_benchmark_targets.yaml` controls offline preparation:
required `checkpoint_meta_path` and `output_dir`; optional exact `stats_path`
relocation; `dry_run: true`; `device: cuda`; `extraction_batch_size: 1`;
`positive_cap: 1024`; `background_count: 1024`; `partition_seed: 42`.
The positive cap is per task, after exclusions and length checks. `revisions`
contains immutable Hub commits for `mmlu`, `arc`, `gsm8k`, `humaneval`, `mbpp`
and `background`; all six are required. `coding_exclusions` maps MBPP task-ID
strings to reviewed HumanEval-overlap reasons. See the
[fitting and evaluation runbook](../pipelines/self-conditioned-chatcore.md#benchmark-identity-interpolation)
for split ownership, rendering, artifact admission and calibration limits.

### Turn target-real ChatCORE

`configs/eval/chat_eval_turn.yaml` extends `chat_eval.yaml`, clears fixed SAE
identity/bucket defaults, and selects static pooled reference conditioning.
The default turn decoder context is 2,048; it must match both the cache and
loaded model. `segment_mode` is an optional checkpoint assertion in the base
config (`null` resolves every metadata witness), and explicitly
`turn_delimited_fixed_tokens` in the turn preset.

`extraction_tokenizer_id: null` resolves the model in
`structured_sft_online_sae_lineage_v1` through the extraction model registry.
All six SAE coordinates come from that lineage; `user_config.sae` contains
non-authoritative SFT defaults. `user_config.num_buckets: null` means inherit
the checkpoint's resolved bucket count.
`stats_path: null` uses the checkpoint training-statistics path; an override
may relocate identical raw bytes. `conditioning.source_segment_size` and
`conditioning.source_context_window` default to null and become exact lineage
assertions when set. Identity is derived from the complete checkpoint contract;
an explicit `conditioning.identity.feature_selection_hash` is only an assertion.
No cadence knob is exposed or required for static pooled/lockstep execution.

`target_real_output_dir: null` preserves the default
`{checkpoint_dir}/chat_eval/target_real/<name>.yaml` output. Set distinct roots
for separate task or shard jobs; each writes `<root>/target_real/<name>.yaml`.
`problem_start: null` means index 0 on the turn path. An explicit nonnegative
integer selects `[problem_start, problem_start + max_problems)` in the task's
canonical order, exactly as used by cache enumeration's `chat_max_problems`.
`max_problems: null` evaluates the remaining suffix. Empty or out-of-bounds
ranges and nonpositive/noninteger counts fail; turn ranges are never clipped.
Fixed static/target-real evaluation rejects any non-null `problem_start`, including 0.
Self-conditioned fixed evaluation supports exact ranges as described below.
Every selected problem's full and independently rendered prefix cache rows are
checked across all selected tasks and target configs before the first forward.
The turn YAML records resolved integer `problem_start`, `max_problems`,
`task_size`, `num_passed`, and `total` under `seeds.adhoc.task_details.<task>`.
`configs/eval/merge_chat_eval_target_real.yaml` takes `input_files: []` and a
required `output_file`. With `task_totals: null`, it retains the complete,
disjoint six-task union contract. For problem shards, declare the exact task
set and N per task, for example `task_totals={GSM8K:500,HumanEval:164}`. Every
task must cover `[0, N)` without overlaps or gaps; tail omissions also fail.
Counts are summed before division. A subset merge emits per-task results;
ChatCORE is emitted only for all six tasks. Version-2 identity receipts keep
generation settings separate from ranges; legacy version-1 task merges retain
their original settings comparison, including `max_problems`. Mixed versions
are rejected. Publication is atomic. See the runbook below for launch commands.

The `build_eval_context_cache_turn` and `validate_eval_context_cache_turn`
presets opt into schema 5 with `evals=chat_target`, `model_max_seq_len=2048`,
and `chat_max_problems=null`, matching evaluation's uncapped suite. Both take
`checkpoint_meta_path`; deep validation additionally requires JSONL, parquet,
and raw source-shard inputs. The builder's turn preset clears the legacy
threshold selector and BOS-width/SAE defaults. All non-null selection/geometry
overrides must agree with checkpoint lineage. Fixed presets retain their
existing values. See the [complete s32 runbook](../pipelines/evaluation.md#turn-delimited-target-real-context-caches).

### AxBench-Base percentile-threshold units

Applies to `configs/eval/axbench_base/axbench_base.yaml`
(`AxBenchBaseConfig` in `steering/eval/axbench_base/config.py`).

```yaml
verification_percentile_threshold: 0.7
accept_threshold: 0.3
```

Both values are finite percentile fractions in the inclusive range `[0,1]`.
The SAE QC oracle returns raw, unquantized percentile points on the nominal
0-100 scale, so both fractions are converted through the shared canonical
mapping `points = 100 * fraction` before comparison.

- Stage 0c passes one verification continuation only when its absolute raw
  percentile is strictly greater than the resolved threshold (`> 70` by
  default).
- Stage 1b accepts one concept-rich continuation when
  `output_percentile - seed_percentile` is greater than or equal to the
  resolved threshold (`>= 30` by default).

Stage 1b records the fraction, resolved points, comparison, algorithm version,
and exact percentile currency (`0.0` or `[0.5,99.5]`) in
`synth_training_positive.parquet`. Positive-corpus consumers fail closed on
missing or malformed contract metadata; Base consumers also require its
threshold to equal the current config. Regenerate legacy corpora under a fresh
`run_tag` rather than annotating their old acceptance decisions.

Both Stage 1 corpora also record the static
`axbench_stage1_continuation_contract`. Its role is `stage1b_positive` for the
positive corpus and `stage1c_negative` for the negative corpus; the remaining
fields pin marker derivation, accepted suffix-echo variants, terminal trimming,
copy/loop/control/special-token guards, pre-truncation validation, and
post-decode validation. No configuration knob disables this contract. Current
Base, BPB, and BPB-Adv consumers fail closed on absent, malformed, stale, or
role-swapped metadata.

### `verification_prefill_mode` — AxBench-Base Stage 0c Prefill Algorithm

Applies to `configs/eval/axbench_base/axbench_base.yaml`
(`AxBenchBaseConfig` in `steering/eval/axbench_base/config.py`).

```yaml
verification_prefill_mode: shared  # "shared" | "repeated"
```

`shared` is the default. For a verification chunk with `K` candidate prefixes
and `M = sum(seed_counts)` continuation rows, it prefills the `K` prefixes once,
expands the public Transformers KV cache and the remaining decode state into the
existing prefix-major `M`-row layout, and then uses the common autoregressive
decode loop. `repeated` is the rollback/reference algorithm: it prefills all
`M` repeated prefix rows, exactly as Stage 0c did before schema 8. A one-prefix
shared chunk still prefills once; a one-prefix repeated chunk retains the legacy
`batched_generate_seeds` path.

The two modes are separately deterministic but are not required to produce the
same tokens. Prefill batch shape can change floating-point execution and hence
sampled tokens. The selected mode is therefore part of the Stage 0c fingerprint,
and switching modes requires a fresh Stage 0c artifact set or a new `run_tag`.
Stage 0c checkpoint and completed-artifact schema 8 does not migrate schema-7
work. The removed `batched_generation_mode` key has no alias; use
`verification_prefill_mode=repeated` when explicitly selecting the old
algorithm.

Shared prefill requires the public `Cache.batch_select_indices` API in
Transformers 5.6.2 or newer. The project dependency minimum is
`transformers>=5.6.2`. Unsupported cache objects fail loudly and never trigger
an automatic repeated-mode retry.

### `verification_api_model` — AxBench-Base Stage 0c API Continuations

Applies to `configs/eval/axbench_base/axbench_base.yaml`
(`AxBenchBaseConfig` in `steering/eval/axbench_base/config.py`).

```yaml
verification_api_model: inclusionai/ling-2.6-flash  # OpenRouter model id | null
verification_api_prefix_wave_size: 100
```

The default `inclusionai/ling-2.6-flash` replaces only the source of Stage 0c
verification continuations with OpenRouter. Candidate discovery still scans
validation shards for prefixes where the feature is active, and every returned
continuation must still pass the existing local SAE percentile and
`min_consistency_passes` gate. This option does not generate Stage 1's paired
concept-rich/concept-avoiding training corpus and does not bypass rejection
sampling.

Set `verification_api_model=null` to preserve the schema-8 Stage 0c contract
exactly: candidate prefixes are continued by the local feature-extraction LM
using `verification_prefill_mode`, then rejection-scored by the local SAE.
This branch does not construct an OpenRouter client, fetch activation examples
for API generation, open the synth-response cache when examples QC is also
off, or add any API-generation fields to the legacy Stage 0c fingerprint.

The API branch deliberately never supplies the candidate's published
Neuronpedia description to its generator. It fetches up to
`4 * n_activation_examples` raw top-activating examples, validates and
deduplicates them, keeps the first `n_activation_examples`, renders source text
as separate before/peak-token/after fields, and asks the configured model to
infer a concise operational definition from that evidence. If every independent
peak normalizes to the same exact text, a local deterministic check requires the
definition to quote that text and explain its role; this prevents a model from
substituting an unsupported capitalization or adjacent-context theory. It then asks the
same model for only a continuation of each untrusted prefix toward that
definition. Continuation calls use a dedicated system instruction plus a
short contract-first user prompt. The independently validated operational
definition is repeated as quoted data in that system instruction; it is never
replaced by the published Neuronpedia description. A hard
literal-applicability gate makes local coherence authoritative: if the source's
existing final-sentence entity or action cannot directly bear the property,
realistic unsteered prose is correct and the local SAE gate rejects it later.
The provider may not invent a new actor, analysis, lineage, itinerary, goal,
parameter regime, example, analogy, scene, or domain to force activation. At
the default 64-token target, the prompt
asks for 90-130 new words in one continuous block, with the end marker on its
same line after exactly one ASCII space; this leaves extraction-tokenizer headroom without inducing the
two-paragraph and overlong-output failures of the earlier contract. The
instructions make local coherence outrank forced
steering and forbid invented glyphs; a narrow local gate also rejects new
combining, compatibility, supplementary-plane, foreign-script, or unassigned
letter forms and dense clusters of new non-ASCII Latin forms at a Latin source
boundary unless the code point is already present in trusted source text or
trusted recurring-peak evidence. Definition inference remains an exact
user-only request. Segment cuts are character-aligned rather than linguistic,
so the candidate index also retains up to 64 exact characters immediately
following
the prefix in the source document. It ends that trusted bridge at the first
sentence boundary when possible, otherwise at the final available whitespace.
The provider sees prefix plus bridge and must reproduce only their final 96
characters inside deterministic nonce-qualified begin/end markers. The parser
authenticates and removes that short echo, then prepends the bridge from local
source bytes; the provider can neither guess nor corrupt the text that repairs a
mid-word or mid-clause cut. A candidate at document end has no bridge and
becomes a shape-stable miss without an API call.

With a validated clustered target, the evidence-inference contract runs once
for every sorted mask-allowed member, producing distinct evidence-derived
member descriptions. Continuation proposal slots then use those descriptions
round-robin over the global flattened candidate/seed order. The cycle-policy
version enters the conditional Stage 0c fingerprint. At request construction,
the member ID and evidence identity enter the authenticated envelope and
therefore the prompt-derived cache key; the member ID and global
candidate/seed position also enter the cache tag, while request attrs record
the member ID and global proposal index. Those per-request identities do not
enter the conditional API-generation fingerprint subobject. API wave
partitioning and retries cannot reassign a slot. Member assignment is
nevertheless resume-pinned upstream:
`representative_to_allowed_members` is part of the feature-contract semantic
payload, and its `feature_contract_fingerprint` is folded into the full Stage
0c fingerprint. Changing a representative's resolved members under the same
`run_tag` therefore changes the Stage 0c fingerprint and makes resume fail
loudly before generation performs any cache lookup. The cache-keyed but not
resume-pinned surface is the external evidence identity: on a cold or missing
evidence-cache entry, Neuronpedia top-activation evidence can drift under an
unchanged feature contract and can change the inferred member definition;
regardless of whether the resulting definition text happens to match, the new
evidence identity changes the authenticated envelope and prompt-derived cache
key without changing the Stage 0c fingerprint. No published representative or
member label is used. A one-allowed-member cluster takes the exact ordinary
request, prompt, cache-key, and response path.

The parser discards text outside the envelope, but rejects missing nonces,
duplicated begin markers, full-prefix echoes or substantial prefix-suffix echoes,
substantial normalized source-phrase echoes anywhere in the response,
assistant preambles/sign-offs, control characters, protocol-like text,
special-token pollution, prompt/meta prose, copied four-word definition
phrases, repeated phrase loops, unsupported writing-system fragments, newly
fabricated or densely mutated Unicode letter forms at Latin boundaries, and
unnatural recurring-peak behavior (missing, quoted, adjacent, overused, or metalinguistic
same-script forms; copied foreign-script forms). Exactly one
complete leading sentence may be removed only when the copied definition
phrase is wholly confined to that sentence; the remainder must independently
pass every check. A clean short response is retained and extended by the next
content attempt. All pieces are concatenated and encoded once, preserving BPE
seams, then the entire text and exact `verification_length` decoded-token window
are validated separately so appended garbage cannot hide after truncation. An
exhausted continuation slot becomes a zero-percentile rejection-sampling miss,
while inability to derive a valid evidence-grounded definition fails Stage 0c
loudly.

The prompt requests one ASCII space before the authenticated end marker. The
parser strips terminal ASCII framing whitespace even when it is missing or
repeated. It also normalizes the live-observed 95-of-96 left-truncated suffix
echo and a nonce end marker missing its final `]`; these presentation bytes are
never scored. Duplicate trailing end-marker chatter is discarded after the
first authenticated boundary. Missing nonces, duplicate begin markers, right-truncated or
mutated echoes, and shorter echoes remain hard failures. After a transport,
framing, or quality failure, the next content attempt receives a fixed
failure-class instruction; rejected provider text is never copied into the new
prompt.

The existing synth client supplies the transport controls. `SynthConfig`
defaults to model `inclusionai/ling-2.6-flash`, `max_concurrent=100`, and
`requests_per_minute=6000`, independently of the judge client's conservative
10/60 defaults: `synth.max_concurrent` bounds independent continuation calls
within a Stage 0c API wave, and `synth.requests_per_minute` controls admissions.
The selected model
comes only from `verification_api_model`; changing dormant `synth.model` does
not change this path. Active mode requires both `OPENROUTER_API_KEY` and
`NEURONPEDIA_API_KEY`, requires `synth.provider=openrouter`, and rejects empty
model IDs, `synth.max_concurrent < 1`, `synth.requests_per_minute < 0`, or
`n_activation_examples < 1` during config validation. Rate and concurrency
limits are per process, so aggregate DDP traffic scales with `world_size`. The
API-specific `verification_api_prefix_wave_size` defaults to 100 speculative
candidate prefixes, capped by candidate-attempt/pool and zero-hit-bail limits
but intentionally not by the remaining target. With five seeds, a full wave
contains 500 logical calls and can keep all 100 workers occupied. Every
proposal is scored locally, while only the first target passes in candidate
order are retained. The separate `verification_cross_prefix_batch_size=4`
controls only explicit-null local Gemma generation and remains part of its
exact schema-8 fingerprint.
Stage 0c also sends a provider `max_tokens` cap of
`max(1024, 8 * verification_length)`: OpenRouter counts provider-tokenizer and
potential hidden tokens, while exact visible length remains enforced with the
Gemma extraction tokenizer. Other OpenRouter callers omit this field and keep
their historical request body unchanged.

Validated activation evidence and raw LLM responses are committed immediately
to the run's `synth_response_cache.sqlite`. Active-mode partial/completed
artifacts use Stage 0c schema 9 and include the API protocol, model, client
limits, retry/evidence settings, and prompt hashes in their conditional
fingerprint. They cannot be mixed with schema-8 local-generation artifacts.
Changing the option or any conditional identity field requires a new
`run_tag` or an explicit Stage 0 artifact restart.

### `concept_feature_whitelist_path` — AxBench-Base Stage 0a Concept Whitelist

Applies to `configs/eval/axbench_base/axbench_base.yaml`
(`AxBenchBaseConfig` in `steering/eval/axbench_base/config.py`). The default is
`null` (disabled — every count-filtered, described feature is a candidate, the
pre-existing behavior).

```yaml
concept_feature_whitelist_path: null  # optional path: accepted_feature_ids.txt OR manual_feature_filter.yaml
```

This restricts which SAE features Stage 0a (`prepare_concept_list`) is
allowed to even consider as concept candidates, applied **before** the
Neuronpedia description fetch. It is **concept-candidate filtering,
completely distinct from the encoder-decoder `conditioning.feature_mask`**
documented above: `feature_mask` restricts which features condition the
target model at train/eval time and has no bearing on which concepts
AxBench-Base evaluates; `concept_feature_whitelist_path` restricts
which features are eligible to become AxBench-Base concepts in the
first place. AxBench-Base's own `AxBenchBaseConfig.conditioning.feature_mask`
(`configs/eval/axbench_base/axbench_base.yaml`) *is* that same downstream
`feature_mask` surface, scoped to Stage 4 context-feature selection for
the encoder-decoder target — it is a nested field under `conditioning`,
not a sibling of `concept_feature_whitelist_path`, and resolving it
does not touch the Stage 0a whitelist or vice versa.

`concept_feature_whitelist_path` points to either an
  `accepted_feature_ids.txt` (one integer per line; `#` comments and
  blank lines ignored) or a `manual_feature_filter.yaml` (flat list of
  quoted zero-padded ID strings; commented-out lines are excluded) —
  the exact two whitelist artifacts written by the
  [Autointerp Intruder Filter](#autointerp-intruder-filter--whitelist-producer)
  (`scripts/preprocess/precompute_autointerp_filter.py`). See
  [autointerp-filter.md](../pipelines/autointerp-filter.md) for how to
  produce and manually review one.

**Resolution order** (`resolve_concept_feature_whitelist` in
`steering/eval/axbench_base/concept_whitelist.py`, applied inside
`prepare_concept_list`):
1. Require every whitelist ID to have a `per_feature_stats` entry in the
   configured feature statistics. Any absence is a fatal error before the
   Neuronpedia client is created: the feature either never fired in the
   statistics corpus or the whitelist targets a different SAE/layer.
2. `min_corpus_count` (the pre-existing corpus-count filter).
3. Optional `concept_description_fetch_limit` truncation. If this leaves fewer
   than `max_concepts` whitelist candidates, Stage 0a fails before creating the
   Neuronpedia client.
4. Neuronpedia description-quality filter (`min_description_length`,
   after the description fetch, which itself only queries the
   whitelist-surviving IDs from step 1-2 — not the full statistics
   universe).

`candidate_concepts.json` is guaranteed to contain no `feature_index`
outside the configured whitelist.

**Validation** (all errors are `AxBenchBaseError`):
- An empty resolved selection (for example, an all-commented
  `manual_feature_filter.yaml`) → error.
- Duplicate IDs, any ID outside `[0, n_sae_features)`, or a
  `concept_feature_whitelist_path` with an extension other than
  `.txt`/`.yaml`/`.yml` → error.
- Any requested in-range ID absent from `per_feature_stats` → error before
  description fetching or artifact writes.
- When the whitelist is active, the surviving candidate pool drops
  below `cfg.max_concepts` at any filtering stage (after
  `min_corpus_count`, after `concept_description_fetch_limit`, or after the
  description filter) → error
  (fast-fail rather than silently handing Stage 0c an under-sized
  pool).

**Stage 0c resume interaction:** both the live per-rank Stage 0c checkpoint
fingerprint (used at every `world_size`) and the completed
`negative_mode_prefixes.parquet` metadata include a content hash of the ordered
candidate pool. Changing `concept_feature_whitelist_path` (or
`min_corpus_count`, or the underlying Neuronpedia descriptions) and re-running
Stage 0a under the same pinned `run_tag` changes that hash whenever it produces
a different ordered candidate pool. Partial resume
rejects the changed fingerprint; completed resume reloads
`candidate_concepts.json` and rejects a missing or mismatched final-artifact
hash. Neither path can silently mix verification results from two candidate
pools. See
[axbench-base-eval.md § Multi-GPU support](../pipelines/axbench-base-eval.md#known-limitations).

## Chat TUI serving

`configs/serving/chat_tui.yaml` requires `target_meta_path` and matching
`feature_statistics_path`. For fixed checkpoints, `self_cond_segment_size`
(default 256) remains caller-owned. Turn-delimited post-training checkpoints
ignore that setting and use the strict positive top-level
`turn_self_cond_segment_size_rustbpe`; Gemma source segment size never determines
generated cadence. A raw turn-SFT checkpoint without the measured value fails.
Turn post-training leaves with missing conditioning or stale default SAE config
also require `pretrain_meta_path` (default `null`), pointing to their exact
pretraining metadata; a raw turn-SFT leaf may omit it, in which case its
online-SAE lineage is the sole identity witness. Whenever supplied, the shared
loader validates its model tag and SAE against the target and lineage; the
leaf's frozen turn policy and measured cadence remain authoritative. Fixed
Chat rejects non-null overrides and retains its existing ancestor recovery.
In turn mode, `max_new_tokens_per_turn` must fit in full alongside the RustBPE
prompt and one closing marker; the raw-SHA-bound Gemma context limit is checked
separately. Fixed mode retains its existing remaining-capacity sampling cap.
See [serving](../pipelines/serving.md#chat-tui) for the full config table, epoch
transactions, current-turn edits, cancellation, and seed/reset behavior.

### Public HF ChatCORE

`configs/eval/chat_eval.yaml` requires exactly one of the native `source` chain
or `hf_path`. The latter may be a supported public ID or one concrete local
repository HF `step_NNNNNN` export. The presets `chat_eval_qwen2_5_3b_instruct` and
`chat_eval_smollm3_3b` set the supported public instruction-tuned model ID and
full immutable `hf_revision`. Local exports instead require `hf_revision=null`
and `dtype=bfloat16`; their export metadata supplies the pinned source model ID
used for family geometry validation. HF mode requires a fresh `output_dir`; SmolLM3
also requires `hf_template_date` (`YYYY-MM-DD`, preset `2026-09-10`) and always
disables thinking. `model_tag`, `step`, and nonempty conditioning configurations
are rejected in HF mode.

`batch_size` (8) controls categorical inference; generation is sequential per
prompt per rank. Shared generation knobs are `temperature` (0), `num_samples`
(1), `max_new_tokens` (512), and `top_k` (50, nullable to disable). `max_problems`
is null for full datasets. `task_name` is null for all six tasks, or a quoted
pipe-separated exact-name selection such as `'task_name=GSM8K|ARC-Easy|MMLU'`.
See [HF ChatCORE](../pipelines/evaluation.md#public-instruction-tuned-hf-baselines)
for the full rendering, strict-grading, RNG, stopping, and evidence contract.

## Hydra Output Behavior

All configs set:
```yaml
hydra:
  run:
    dir: .          # Don't create output directories
  output_subdir: null  # Don't write .hydra/ config copies
  job:
    chdir: false    # Don't change working directory
```

This keeps the project directory clean. Artifacts are written to explicit output directories.
