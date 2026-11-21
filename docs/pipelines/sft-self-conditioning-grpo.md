# Structured-SFT Self-Conditioning GRPO

This is a separate, opt-in research pipeline. It does not replace conditioned
SFT, legacy SFT parquet, inference-time self-conditioning, or decoder-only
`chat_rl`. The checked-in configuration is an implementation surface, not a
qualified production or H200 preset.

## Data flow

```text
structured messages
  -> authoritative RustBPE targets, assistant mask, and logical turns
  -> one conversation environment (no packing)
  -> sequential feature-policy actions between target segments
  -> detached teacher-forced LM reward
  -> conversation-local GRPO advantages
  -> differentiable policy replay and one DistMuonAdamW step
```

The structured renderer is additive. `render_conversation()` and every legacy
SFT output remain unchanged. A structured row stores non-BOS token IDs,
target-aligned mask bits, logical-turn spans, decoded text for alignment only,
and the stable source `original_index`. Token IDs are authoritative: decoded
text is never re-tokenized to reconstruct targets.

Convert a separate structured artifact family with:

```bash
uv run python -m scripts.preprocess.convert_sft_to_parquet \
  mixture=standard split=both \
  artifact_format=structured_nanochat_rollout_v1 \
  output_dir=/absolute/path/to/structured-rollouts
```

Legacy conversion remains the default. Structured and legacy loaders reject
the other format immediately. Publication is atomic, and resume audits every
completed source row against its complete token/mask/span/content digest.

## Benchmark-focused subset

Select `mixture=chatcore` when converting to focus GRPO on these four benchmarks:

| Task | RL training source | Held-out validation source |
|---|---|---|
| ARC-Easy | `ARC-Easy/train` | `ARC-Easy/test` |
| ARC-Challenge | `ARC-Challenge/train` | `ARC-Challenge/test` |
| MMLU | `auxiliary_train/train` | `all/test` |
| GSM8K | `main/train` | `main/test` |

Training contributes one pass over retained rows, weighted by source size and
mixed with the existing seed-42 shuffle. Before mixing, training questions are
compared against all four held-out sources after Unicode NFKC, case folding,
and whitespace normalization. Matches are removed regardless of choices or
answers: official ARC/MMLU splits contain overlapping questions. Filtering
preserves source order, validation rows, and the existing task renderers
(including GSM8K calculator parts). Exclusion counts are logged; an entirely
empty training mixture fails. There are no repetitions, caps, or other tasks.

```bash
uv run python -m scripts.preprocess.convert_sft_to_parquet \
  mixture=chatcore split=both \
  artifact_format=structured_nanochat_rollout_v1 \
  output_dir=/absolute/path/to/chatcore

uv run torchrun --standalone --nproc_per_node=1 \
  -m scripts.train.conditional_encdec_sft_grpo -- \
  config_name=conditional_encdec_sft_grpo_chatcore \
  base_checkpoint_dir=/absolute/path/to/conditioned-sft-checkpoint \
  train_artifact_dir=/absolute/path/to/chatcore \
  validation_artifact_dir=/absolute/path/to/chatcore_val \
  global_conversations_per_step=1 group_size=4 \
  device_conversation_batch_size=1 trajectory_microbatch_size=1 \
  num_iterations=10
```

The example's batch sizes are illustrative. The preset changes only
`segmentation_mode=turn_delimited_fixed_length` and `min_reward_tokens=1`,
allowing conditioning before short answers and admitting one-letter answers
plus their end token. Base fixed-length/eight-token defaults reject these rows;
context truncation can still make a row ineligible.

Artifact paths select the subset. Convert other mixtures into their own
structured train/val artifacts; existing `standard` artifacts cannot be filtered
at runtime. Loaders check split tags, and checkpoint resume identity binds both
artifacts, rejecting a changed subset.

Validation uses masked teacher-forced CE reward, consuming
`validation_conversations` new eligible rows per rank per evaluation (default:
one). Both cursors advance without cycling and fail on exhaustion. Use the
separate ChatCORE evaluator for benchmark accuracy. Question matching does not
detect paraphrases or prior model exposure; selecting models on these test
splits leaves no untouched final test set.

`scripts.preprocess.count_tokens mixture=chatcore` estimates retained-source SFT
tokens with separate ARC labels; it does not estimate GRPO replay compute or
post-truncation eligibility.

## Environment and causal routing

The environment keeps the checkpoint context prefix and reconstructs
`inputs=[BOS,*targets[:-1]]`. `target_mask` is applied directly to targets; it
does not pass through the legacy BOS-aligned completion-mask shifter.

`fixed_length` chunks the target stream without respecting roles.
`turn_delimited_fixed_length` starts a new segment at every logical turn and
chunks each turn independently. Both default to 256 tokens. The segment map is
input-aligned: for targets `A B | C D`, inputs are `BOS A B C` and their slots
are `0 0 0 1`. Consequently C1 first affects the prediction of D, not C.
Both chunking modes use the dependency-light shared primitives in
`steering/segmentation_primitives.py`, re-exported by `steering/segmentation.py`;
their behavior is unchanged.

Rows are refilled deterministically after truncation if they retain at most one
logical turn, at most one segment, or fewer than `min_reward_tokens` valid
reward targets.

## Policy and objective

The base must be a conditioned-SFT-shaped encoder-decoder checkpoint with a
final-layer `MAX_POOL` feature-prediction head, cross-attention at or before the
capture layer, exactly two buckets, recoverable positive and negative geometric
count lineage, zero label dropout, and an exact mask/cluster/feature-vocabulary
contract. Compact head rows are scored in compact space and translated back to
original SAE IDs only at the encoder boundary.

`weighted_without_replacement` samples ordered positives and then ordered
negatives from stable FP32 log-sigmoid weights and replays their exact ordered
likelihood. `bernoulli_topk_bottomk` classifies each row independently, then
takes raw-logit top/bottom subsets with original-ID tie breaking; it uses the
documented selected-feature surrogate. Empty decisions stay in the trajectory
mean with a graph-connected ratio of one.

`action_policy_version=2` is the weighted-policy default. It draws counts and
ordered rows with one deterministic RNG stream per trajectory microbatch,
reducing generator construction and CUDA multinomial launches while preserving
each row's target probability law. This intentionally changes the seed-to-row
mapping and makes trajectory microbatch geometry part of sampling identity.
Version 1 retains the previous per-trajectory stateless mapping only for exact
resume of v1 runs. Behavior terms are still recorded from the sampled trace and
must reproduce under differentiable replay within the configured FP32 tolerance.

Reward is detached negative teacher-forced CE, assistant-only by default.
Advantages use FP32 population variance within each rank-local conversation
group only. There is no PPO epoch, ratio clipping, packing, or replay buffer.
`DistMuonAdamW` performs the only distributed gradient averaging; the model is
not wrapped in DDP.

### Training scope

`train_feature_head_only=true` is the default in the base, pilot, and ChatCORE
presets, with `ntp=null`. It trains the checkpoint's complete `feature_pred_head`,
including its output projection and any nested residual blocks. The encoder,
decoder, embeddings, cross-attention, LM head, and model residual scalars are
frozen. This option does not construct or upgrade the head architecture. The KL
reference remains fully frozen in every scope.

Set `train_feature_head_only=false` for the legacy full-policy scope: policy-only
GRPO trains every parameter except the untied LM head; auxiliary NTP also trains
that LM head. Every non-null `ntp` block requires this explicit override.
Head-only mode rejects NTP before runtime setup because detached integer
conditioning actions give it no differentiable path into the feature head.

Frozen parameters are excluded from optimizer groups and state. The existing
head AdamW group uses `unembedding_lr * (decoder_width / 768)^-0.5`, then
`lr_multiplier`, `init_lr_frac`, and the existing LR schedule. Global gradient
clipping covers the trainable head only, so its updates need not equal the head
updates in a full-scope run even when their unclipped head gradients match.

For exact resume, retain the same scope and add, for example,
`resume_from_step=10`. Resuming an older full-policy checkpoint requires
`train_feature_head_only=false` under the new preset default. The ordered
trainable manifest already pins the scope:
changes in either direction fail before installing resumed model or optimizer
state. Changing scope requires a fresh run in a separate output directory;
there is no optimizer-state migration.

Startup logs report the resolved flag and trainable/total parameter-element
counts. New resolved configurations record the flag in the loop's `user_config`;
no new resume-identity field or checkpoint schema is introduced.

### Auxiliary next-token prediction

Auxiliary NTP requires `train_feature_head_only=false`. Supplying an `ntp` block
then adds differentiable next-token CE for the first `k` trajectories of each
collected group before the same clip and optimizer step.
Each trajectory uses one batch-1 full-LM forward over its complete sampled
conditioning history. It reuses the reward mask and mean-over-valid-targets
reduction, so with collection microbatch size 1 its unweighted loss is exactly
the negative collected reward. The LM head is trainable in this mode; the FP
head receives gradients only from policy replay because integer conditioning
IDs carry no NTP gradient. Both objectives meet only in the shared trunk.

The batch-1 loop minimizes peak memory, but a 1024-token, 32,768-vocabulary
FP32 logit/softcap graph is still roughly 0.4 GiB. The per-step compiler reset
also means this added backward can recompile just as policy replay already can.
Validation is unchanged: its existing `masked_ce` is the mean NTP loss over all
rollouts because reward is negative masked CE.

`ntp` is a block, not a flag: omit it (or set it to `null`) and the objective
does not exist. Supplying it enables the objective and must define every field,
so a typo cannot silently fall back to a default. There is nothing to prove inert
inside the block, and a disabled run keeps the policy-only parameter manifest,
the literal `1.0/C` loss scale, six timing-row names, optimizer groups, and a
byte-exact resume identity.

`lr_decay_start` sits outside the block because it is not part of the auxiliary
objective: it is the RL run's own learning-rate decay, always active. Both it and
`ntp.decay_start` name the fraction of the run at which their schedule leaves its
flat region and decays linearly to zero by the final step -- the knob says WHEN
decay begins rather than scaling a horizon. `0.8` is the historical shape, `0.0`
decays across the whole run, and `1.0` never decays within it. Setting
`ntp.decay_start` earlier than `lr_decay_start` anneals the auxiliary objective
away relative to the policy over the intervening steps.

The policy objective carries no loss weight at all, because a COMMON rescale of
both objectives is inert: Muon orthogonalizes, AdamW is scale-invariant, and the
global clip renormalizes on top. Measured on parameter updates for one real
optimizer step, `(w_pg, w_ntp)` `(2, 1)` and `(1, 0.5)` differ by `9.6e-8` —
float noise — while `(2, 1)` and `(1, 1)`, which change the ratio, differ by
`2.9e-1`. So the mixture is fully described by `ntp.loss_weight` alone.

That ratio is a real knob and it moves everything, the LM head included: raising
`ntp.loss_weight` from 1.0 to 3.3 changes the LM head's update by `7.1e-2`. What
it is NOT is a per-parameter step size. The step size comes from the optimizer's
learning rates, which follow the repo's RL convention from `chat_rl.yaml`:
`init_lr_frac 0.05` (a 20x cut on every group) and `embedding_lr 0.2`. On the
400M model that puts the LM head at `2.0e-4`, against `4.0e-3` for conditioned
SFT; raise `init_lr_frac` to `1.0` to train the readout at the full SFT rate.

One thing a disabled run does not preserve byte-for-byte is the saved
`meta_*.json`: its `user_config` block echoes the whole resolved config, so it
gains the `ntp` block and `lr_decay_start`. Resume is unaffected — `assert_grpo_resume_identity`
compares only the `resume_identity` block, and `optimization` is built from an
explicit key allowlist — but a GRPO output checkpoint's own
`_resolved_checkpoint_sha256` differs from a pre-change one, which matters only
if such a checkpoint is later used as another run's `base_checkpoint_dir`.

Enabling NTP still changes the effective policy step: a single global
`clip_grad_norm_` renormalizes the *sum* of both gradients, so the larger
pre-clip norm shrinks both contributions together. The mixture ratio is
preserved, so the averaging invariants are untouched, but a policy-only run and
an NTP run at identical settings are not step-for-step comparable.

The `1/k` mean is exact in each parameter's own dtype, but a conditioned-SFT
checkpoint stores its embedding tables in BF16 (`transformer.wte`,
`value_embeds.*`, and the encoder feature/magnitude embeddings; everything else
is FP32). `.grad` follows the parameter, so the per-trajectory backwards
re-round those tables on every accumulation, and raising
`ntp.trajectories_per_group` shifts them slightly. This is not specific to the
auxiliary objective: the policy replay already accumulates one backward per
trajectory microbatch into the same tables, and measurably drifts by the same
order of magnitude at the same accumulation count. Treat `k` as a
variance-reduction knob rather than a bit-exact one; the FP32 parameters,
including both heads, are unaffected.

`kl_beta=0` is an exact no-reference path. A positive beta loads a frozen copy
of the immutable pre-RL checkpoint and applies all-feature current-to-reference
Bernoulli KL during replay.

## Initial conditioning

`initial_conditioning_mode=none` is the default and required for the first
experiments. It creates the canonical all-masked C0 without invoking an SAE
resolver, probe, tokenizer, model, CUDA allocation, forward, or SAE RNG.

`online_sae` is explicit. A conditioned-SFT checkpoint must carry
`structured_sft_online_sae_lineage_v1`, including the exact feature directory,
statistics digest, SAE coordinates, activation site, source segmentation,
selection policy, and missing-percentile policy. The provider accepts only
context-independent, no-BOS, max-pooled, fixed-token source lineage;
turn-delimited conditioned-SFT lineage is rejected. It lazily retains one probe
per rank, decodes the complete first RustBPE segment cumulatively, excludes
injected and naturally occurring Gemma BOS rows, max-pools SAE activations,
reproduces persisted FP16 magnitude semantics, and delegates mask/cluster/
percentile/positive/negative/bucket processing to the conditioned-SFT selector.
Missing or conflicting lineage fails; an empty extraction remains labeled
`online_sae` and is counted as an error.

`segmentation_mode` governs only target-stream chunking and is independent of
source lineage. C0 always comes from rollout segment 0, whose length is
unrelated to `source_segment_size` in either mode.

## Training and exact resume

The base config deliberately leaves hardware geometry mandatory:

```bash
torchrun --standalone --nproc_per_node=1 \
  -m scripts.train.conditional_encdec_sft_grpo -- \
  base_checkpoint_dir=/absolute/sft/checkpoints model_step=500 \
  train_artifact_dir=/absolute/structured/train \
  validation_artifact_dir=/absolute/structured/val \
  global_conversations_per_step=1 group_size=256 \
  device_conversation_batch_size=1 trajectory_microbatch_size=1 \
  num_iterations=1
```

`conditional_encdec_sft_grpo_pilot` pins only `group_size=256`; it does not
claim that the remaining geometry is safe or performant. Groups never split
across ranks, trajectory microbatches divide the group, and the local loss is
scaled once by the local conversation count before the optimizer's rank mean.

`profile_timing=true` writes synchronized rank-local attribution rows for data
staging, collection, differentiable replay/backward, gradient clipping,
optimizer work, and cursor commit. The synchronizations perturb overlap, so a
timing profile and a throughput benchmark are separate experiment arms.

CUDA GRPO resolves and pins the attention backend from the allocated device:
FA4/CuTE on Hopper or newer, and SDPA on earlier devices such as L40S. A
Hopper-class allocation without a working FA4/CuTE backend fails before model
or artifact construction rather than silently falling back. The resolved
backend and deterministic-algorithm mode are part of resume identity.
PyTorch's FA4 block-mask backward cannot run under hard deterministic mode on
SM90, so H100/H200 policy replay and NTP backward use FA4 with deterministic
algorithms disabled. SM12 uses native CuTe causal/sliding-window attention
through a project-owned adapter because FlexAttention block masks are
unsupported there; native SM12 backward also requires deterministic algorithms
to be disabled. Both trainers and the standalone RLVR evaluator use this
admission, including after each checkpoint exec. Cached token decoding and packed encoder/cross-attention
keep their existing implementations. SM100 FA4 and pre-Hopper SDPA retain hard
deterministic mode. Attention-local compilation
remains enabled, while the whole encoder-decoder model remains uncompiled.

Data cursors are rank-strided and transactional. Attempted/accepted positions
and ordered refill buffers commit only after a successful optimizer step.
Checkpoints are saved only at step boundaries beneath
`selfcond_grpo_checkpoints`; model keys and shapes remain deployment-compatible,
while a versioned loop block pins artifacts, topology, geometry, policy,
parameter manifest, optimizer settings, conditioning identity, reference/KL,
and both train and validation cursors. Resume requires every pinned field and
the same world/rank topology. Every durable nonfinal checkpoint synchronizes the
ranks and re-executes each worker from that checkpoint. Normal continuation and
recovery initialize NCCL in a fresh store namespace derived from the checkpoint
step and torchrun restart generation, so pre-exec bootstrap data cannot be
reused. Both start from the same process-local CUDA state instead of
comparing a long-lived worker with a fresh one. SDPA and SM100 FA4 retain exact
resume; SM90/SM12 FA4 require explicit numerical checks because their
backward uses nondeterministic reductions. For action-policy v2, identical
sampling also requires the same trajectory microbatch geometry; v1 and v2
resumes are never interchangeable.

GRPO metadata is a superset of the conditioned-SFT checkpoint shape, so the
existing chat loader continues to select `build_engine_self_conditioned`; no
generation policy is added here. Paired pre/post evaluation should use the
existing fixed-prompt chat evaluator and the upstream AxBench direct c2
`SelfConditioning` path. Do not introduce a GRPO-specific deployment method or
combine its results with artifacts produced under another policy identity.

## Qualification boundary

Before treating any preset as operational evidence, run real structured
conversion/reload, both action and segmentation modes, one- and multi-rank CUDA
rollouts, G=256/microbatch-1 memory, same-topology resume, and the staged reward
and benchmark protocol. Queueing, allocation, synthetic tests, or a smaller
group are not substitutes. Paid AxBench judge calls remain separately
approval-gated.
