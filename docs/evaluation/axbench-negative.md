# Natural-chat negative AxBench

This evaluation measures feature suppression on original conversations that
empirically elicit a concept under ordinary API sampling. It preserves the
conversation, obtains a fixed set of neutral responses, measures their SAE
activity, and constructs a counterpart that passes strict-zero SAE scoring and
a separate usefulness judgment. A frozen bank then supplies the same conversations to native decoder and encoder-decoder
targets, each with fresh unsteered controls.

The producer is separate from Base/BPB Stage 0c. That stage's default API
generator conditions on the feature and can insert future source text and
repair continuations. None of those operations belongs in neutral sampling.
Existing positive AxBench and Base/BPB artifact formats remain unchanged.

## Prepare original conversations

The supplied preparation profile pins UltraChat `test_sft` to revision
`8049631c405ae6576f93f445c6b8166f76f5505a` and verifies its Parquet SHA-256
`c18fd6e77395577652bdefbc5a87044ea799e989451ecfca5c6cf977ae5c6f70`.
It selects the first N original rows before filtering. It records rejected
structures without replacing them with later rows.

```bash
uv run python -m scripts.eval.axbench_negative_prepare \
  row_count=64 output_dir=out/axbench-negative/ultrachat-test64
```

Use `snapshot_dir=/path/to/pinned/snapshot` to read an existing local snapshot.
The output contains `source.jsonl` and `source_manifest.json`; the manifest
records the original row ordinals, prompt IDs, revision, file hash and selection.
Every selected conversation remains in the test split.
The current loader materializes the selected records and their legal cuts.
Use `row_count` to bound that population; the later per-feature candidate cap
bounds screening requests, not source-loading memory.

Other corpora can provide strict local JSONL records:

```json
{"source_id":"dataset/revision/row0","group_id":"conversation-family0","split":"test","source_row":0,"messages":[{"role":"user","content":"Original question"},{"role":"assistant","content":"Original answer"}]}
```

The source supports optional leading system text and alternating user/assistant
text messages. It rejects unsupported fields, tools, multipart content, blank
messages and malformed role order. It preserves whitespace and Unicode.
Each user-ending cut becomes a candidate; later source answers are omitted.
All cuts/variants of a group must share a split. Exact conversation/prefix
duplicates cannot cross splits. Group IDs must represent actual source families;
the validator cannot discover undocumented paraphrase relationships.

## Build a reference bank

Configure `configs/eval/axbench_negative/build.yaml`. Required inputs are the
source path/hash, prepared AxBench metadata path/hash, concept IDs, native
tokenizer, both target checkpoint metadata paths, encoder-decoder feature
statistics and any required pretraining metadata, and explicit API models.
The default detector matches the canonical prepared AxBench concepts:
Gemma-2-2B, residual layer 20, canonical 16k GemmaScope SAE.

The new Hydra profiles use direct OpenAI Chat Completions with `OPENAI_API_KEY`.
Set explicit OpenAI model IDs for the neutral sampler and quality judge;
avoidance inherits the neutral provider and model. A pinned snapshot such as
[`gpt-4.1-2025-04-14`](https://developers.openai.com/api/docs/models/gpt-4.1)
supports this profile's non-reasoning sampling contract. Account access still
needs verification before a run. To use OpenRouter, set `neutral.provider=openrouter`
and `quality.provider=openrouter`, and provide `OPENROUTER_API_KEY`; evaluation
judges use the separate `judge.provider` setting. Each provider reads only its
own key and uses a fixed endpoint. There is no credential fallback or arbitrary
endpoint override. Existing programmatic API configs that omit `provider`
retain OpenRouter behavior.

```bash
uv run python -m scripts.eval.axbench_negative_build \
  source_path=/path/to/source.jsonl source_sha256=<sha256> \
  source_manifest_path=/path/to/source_manifest.json \
  metadata_path=/path/to/prepared/generate/metadata.jsonl \
  metadata_sha256=<sha256> 'concept_ids=[0,1]' \
  render_tokenizer_path=/path/to/tokenizer/tokenizer.pkl \
  target_admission.0.meta_path=/path/to/native/meta_001001.json \
  target_admission.1.meta_path=/path/to/encdec/meta_000749.json \
  target_admission.1.feature_statistics_path=/path/to/feature_statistics.safetensors \
  target_admission.1.pretrain_meta_path=/path/to/pretrain/meta_054600.json \
  neutral.model=<sampling-model> quality.model=<judge-model> \
  output_root=out/axbench-negative/bank
```

The target list can contain either family or both. Each entry binds model,
metadata and tokenizer bytes and reserves `max_new_tokens` inside its context.
Candidates must fit every configured target, including the encoder-decoder's
separate conditioning tokenizer and extraction window. Set its
`pretrain_meta_path=null` only when the SFT metadata already supplies the complete
conditioning configuration. Assessment extraction has its own tokenizer
and context limit; `reserved_extraction_tokens` reserves headroom before API
calls. Actual responses still undergo complete-length validation. No prompt or
answer is silently truncated to obtain a passing measurement.

Native admission applies the direction-method consumer's reviewed geometry:
12 layers with width 768 or 14 layers with width 896, both with context length
1024, in addition to the native SFT metadata checks.

Encoder-decoder admission resolves feature-head metadata before API sampling.
If local metadata omits the head, it uses the target loader's closest-ancestor,
latest-checkpoint lookup and validates both ancestor and recovered child
metadata. The resolved head must record `block` and `head_layers` and use
`loss_mode=max_pool`; missing or incompatible heads are rejected before screening.
For a legacy architecture-schema error, use the existing
[metadata conversion procedure](../pipelines/feature-head-metadata-migration.md)
in a new checkpoint view:

```bash
mkdir /path/to/converted
cp --reflink=auto /path/to/original/model_000749.pt /path/to/converted/model_000749.pt
uv run python -m steering.tools.migrate_feature_head_metadata \
  source_metadata=/path/to/original/meta_000749.json \
  weights=/path/to/converted/model_000749.pt \
  output_metadata=/path/to/converted/meta_000749.json
```

The converter validates the paired tensor layout and refuses an existing output.
Preserve original metadata and weights, record both metadata hashes, and verify
that the copied model hash is unchanged. Convert legacy ancestor metadata when
needed while preserving the loader's ancestor-directory layout. Point build and
evaluation at the converted metadata; changed metadata identity requires a fresh
bank. Conversion does not make an explicitly incompatible loss mode suitable
for SelfConditioning; do not invent architecture fields or change the saved loss
mode to bypass admission.

The default screening profile uses `n_verification_seeds=5` and
`min_consistency_passes=3`. These name k draws and m required hits; internal
slot IDs are not provider RNG seeds. An optional sampling seed is a master for
distinct, deterministic per-call seeds outside the messages. The sampler sends
only the unchanged original messages plus explicit API parameters. The same
draws can be assessed against multiple features, so those estimates share samples.
Requested sampling settings and returned provider metadata are retained; cold
API calls are not guaranteed reproducible. See the provider's
[OpenAI Chat Completions contract](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
or [OpenRouter parameter contract](https://openrouter.ai/docs/api_reference/parameters).
The transport sends the configured sampling fields unchanged, so choose a model
that supports them. Provider, endpoint, model and sampling settings bind the
API identity and caches; changing them requires a fresh run contract.
The k/m rule is an empirical screen, not a guarantee of the true occurrence
frequency or that rich answers are more natural than avoiding answers. It measures
elicitation under the configured API sampler; it does not establish the same
prevalence under either target model. Fresh target controls measure that separately.

Every logical slot survives in the evidence:

| Outcome | Screening meaning |
|---|---|
| Complete nonempty response | Measure SAE activity; ACTIVE is a hit. |
| Complete refusal text | Measure raw activity; separately reject it as a useful reference. |
| Duplicate response | Keep each slot as an observation. |
| Empty or provider-truncated answer | Terminal invalid nonhit, with available raw diagnostics. |
| Malformed response, missing finish metadata, transport/extraction failure or overflow | Unknown; candidate cannot be admitted. |

A completed miss is never replaced. HTTP retries send the same request.
Unknown outcomes committed to a run are replayed unchanged. A fresh run can
reuse valid cached responses for reassessment while preserving the old evidence.
An interruption between a provider response and its durable cache write cannot
guarantee exactly-once provider execution.

Mining measures activity in included history only. SAE QC uses the full rendered
conversation as causal context, with exact content spans mapped through joint
tokenizer offsets. Completion scoring includes only tokens overlapping the
returned assistant body, including boundary-crossing tokens. It excludes inserted
role framing and every extraction BOS ID. ACTIVE means an activation strictly
greater than zero; no-token output is never a valid inactive counterpart.
This is different from the response-only positive AxBench roundtrip metric.

The scoring renderer is versioned, untruncated native RustBPE rendering, including
its trained system-message folding. Original provider/judge messages are retained
separately. Target conditioning uses checkpoint-governed feature segmentation;
that does not change the full-context SAE measurement.

Current extraction loaders do not expose immutable revision arguments. The
builder therefore binds actual model/SAE tensor and configuration hashes,
tokenizer content and numeric backend before sampling, and verifies configured
expected hashes. Requested model IDs are recorded separately; an unavailable
observed revision remains unknown. Do not interpret a floating model name as a pin.

Only a fully measured candidate with m hits enters reference construction.
An independent quality call checks unchanged rich answers for useful task
fulfillment. A private concept-aware call proposes an avoiding answer, followed
by strict-zero SAE QC and a separate quality call. Avoidance retries are bounded;
neutral draws have no semantic retry. The quality rubric rejects refusals, task
changes, filler and fabricated details. Publication requires positive usefulness
judgments for both rich and avoiding references; those model judgments can be
wrong. Excluded groups retain their screening evidence.

## Evaluate both target families

The v1 methods have explicit suppression semantics:

| Family | Suppression | Unsteered control |
|---|---|---|
| Native SFT decoder | DiffMean/LinearProbe residual update `-magnitude * fit_max_activation * unit_direction` | Same checkpoint with no hook |
| Encoder-decoder | SelfConditioning interpolates toward the negative dense concept vector, weight `magnitude / 6` | Two-sided pure SelfConditioning at weight zero |

Fits use separately authenticated canonical AxBench TRAIN data. Reference-bank
answers never enter fitting. Existing direction-fit numerics remain unchanged.
Encoder-decoder setup retains checkpoint
vocabulary, masks, statistics and cadence checks. Masked SelfConditioning, LoRA
and prompt-based avoidance are not supported v1 methods; their semantics cannot
be obtained by blindly negating a positive factor.

Start from `configs/eval/axbench_negative/eval.yaml`. Supply the same detector
configuration, target checkpoint admitted by the bank, and independent TRAIN
artifact hashes. Set `NANOCHAT_BASE_DIR` to the native tokenizer's cache root.

```bash
uv run python -m scripts.eval.axbench_negative_eval \
  source_dir=/path/to/bank output_dir=out/axbench-negative/native \
  target.family=native 'target.methods=[DiffMean,LinearProbe]' \
  target.checkpoint_dir=/path/to/native/checkpoints target.step=1001 \
  render_tokenizer_path=/path/to/tokenizer/tokenizer.pkl \
  render_tokenizer_sha256=<sha256> \
  'extractor={model:gemma-2-2b,sae_release:gemmascope,sae_width:16k,sae_l0:null,layer:20,activation_type:residual,device:cuda,batch_size:1,context_limit:2048}' \
  target.training.prepared_dir=/path/to/prepared \
  target.training.metadata_sha256=<sha256> target.training.data_sha256=<sha256> \
  target.training.ref_prefix=https://www.neuronpedia.org/gemma-2-2b/20-gemmascope-res-16k \
  judge.model=<judge-model>
```

Keep `extractor` identical to the bank's scoring configuration; for repeated runs
it can live in a local YAML config extending `eval`. For encoder-decoder use `target.family=encdec`,
`target.methods=[SelfConditioning]`, `target.meta_path`, matching
`target.feature_statistics_path`, and the checkpoint's required pretraining
metadata. Use a separate output directory per target. `max_cells` bounds a
resumable pilot without shrinking the declared grid.
Evaluation output, response-cache and fit directories must each be disjoint
from the frozen source directory, including ancestor/descendant paths.

All methods see the same original messages; reference/synthesis text is excluded.
Each family generates fresh controls. Prompt/replicate seeds pair control and
intervention randomness. Cell identities include polarity, exact magnitude,
replicate, target and fit identity. Strength selection belongs on validation;
test results must not select or filter the bank.

Reports keep SAE activity, semantic concept presence, task quality, refusal,
empty/truncated status, length and paired control changes separate. A generation
or unavailable judgment remains in declared coverage. Paired effects state their
observed denominator; missing data never becomes successful suppression.
Budget exhaustion is conservative: the current engine hides the terminating
token, so exhausting the token budget is marked incomplete even if the text
looks finished.

## Artifacts, resume and validation

The bank uses `axbench-negative-source-v1`, source kind
`natural_chat_negative`. Canonical immutable JSON records contain candidates,
mining, every neutral slot/assessment, decisions, rich quality, bounded avoiding
attempts and accepted references. `manifest.json` binds the identity, dependent
file hashes, complete groups and yield counts, and is written last. The existing
SQLite response cache accelerates replay; it is not the source of record.
Consumers authenticate the completed bank before use. Changing source, sampler,
detector, rubric or split policy requires a new compatible identity.
Identities retain actual model, data, tokenizer, configuration and output hashes
and numerical runtime versions. Replay semantics use explicit format/policy
versions: a change in meaning requires a version bump; a harmless refactor does
not. Historical identities also contained implementation-file hash maps. Those
v1 artifacts remain readable, but current reduced run identities require fresh
output roots.

Encoder-decoder fit identities include the extraction model/SAE configurations,
library versions and numerical settings, in addition to weights and tokenizer
content. Generation identities also include the actual loaded feature-head
configuration, including a head recovered from ancestor metadata. A changed
fit identity requires a fresh fit directory; a changed generation identity
requires a fresh evaluation directory. Existing source banks remain readable.
Assessed quality and semantic-presence judgments must retain their provider
responses; loading or resuming rechecks the derived verdict against that
evidence. Rehashing a record with missing or contradictory evidence does not
make it valid.

Run the focused suite with:

```bash
uv run pytest tests/steering/eval/axbench_negative/ tests/scripts/eval/test_axbench_negative_consumer.py -q
```

Implementation and validation history is retained in
[PR #1782](https://github.com/Algorithmic-Alignment-Lab/nano-gpt-pretrain-steer/pull/1782).
The blinded source audit found quality-judge false positives, and the pilot did
not demonstrate useful suppression. Code checks, source quality and target
effectiveness remain separate qualifications; a smoke-only checkpoint cannot
qualify research conclusions.
