# nanochat/ — Vendored. Do Not Modify.

This directory is a **vendored copy** of [Andrej Karpathy's nanochat](https://github.com/karpathy/nanochat) (MIT License — see `LICENSE` in this directory). It is treated as **read-only** by this project.

## The Rule

**Do not edit any file under `nanochat/`.** This includes bug fixes, refactors, formatting, comment edits, and adding type hints. If you think you need to, **stop and open an issue first.**

The maintainer will decide whether to:
1. Backport the fix from upstream nanochat
2. Wrap the desired behavior in a `steering/` adapter
3. Make a documented exception (rare; requires a deliberate review)

`.github/CODEOWNERS` enforces this: any PR touching `nanochat/` requires the maintainer's review.

## Why this rule exists

- **Diff hygiene with upstream.** Local edits make it painful to pull future improvements from karpathy/nanochat.
- **License attribution.** This is third-party code under its own MIT license.
- **Boundary clarity.** Treating `nanochat/` as a library forces clean interfaces in `steering/` instead of in-place hacks.

## What this directory contains

- The vendored framework code (`gpt.py`, `optim.py`, `tokenizer.py`, `dataloader.py`, `dataset.py`, `engine.py`, etc.)
- `LICENSE` — upstream MIT license (kept here for attribution)
- `pyproject.upstream.toml` — upstream project config, kept for reference (the active project config is the top-level `pyproject.toml`)

## What goes in `steering/` instead

If you need behavior that isn't in vendored nanochat, build it in `steering/` and import the nanochat pieces you need. Examples of acceptable patterns:

- A new training loop variant → `steering/conditional/encoder_decoder/`
- A new optimizer wrapping `nanochat.optim` → `steering/<your_module>.py`
- Shared utilities (atomic IO, validation, etc.) → `steering/_common/` (create the dir if needed)

## Documented exceptions

The do-not-modify rule has been deliberately, reviewably waived for the
following edits. Each was a conscious decision (CODEOWNERS routes every
`nanochat/` change to the maintainer, which is the "deliberate review").

| Date | File / symbol | What | Why a `steering/` wrapper was not used |
|------|---------------|------|----------------------------------------|
| 2026-09-19 | `checkpoint_manager.py` — feature configuration reconstruction in `build_encdec_model` | Issue #1719, owner-authorized extension of the #1681 exception: delegate to the canonical project parser, require explicit `block` and `head_layers`, and reject conflicting mirrors in local metadata and any selected ancestor before recovering its head configuration. Supersedes implicit legacy-depth loading; old checkpoints require the explicit project migration utility. | Strict loading constructs the project model inside this canonical reader, before a wrapper can intervene. Only feature-configuration reconstruction changes; unrelated checkpoint recovery and vocabulary checks are retained. Transformer blocks, attention admission, migration and cache integration remain in `steering/`. CODEOWNERS review remains required. |
| 2026-09-18 | `checkpoint_manager.py` — `build_encdec_model` | Issue #1681, owner-authorized narrow exception: forward the saved `mlp_layers` field when reconstructing `FeaturePredictionConfig`, defaulting missing legacy metadata to zero. | The canonical loader manually reconstructs the configuration before it constructs the project-owned model. A wrapper receiving the loaded model cannot recover a residual head that strict loading already rejected. All head implementation and initialization remain in `steering/`; CODEOWNERS review remains required. |
| 2026-05-15 | `common.py` — `compute_init`, `_resolve_nccl_timeout_seconds` | Apply an `NCCL_TIMEOUT_SEC` override to the DDP process-group timeout **only when the env var is set** (PyTorch's backend default ~10 min NCCL / 30 min GLOO is preserved otherwise — no global default change); validated parser (positive int, overflow-of-`timedelta` rejected). | The timeout must be passed to `dist.init_process_group`, which is called *inside* `compute_init`. `compute_init` does not check `is_initialized()`, so a wrapper cannot pre-init the PG without a double-init error, and there is no other seam. Every DDP entrypoint already funnels through `compute_init`, so the fix belongs there. Scope is strictly the timeout knob + its validation; no other behavior changed.
| 2026-07-11 | `checkpoint_manager.py` — `build_encdec_model` (+ the module import block) | Issue #695 (encoder feature-dictionary compaction). Make `build_encdec_model` vocab-aware: recover `(n_sae_features, feature_vocab)` from the state dict via `feature_vocab_from_state_dict(model_data)` instead of inferring `n_sae_features = model_data["encoder.feature_embed.weight"].shape[0]`; pass `feature_vocab=` into the `EncoderDecoderGPT` construction; cross-check the meta's `feature_vocab` fingerprint against the load-bearing vocab buffers (absence is fine, **disagreement** is fatal) and call `model.assert_vocab_consistent()` after the load. Separately, the import block was reordered so `torch.serialization.add_safe_globals([OptimizerKind])` runs **below** the imports. | Once the encoder's feature dictionary is compacted, `feature_embed.weight.shape[0]` is **V** (the compacted row count), not `n_sae_features` — so the old inference silently redefines the whole feature space. A `steering/` wrapper cannot intercept it: the inference happens *inside* `build_encdec_model`, before the model is constructed, and callers only ever receive an already-built model — by then `model.encoder.config.n_sae_features` is already wrong. It is also THE loader every eval consumer funnels through, so leaving it would silently corrupt every consumer that bounds-checks ORIGINAL feature ids against `model.encoder.config.n_sae_features` (`eval_adapter`, `target_conditioning.validate_feature_dict`, `model_detection`, `harmfulness_eval`) — a wrong answer, not an error. The import reorder was forced by the project's imports-at-top rule: `add_safe_globals` sat *interleaved* with the import block, leaving six module-level imports below executable code. Registration happens at module import either way, so the move is behaviour-preserving. Legacy (uncompacted) checkpoints carry no vocab buffers and load exactly as before.
| 2026-07-13 | `engine.py` — `__main__` demo block (+ the module import block) | Issue #695 Task 3 (eval-conditioning load-time contract). The `__main__` smoke-test calls `load_model("base", device, phase="eval")`; threaded an explicit `eval_conditioning=EvalConditioning.deferred("engine_demo")` so this call complies with the new required-stance contract once Task 4 removes the transient default. | `deferred(...)` — not `not_evaluating()` — because `load_model` auto-detects an encoder-decoder checkpoint from on-disk metadata even under the bare `base_checkpoints/<tag>/` root that `"base"` resolves to (see `_is_encdec_checkpoint`, proven by `test_base_source_with_encdec_meta_uses_encdec_builder`), so this load CAN reach `build_encdec_model` at EVAL. `deferred` loads a legacy encdec checkpoint unchanged and defers only a COMPACTED one; `not_evaluating()` would raise on any encdec here — including a legacy one that loads fine today. A `steering/` wrapper cannot intercept it: the call is a literal demo snippet inside `nanochat/engine.py`'s own `if __name__ == "__main__":` block, not a call a wrapper module could shadow.
| 2026-07-14 | `checkpoint_manager.py` — the four model-load funnels (`build_encdec_model`, `load_model`, `load_model_from_dir`, `load_encdec_model_from_dir`) + a small import/alias block | Issue #695 eval-conditioning load-time contract (this PR — distinct from the 2026-07-11 vocab row above). Added a REQUIRED keyword-only `eval_conditioning: EvalConditioning` to all four loaders and, on `build_encdec_model`'s EVAL path, call `eval_conditioning.assert_loadable(model.feature_vocab, checkpoint_dir=...)` — replacing #703's blanket compacted-EVAL `NotImplementedError`. Also typed the four signatures (params + `-> tuple[<model>, <tokenizer>, dict]`) and added `LoadedModel` / `LoadedTokenizer` aliases, re-importing `EvalConditioning` for the annotation. | These four functions ARE the single funnel every eval / train / serving load goes through; the contract must be enforced where the model is built and dispatched (inside `checkpoint_manager`), and a `steering/` wrapper cannot make the keyword *required* on the real functions callers import. The type annotations are documentation/IDE-only (`nanochat/` is excluded from `ty`) but capture the real load contract.
| 2026-08-08 | `checkpoint_manager.py` — checkpoint source resolution | Issue #1282: resolve encoder-decoder SFT segments through bounded mask, gradient-attribution, and feature-cluster leaves; remove dead unconditioned aliases; add `standard_cond`; and forward explicit encoder-decoder path pins from `load_model`. | These resolver functions are local additions with no upstream counterpart. A `steering/` wrapper would leave the public `source=` path sharp and require every evaluation entry point to bypass the canonical loader. The CODEOWNERS-reviewed PR is the approval gate for this in-place waiver. (#1282)
| 2026-08-27 | `tokenizer.py` — `LogicalTurnSpan`, `StructuredConversationRender`, `RustBPETokenizer.render_structured_conversation` and private serialization helpers | Add an opt-in, untruncated structured SFT renderer with authoritative RustBPE IDs, a BOS-aligned assistant mask, logical role spans, strict tool grammar, and canonical generic-tool wire text. The legacy renderer, completion renderer, and vocabulary are unchanged. | Token IDs and masks must be produced by the tokenizer itself: decoded special-token spellings do not re-encode to the same IDs, and a project wrapper could only duplicate the vendored renderer's private framing/tokenization logic. The additive method is the narrow seam that lets project-owned artifact and GRPO code consume one authoritative render. CODEOWNERS review remains required. |
| 2026-08-30 | `checkpoint_manager.py` — `load_checkpoint` | Issue #1598 item B (owner-authorized 2026-08-30): add an opt-in rank-0 model read followed by an error-first schema/tensor broadcast for all-rank encoder-decoder resume. Optimizer shards and metadata remain per-rank reads; the default path is unchanged. | The checkpoint heartbeat and rank-skew timing seam already lives inside the canonical vendored loader (#1568 precedent), and the model archive must be read before project-owned code receives the returned state dict. A wrapper would duplicate checkpoint loading or bypass the existing safety, heartbeat, and timing contracts. The collective is default-off so rank-conditional callers remain safe. |
| 2026-08-30 | `flash_attention.py` — `flash_attn_func` | Issue #1601: make only short, non-FA4 FlexAttention queries contiguous before PyTorch auto-selects FlexDecoding, preventing silent wrong output for the standard transposed FP32 query layout on PyTorch 2.13. FA4 and query lengths at or above 128 preserve their original layout. | Every base and encoder-decoder self-attention layer calls the vendored module-level `flash_attn` API directly. A project wrapper would require global monkeypatching at every model-loading and evaluation entrypoint, could miss consumers, and would leave the canonical fallback unsafe. The exact boundary guard is therefore kept at the single shared dispatch point; CODEOWNERS review remains required. (#1601) |

## Found a real problem?

Open an issue with the title prefix `[nanochat]`. Include:
- The exact upstream file + line
- What's broken / surprising
- Whether the same issue exists in karpathy/nanochat HEAD

Don't open a PR against `nanochat/` — it will be closed.
