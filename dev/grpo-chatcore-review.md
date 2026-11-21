# ChatCORE GRPO subset review — 2026-09-12

PR #1660 incorporates main `fd45aa81daf1af51a81bf94cad9fc7ddc561c6f6`.
Its 39 newer commits add the separate ChatCORE evaluator, with no duplicate
subset or conflicting GRPO changes. Three independent reviewers traced data,
runtime state, and test oracles; no merge blocker or further code change was
found. Existing conversion, rendering, cursor, objective, and checkpoint
machinery is reused.

## Implementation and simplification

Official split names did not prevent question overlap. Training now excludes
questions matching the held-out union after NFKC, case folding, and whitespace
normalization, preserving retained order and validation rows. Token counting
uses the shared builder with strict overrides and distinct ARC labels. An
inherited converter test was corrected to require incomplete metadata after
skip-rate rejection, including on resume; converter behavior is unchanged.

Filtering lives in `tasks/mixtures.py`; its executable AST is identical to the
reviewed version. Tests share offline sources, real RustBPE, conversion helpers,
and the existing GPU checkpoint builder. Native metadata equality and exact
PyTorch tensor comparison replace the custom recursive checker. The redundant
character-tokenizer roundtrip was removed; independent source-order, leakage, tampering, masks, and cursor/replay
assertions remain. Split definitions and usage are in the
[pipeline guide](../docs/pipelines/sft-self-conditioning-grpo.md#benchmark-focused-subset).

## Evidence

- Broad CPU regression: **1,094 passed, 3 skipped, 11 deselected**, with
  `CUDA_VISIBLE_DEVICES=''` to isolate CPU tests from unrelated GPU jobs.
- 434 affected CPU tests passed; two legacy tokenizer-dependent skips and one
  slow deselection. Ruff and targeted ty checks pass.
- Changed production coverage: **32/32 executable statements, 8/8 branch arcs**.
  This is PR coverage, not repository-wide coverage or proof of correctness.
  Coverage was collected separately: the initial broad invocation mixed
  multiprocessing statement/branch data and encountered CUDA OOM in CPU tests.
  The affected distributed tests also failed on untouched main, then passed
  on both branches with CPU isolation; the complete isolated rerun is green.
- All 14 deliberate mutations were detected after a 31-test clean baseline,
  covering normalization, exclusion, order, splits, labels, counting, overrides,
  and short-answer geometry.
- Both extracted checkpoint builders produce exactly identical saved tensors
  and metadata to their previous implementations (CPU comparison).
- All three RTX 4090 actual-trainer tests passed after simplification and the
  main merge (123 seconds): all four benchmark types, policy updates, frozen
  LM head, validation, exact fresh-process model/optimizer/cursor resume, and
  inherited NTP modes/schedule checks. The earlier GPU contention is resolved.
- Independent randomized HF Dataset probes passed 1,500 cases, including a
  hand-labelled Unicode oracle, duplicates, empty sources, idempotence, ordering,
  and validation immutability. All 14 knobs reject inapplicable overrides; five
  deliberately corrupted checkpoint states were detected by exact comparison.

The complete cached-source audit was rerun on the final combined code:

| Source | Raw train | Excluded | Retained train | Held-out |
|---|---:|---:|---:|---:|
| ARC-Easy | 2,251 | 10 | 2,241 | 2,376 |
| ARC-Challenge | 1,119 | 7 | 1,112 | 1,172 |
| MMLU | 99,842 | 35 | 99,807 | 14,042 |
| GSM8K | 7,473 | 0 | 7,473 | 1,319 |
| Total | 110,685 | 52 | 110,633 | 18,909 |

All retained rows rendered with real RustBPE, with zero normalized question
overlap and unchanged held-out rows. At context 1024, 249 long MMLU training
rows were filtered; all validation rows were eligible. Every accepted first
answer was conditioned without answer inputs in its preceding decision segment.
Machine-local scripts, source revisions, coverage, and validation logs are in
`out/grpo-chatcore-final-review/` (earlier evidence: `out/grpo-chatcore-review/`).

Exact matching does not detect paraphrases or prior model exposure. These
held-out splits also serve ChatCORE evaluation; model selection on them leaves
no untouched final test set. Validation samples advancing rows using CE reward,
not full benchmark accuracy. Tiny-model GPU testing does not qualify production
scale or multi-GPU training.
