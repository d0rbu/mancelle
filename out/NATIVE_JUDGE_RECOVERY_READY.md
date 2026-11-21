# Native standalone judge recovery ready for root review

Prepared in the isolated `native-judge-recovery-20260923` checkout, based on
`6fd8e390896247653f70af616a3fc9148584b2ab`. No provider calls, GPU jobs, SSH,
uploads, PRs, merges, or live-source/artifact changes were performed.

## Exact cause and evidence

The blocker is the shared zero control (`points/p0`), not a GEPA prompt mutation
or method-specific cache key. `native_methods._selected_raw` correctly retains
standard100's original paid concept/fluency completions. The old
`native_method_judge._judge_point` nevertheless sent that control through
`run_final_judge` using the separate adversarial job cache. Those repeated
judgments produced different completion text; temperature zero does not make
independent provider responses byte-identical. The completion guard correctly
rejected them.

Read-only evidence under
`/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/out/fp-head-campaign-20260921/method-completion-400m/decoder-adv-jobs/`:

| Saved job | Failure in run.log | Zero concept completions differing | Zero fluency completions differing | Nonzero concept/fluency exact matches |
| --- | --- | ---: | ---: | ---: |
| decoder-c000-001 | line 392, DiffMean/c0/i0 | 28/30 | 30/30 | 720/720 |
| decoder-c001-002 | line 294, DiffMean/c1/i0 | 29/30 | 30/30 | 720/720 |

All 26 point inputs and emission files pass their recorded identity/hash and
frame/emission checks. The audit covers DiffMean, LinearProbe and LoRA, GEPA
p1/p2/p4/p8, and both additive counters at 0.25/0.5/1/2. The actual vendor
LanguageModel/JudgeCache and LMJudgeEvaluator reproduced both failures with
90/90 cache hits per job, without a provider client. Both repaired controls
then passed the unchanged final validator using the exact original union.
The original cache SHA-256 values were identical before and after the tests.
Private details and logs remain local; `out/recovery-ready-audit.json` contains
only counts, coordinates and source hashes.

## Fix and identity

The zero control now replays all three original paid judgments from `final_root`,
the standard100 union already authenticated and hashed by generation. The
repair checks its recorded provenance SHA, exact file inventory and hashes,
source JSONL/Parquet agreement and arithmetic, and exact selected input prompt,
concept, factor and generated text. It copies original per-row ratings,
completion text, and factor summaries into a new sparse judged artifact. The
ordinary final validator then enforces exact worker completion/rating parity,
sparse availability, JSONL/Parquet agreement and score arithmetic before atomic
publication. No generation or emission is rewritten and no mismatch is ignored.

The new control manifest/receipt explicitly identifies
`native-control-judge-replay-v1`, the exact generation run digest and original
union hashes. It reports **zero** new evaluator calls and paid misses, rather
than claiming that replay invoked a judge. Nonzero points retain the ordinary
final evaluator and existing job cache. All points are preflighted before any
judging. Missing or changed control evidence fails closed.

All 13 source hashes in each original `implementation_sha256` still match this
checkout. No generation-fingerprinted source was edited, including
`native_methods.py`, `native_provider.py` and `final_judge.py`; original
`identity.json` and point identity hashes remain valid. No relabeling of the
production implementation or broad compatibility exception is needed.

The existing campaign archive verifier imports `_validated_judge_receipt`,
which now admits only the explicitly tagged control replay at `points/p0`,
with exact run/union lineage and zero call accounting. Use this checkout on
`PYTHONPATH` when invoking that verifier. The campaign-owned verifier itself
was read but not edited. Existing legacy manifests remain supported. Resume
also accepts relocated completed receipts exactly as the archive reader does:
only the path prefix may change; coordinate suffix, hashes, counts and lineage
must still agree. The old receipt is preserved, not rewritten.

Changed files:

- `steering/eval/axbench_adv/native_control_judge.py`: authenticated control projection.
- `steering/eval/axbench_adv/native_method_judge.py`: control routing, lineage reader, relocation and all-point preflight.
- `configs/eval/axbench_adv/native_method_final_judge.yaml`: `final_root: null` (required when replaying an unjudged control).
- `docs/evaluation/axbench-adv.md`: recovery contract and invocation.
- `tests/axbench_adv/unit/test_native_control_judge.py`: full-grid source fixtures, source/output corruption, provider exclusion, relocation, strict lineage and preflight.
- `tests/axbench_adv/integration/test_native_method_recovery_artifacts.py`: opt-in provider-free actual-artifact/cache reproduction and audit.
- `out/NATIVE_JUDGE_RECOVERY_READY.md`: this handoff.

## Exact provider-free validation commands

Run from this isolated checkout. `.venv` is a symlink to the existing root venv;
no shared environment was installed into or modified.

```bash
export UV_CACHE_DIR=/tmp/native-recovery-uv
export HF_HUB_OFFLINE=1
export HF_XET_CACHE=/tmp/native-recovery-xet
export MPLCONFIGDIR=/tmp/native-recovery-mpl
export NATIVE_ADV_RECOVERY_ARTIFACT_ROOT=/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/out/fp-head-campaign-20260921/method-completion-400m
uv run --no-sync pytest \
  tests/axbench_adv/unit/test_native_control_judge.py \
  tests/axbench_adv/unit/test_native_methods.py \
  tests/axbench_adv/unit/test_final_judge.py \
  tests/axbench_adv/unit/test_native_decoder.py \
  tests/axbench_adv/integration/test_native_method_recovery_artifacts.py -q
uv run --no-sync ty check \
  steering/eval/axbench_adv/native_control_judge.py \
  steering/eval/axbench_adv/native_method_judge.py \
  tests/axbench_adv/unit/test_native_control_judge.py \
  tests/axbench_adv/integration/test_native_method_recovery_artifacts.py
uv run --no-sync ruff check \
  steering/eval/axbench_adv/native_control_judge.py \
  steering/eval/axbench_adv/native_method_judge.py \
  tests/axbench_adv/unit/test_native_control_judge.py \
  tests/axbench_adv/integration/test_native_method_recovery_artifacts.py
git diff --check
```

Result: **86 passed**, focused ty and Ruff clean, diff whitespace check clean.
Warnings are dependency deprecations/NVML initialization warnings; no GPU work
was launched. The optional actual-artifact test disables socket connections and
replaces the unavailable root-environment OpenAI SDK import with an empty
module, while executing the real vendor cache, prompt templates, rating parser,
language-model cache-only path and evaluator. It does not synthesize responses.

## Safe root recovery commands (prepared, not run)

Create private local copies. This does not copy credentials or top-level logs;
the output tree may still contain private GEPA diagnostics. Never publish it
without the campaign's normal private-artifact handling.

```bash
umask 077
uv run --no-sync python - <<'PY'
import shutil
import sqlite3
from pathlib import Path

campaign = Path('/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/out/fp-head-campaign-20260921/method-completion-400m')
recovery = Path.cwd() / 'out' / 'root-final-judge-recovery'
recovery.mkdir(parents=True, exist_ok=False)
for name in ('decoder-c000-001', 'decoder-c001-002'):
    source = campaign / 'decoder-adv-jobs' / name
    destination = recovery / name
    shutil.copytree(source / 'output', destination / 'output')
    with sqlite3.connect(f'{(source / "judge-cache.sqlite").as_uri()}?mode=ro', uri=True) as original:
        with sqlite3.connect(destination / 'judge-cache.sqlite') as snapshot:
            original.backup(snapshot)
PY

for job in decoder-c000-001 decoder-c001-002; do
  uv run --no-sync python -m scripts.eval.axbench_adv.judge_native_method_points \
    run_dir="$PWD/out/root-final-judge-recovery/$job/output" \
    judge_cache="$PWD/out/root-final-judge-recovery/$job/judge-cache.sqlite" \
    final_root=/home/d0rb/Documents/Github/nano-gpt-pretrain-steer/out/fp-head-campaign-20260921/method-completion-400m/decoder100-final \
    cache_only=true \
    > "out/root-final-judge-recovery/$job/final-judge-private.log" 2>&1
done
```

The module above is the final-judge-only entry point. It never launches the
native generation runner or mailbox provider. The cache-only command should
publish the recovered control, then fail closed at the first unavailable
instruction judgment. To finish after root review and separate provider
approval, run that same invocation with `cache_only=false` in the authorized
provider environment, using the same copied run/cache. Do not run the original
job lifecycle wrapper: it invokes generation. Do not alter the 16 remote jobs'
source trees, identities, caches, outputs, or process lifecycles.

## Remaining work and limits

**Both failed jobs can be recovered without regeneration.** They cannot finish
all final instruction judgments purely from the supplied caches:

| Job | Nonzero instruction positions already cached | Missing instruction positions | Distinct missing exact requests |
| --- | ---: | ---: | ---: |
| decoder-c000-001 | 12/360 | 348 | 278 |
| decoder-c001-002 | 5/360 | 355 | 264 |

None of those distinct missing requests matched an already-paid instruction
judgment elsewhere in the supplied baseline union for these concepts. All
original results remain intact, including repeated control results in the job
cache; authoritative control replay chooses the original baseline judgments.
Distinct request counts are not a billing estimate: retries and concurrent
identical misses can change physical provider request counts.

No live final judgment, full 26-point completion/aggregation, remote recovery,
archive publication, GPU qualification, or full repository test suite was run.
The vendor CLI subprocess environment is not installed in this checkout;
provider-free validation exercised its real evaluator/cache modules in-process.
Root must provide the usual vendored environment for the final CLI invocation.
Local successful zero-control copies are under
`out/recovery-check/decoder-c000-001/output/points/p0/judged` and
`out/recovery-check/decoder-c001-002/output/points/p0/judged`.
