"""SAE feature-equivalence kernels and the production compile-gate threshold.

Test-only acceptance thresholds and measurement oracles live under
``tests/steering/native/``.

Only ``torch`` is needed — no pytest or test-only dependencies.
"""

from __future__ import annotations

import torch as t

# --- Ideal-math-relative compile-gate SLACK constant ---------------------- #
# The PRODUCTION compile gate measures the candidate compiled config against
# *fp32-eager* (the "ideal math" reference; model + SAE in fp32, eager
# forward) and PASSES iff the compiled config is no worse than the bf16-eager
# baseline relative to that ideal — within this SLACK band.
#
# EMPIRICAL CALIBRATION (FineWeb-Edu shard 00388, gemma-3-1b-pt @7 bf16,
# jaccard vs fp32-eager reference, 4090, 2026-05-20):
#   bf16-eager (baseline)                : 0.988895    ← reference baseline
#   bf16-compiled emT dyn (prior shipped): 0.988893
#   bf16-compiled emT static             : 0.989086
#   bf16-compiled emF dyn (NEW default)  : 0.990183   ← CLOSER to ideal than eager
#   bf16-compiled emF static (best)      : 0.990657
#   bf16-compiled emT + graph-break (S3) : 0.988903
#   default-Inductor (teeth, must fail)  : ~0.970     (~0.019 worse than baseline)
#
# COMPILED-PATH RUN-TO-RUN VARIANCE (5x fresh-process slow-gate reps of
# gemma-3-1b-pt @7 bf16 on the realistic corpus, 2026-05-20): the bf16-eager
# baseline-vs-ref is PERFECTLY STABLE (0.987908 every rep — eager is
# deterministic), but the COMPILED-vs-ref is BIMODAL across reps —
# {0.985472 (1/5), 0.989117 (4/5)} — because Inductor autotune/bf16
# non-determinism selects one of two kernel variants per fresh compile. The
# WORST observed mode (0.985472) is 0.0024 WORSE than the eager baseline; the
# best is 0.0012 BETTER. So the compiled path is, at worst, ~0.0024 worse than
# eager vs ideal-math — i.e. essentially equivalent to eager (well within the
# bf16 regime), just non-deterministic by ~0.0036 across compiles.
#
# SLACK=0.010 is calibrated to that EMPIRICAL variance, NOT the smaller
# single-measurement noise (the earlier 0.005 was ~1.25x the observed 0.0036
# bimodal swing — too tight; a fresh compile landing on the worse mode left
# only a +0.0026 margin, and a 5-sample run cannot rule out a third worse
# mode → flaky startup hard-abort risk). 0.010 is ~4x the worst observed
# degradation (0.0024) and ~3x the bimodal swing (0.0036), calibrated to:
#   (a) admit the borderline-but-no-worse-than-eager gemma-3-bf16 cell on
#       EVERY compile mode (worst-mode margin +0.0076 at slack=0.010), so the
#       gate never flaky-aborts a correct config at startup;
#   (b) decisively reject the default-Inductor teeth (~0.019 worse than
#       baseline) AND any real corruption (the fault-injection teeth is ~0.355
#       worse) — both FAR outside slack=0.010.
#
# MUTATION HINT (do not change blindly):
#   SLACK<=0.004 would (on the worse compile mode) spuriously hard-abort the
#               correct gemma-3-bf16 cell at startup (the 0.0024-worse mode +
#               a fresh-compile dip → negative margin → CompileGateError).
#   SLACK>=0.019 would admit the known-corrupting default-Inductor teeth case
#               (silent corruption — gate becomes vacuous).
BASELINE_RELATIVE_SLACK: float = 0.010

def _seg_jaccard_and_count_rdiff(aid: t.Tensor, bid: t.Tensor) -> tuple[float, float]:
    """The ONE per-segment structural kernel: feature-set jaccard and
    feature-count relative difference for a single (aid, bid) pair.

    This is the literal single source of truth for the structural arithmetic:
    BOTH ``assert_feature_equivalent`` (asserts each result against its gates)
    and ``feature_equivalence_metrics`` (aggregates min-jaccard / max-rdiff)
    call it, so the two can never drift.  Behaviour is byte-identical to the
    formulas they previously each inlined: ``count_rdiff`` uses
    ``denom = max(n_a, n_b, 1)``; ``jaccard`` is ``|∩| / |∪|`` over the
    ``.tolist()`` sets, defined as ``1.0`` when the union is empty.
    """
    n_a, n_b = len(aid), len(bid)
    count_rdiff = abs(n_a - n_b) / max(n_a, n_b, 1)
    set_a, set_b = set(aid.tolist()), set(bid.tolist())
    union = len(set_a | set_b)
    jaccard = (len(set_a & set_b) / union) if union > 0 else 1.0
    return jaccard, count_rdiff


def feature_equivalence_metrics(
    a_ids: list[t.Tensor],
    b_ids: list[t.Tensor],
) -> tuple[float, float]:
    """Observed structural envelope for an (ids) pair: the WORST per-segment
    jaccard and the WORST per-segment feature-count rel-diff.

    Returns ``(min_jaccard, max_count_rdiff)`` by aggregating the per-segment
    results of ``_seg_jaccard_and_count_rdiff`` — the EXACT same kernel
    ``assert_feature_equivalent`` asserts against.  The shared kernel is the
    structural single source of truth: this measure-only sibling and the assert
    physically cannot drift because they call the same function (it asserts
    nothing).  The slow native-vs-TL gate logs these BEFORE asserting so the
    chosen data-grounded ``jaccard_threshold`` is grounded by a printed
    measurement EVERY run, not a one-time claim in a comment.  ``min_jaccard``
    over zero segments is 1.0 / ``max_count_rdiff`` is 0.0 (vacuously perfect —
    the separate non-vacuity guard rejects an actually-empty feature set).
    """
    assert len(a_ids) == len(b_ids), (
        f"feature_equivalence_metrics: segment count mismatch "
        f"({len(a_ids)}) vs ({len(b_ids)})"
    )
    min_jaccard = 1.0
    max_count_rdiff = 0.0
    for aid, bid in zip(a_ids, b_ids, strict=True):
        jaccard, count_rdiff = _seg_jaccard_and_count_rdiff(aid, bid)
        min_jaccard = min(min_jaccard, jaccard)
        max_count_rdiff = max(max_count_rdiff, count_rdiff)
    return min_jaccard, max_count_rdiff


def compiled_passes_ideal_math_gate(
    *,
    ref_ids: list[t.Tensor],
    baseline_ids: list[t.Tensor],
    compiled_ids: list[t.Tensor],
    slack: float = BASELINE_RELATIVE_SLACK,
) -> tuple[bool, dict[str, float]]:
    """Ideal-math-relative compile-gate predicate.

    Measures the compiled candidate config against the bf16-eager production
    baseline, BOTH evaluated jaccard-wise against the fp32-eager (ideal-math)
    reference.  PASSES iff the compiled config is no worse than the baseline
    minus ``slack``.

    Decision rule (all jaccards are min-jaccard over the per-segment feature
    sets, via ``feature_equivalence_metrics``, the same single-source kernel
    every other oracle here uses):

        j_baseline = jaccard(ref, baseline)
        j_compiled = jaccard(ref, compiled)
        PASS  iff  j_compiled >= j_baseline - slack
        margin    = j_compiled - (j_baseline - slack)   (>= 0 ⇒ PASS)

    Returns ``(passed, measurements)`` where ``measurements`` is a dict with
    the four documented keys (``j_baseline_vs_ref``, ``j_compiled_vs_ref``,
    ``slack``, ``margin``) that the gate's PASS info-log and the
    ``CompileGateError`` message both surface verbatim.

    Why this is the production GATE predicate (NOT the conftest-parity
    oracle): the conftest's ``feature_equivalence_passed`` is an
    *absolute* threshold check (jaccard >= 0.99 etc.), still used by the
    slow Layer-4 native-vs-TL assertion (don't touch that contract). The
    compile gate is a *relative* check anchored on ideal-math — empirically
    (4090, FineWeb-Edu shard 00388) the production-shipped emT-dyn compiled
    config diverged from fp32-eager by LESS than bf16-eager itself, while the
    prior absolute-threshold gate occasionally rejected it on a worse-than-
    eager bf16-vs-bf16 noise blip.  Anchoring the gate on ideal-math + a
    SLACK band cleanly separates "compiled is at least as faithful as bf16
    itself" (PASS) from "compiled is materially worse than bf16" (FAIL,
    e.g. the known-corrupting default-Inductor teeth case at ~0.019 worse).
    """
    j_baseline_vs_ref, _ = feature_equivalence_metrics(ref_ids, baseline_ids)
    j_compiled_vs_ref, _ = feature_equivalence_metrics(ref_ids, compiled_ids)
    margin = j_compiled_vs_ref - (j_baseline_vs_ref - slack)
    passed = j_compiled_vs_ref >= (j_baseline_vs_ref - slack)
    measurements: dict[str, float] = {
        "j_baseline_vs_ref": j_baseline_vs_ref,
        "j_compiled_vs_ref": j_compiled_vs_ref,
        "slack": slack,
        "margin": margin,
    }
    return passed, measurements
