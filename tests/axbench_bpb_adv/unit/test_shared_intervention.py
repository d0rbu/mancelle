"""Golden-fingerprint byte-identity guard for the shared steered-intervention
builder (Task 2.1).

The additive counter-steering logic used to be duplicated in
``scoring.ModelBackedDualCompletionScorer._steered_intervention`` and
``general_val._build_combo_intervention``. Task 2.1 extracts it into
``intervention_build`` so a forthcoming completion-generation feature provably
uses the EXACT same steering the eval scores.

This is the byte-identity gate. We fingerprint the ``Intervention`` objects
produced by BOTH call sites on the real ``mini_axbench_base_run`` fixture (real
diffmean / linear_probe direction files on disk) across families, modes and
additive magnitudes, and pin the fingerprints as module constants. The constants
were captured against the CURRENT (pre-refactor) code; after the rewire the SAME
fingerprints must still be produced -> byte-identical scoring + general_val.

A third anchor asserts that ``build_steered_intervention`` (the new convenience
entry point generation will call) yields the SAME fingerprint as
``scoring._steered_intervention`` for the same row identity -- the generation
correctness anchor.
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch as t

from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb.bpb_scoring import _coefficient_from_str
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.general_val import (
    _build_combo_intervention,
    _InterventionCombo,
)
from steering.eval.axbench_bpb_adv.intervention_build import build_steered_intervention
from steering.eval.axbench_bpb_adv.scoring import ModelBackedDualCompletionScorer

_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "axbench_bpb"
    / "fixtures"
    / "mini_axbench_base_run"
)

# diffmean_and_probe layer stamped on every direction in the mini fixture.
_DIFFMEAN_LAYER = 1
_COUNTER_FAMILY = "diffmean"

# Representative coverage: residual-direction families (diffmean / linear_probe)
# carry a real direction tensor AND a counter add-on; the control family is the
# decoder-only no-steering baseline (kind NONE). Both modes; additive magnitude
# 0.0 (control twin -> no counter) and 0.4 (non-zero -> counter attached).
_METHOD_BY_FAMILY_MODE: dict[tuple[str, str], str] = {
    ("diffmean", "positive"): "diffmean_decoder_only",
    ("diffmean", "negative"): "neg_diffmean_decoder_only",
    ("linear_probe", "positive"): "linear_probe_decoder_only",
    ("linear_probe", "negative"): "neg_linear_probe_decoder_only",
    ("control", "positive"): "control_decoder_only",
    ("control", "negative"): "neg_control_decoder_only",
}
_CONCEPT_ID = 1
_COEFFICIENT_STR = "4.0"
_CORE_PREFIX_ID = "core_anchor"
_TARGET_MODEL = "decoder_only"

Fingerprint = tuple[Any, ...]


def _tensor_fp(tensor: t.Tensor | None) -> tuple[Any, ...] | None:
    if tensor is None:
        return None
    # Shape + absolute-sum: a stable, order-insensitive content digest that does
    # not depend on tensor identity (the refactor reloads from the same file).
    return (tuple(tensor.shape), float(tensor.abs().sum()))


def _fingerprint(iv: Intervention) -> Fingerprint:
    return (
        str(iv.kind),
        iv.target_model,
        iv.direction_layer,
        iv.direction_coefficient,
        _tensor_fp(iv.direction),
        iv.counter_direction_layer,
        iv.counter_direction_coefficient,
        _tensor_fp(iv.counter_direction),
    )


def _concept() -> Any:
    # The intervention builders only read ``concept.concept_id`` (to locate the
    # per-concept direction file on disk); the rest of the Concept schema is
    # irrelevant to the fingerprint, so a minimal stand-in suffices and matches
    # the sibling scoring tests.
    return SimpleNamespace(concept_id=_CONCEPT_ID)


def _source_paths() -> SourcePaths:
    return SourcePaths(source_run_dir=_FIXTURE)


def _scorer() -> ModelBackedDualCompletionScorer:
    return ModelBackedDualCompletionScorer(
        # Legacy model stand-in for the scorer's checkpoint-V verify (issue #695
        # PR-1 follow-up). None of the families fingerprinted here are bag-backed
        # (diffmean/linear_probe/control), so the model is threaded but never
        # dereferenced for V -- only its presence (a real .model attribute) matters.
        target_bundles=cast(
            Any, {_TARGET_MODEL: SimpleNamespace(model=SimpleNamespace(feature_vocab=None))}
        ),
        target_model_metas={_TARGET_MODEL: cast(Any, SimpleNamespace())},
        concepts_by_id={_CONCEPT_ID: _concept()},
        source_paths=_source_paths(),
        tokenizer=None,
        device=t.device("cpu"),
        diffmean_layer=_DIFFMEAN_LAYER,
        inactive_sampler=None,
        apply_max_act_scale=False,
        counter_direction_family=_COUNTER_FAMILY,
    )


def _candidate_row(*, method_name: str, mode: str, magnitude: float) -> dict[str, Any]:
    return {
        "method_name": method_name,
        "concept_id": _CONCEPT_ID,
        "mode": mode,
        "core_prefix_id": _CORE_PREFIX_ID,
        "coefficient_str": _COEFFICIENT_STR,
        "target_model": _TARGET_MODEL,
        "adversarial_method": "additive_direction",
        "adversarial_strength": magnitude,
    }


def _combo(*, method_name: str, mode: str, magnitude: float) -> _InterventionCombo:
    return _InterventionCombo(
        target_model=_TARGET_MODEL,
        method_name=method_name,
        mode=mode,
        concept_id=_CONCEPT_ID,
        coefficient_str=_COEFFICIENT_STR,
        adversarial_method="additive_direction",
        adversarial_strength=magnitude,
        # The steered intervention does not depend on this (it forks the
        # faster_gcg doc attack, not the additive_direction intervention here);
        # a fixed value keeps the fingerprint test's combo well-formed.
        strength_index=0,
    )


def _scoring_intervention(*, method_name: str, mode: str, magnitude: float) -> Intervention:
    scorer = _scorer()
    method = get_method_by_name(method_name)
    return scorer._steered_intervention(
        method=method,
        concept=_concept(),
        coefficient=_coefficient_from_str(_COEFFICIENT_STR),
        candidate_row=_candidate_row(
            method_name=method_name, mode=mode, magnitude=magnitude
        ),
    )


def _general_val_intervention(
    *, method_name: str, mode: str, magnitude: float
) -> Intervention:
    method = get_method_by_name(method_name)
    return _build_combo_intervention(
        combo=_combo(method_name=method_name, mode=mode, magnitude=magnitude),
        method=method,
        concept=_concept(),
        source_paths=_source_paths(),
        diffmean_layer=_DIFFMEAN_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=None,
        counter_direction_family=_COUNTER_FAMILY,
        apply_max_act_scale=False,
    )


def _generation_intervention(
    *, method_name: str, mode: str, magnitude: float
) -> Intervention:
    """Build the intervention the way completion generation will: ONE call to
    ``build_steered_intervention`` (base + counter together)."""

    method = get_method_by_name(method_name)
    rng = random.Random(
        hashlib.sha256(
            (
                f"{method_name}|{_CONCEPT_ID}|{_CORE_PREFIX_ID}|{_COEFFICIENT_STR}"
            ).encode()
        ).digest()
    )
    return build_steered_intervention(
        method=method,
        concept=_concept(),
        coefficient=_coefficient_from_str(_COEFFICIENT_STR),
        source_paths=_source_paths(),
        diffmean_layer=_DIFFMEAN_LAYER,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=None,
        rng=rng,
        adversarial_method="additive_direction",
        adversarial_strength=magnitude,
        counter_direction_family=_COUNTER_FAMILY,
        target=_TARGET_MODEL,
        apply_max_act_scale=False,
    )


# --- GOLDEN FINGERPRINTS (captured against pre-refactor code) ------------------
# scoring._steered_intervention. Keyed by (family, mode, magnitude).
_SCORING_GOLDEN: dict[tuple[str, str, float], Fingerprint] = {
    ("control", "negative", 0.0): (
        "none", "decoder_only", None, None, None, None, None, None,
    ),
    ("control", "negative", 0.4): (
        "none", "decoder_only", None, None, None,
        1, 0.4, ((32,), 22.956300735473633),
    ),
    ("control", "positive", 0.0): (
        "none", "decoder_only", None, None, None, None, None, None,
    ),
    ("control", "positive", 0.4): (
        "none", "decoder_only", None, None, None,
        1, -0.4, ((32,), 23.355911254882812),
    ),
    ("diffmean", "negative", 0.0): (
        "diffmean", "decoder_only", 1, -4.0,
        ((32,), 22.956300735473633), None, None, None,
    ),
    ("diffmean", "negative", 0.4): (
        "diffmean", "decoder_only", 1, -4.0,
        ((32,), 22.956300735473633), 1, 0.4, ((32,), 22.956300735473633),
    ),
    ("diffmean", "positive", 0.0): (
        "diffmean", "decoder_only", 1, 4.0,
        ((32,), 23.355911254882812), None, None, None,
    ),
    ("diffmean", "positive", 0.4): (
        "diffmean", "decoder_only", 1, 4.0,
        ((32,), 23.355911254882812), 1, -0.4, ((32,), 23.355911254882812),
    ),
    ("linear_probe", "negative", 0.0): (
        "linear_probe", "decoder_only", 1, -4.0,
        ((32,), 23.90316390991211), None, None, None,
    ),
    ("linear_probe", "negative", 0.4): (
        "linear_probe", "decoder_only", 1, -4.0,
        ((32,), 23.90316390991211), 1, 0.4, ((32,), 22.956300735473633),
    ),
    ("linear_probe", "positive", 0.0): (
        "linear_probe", "decoder_only", 1, 4.0,
        ((32,), 24.190061569213867), None, None, None,
    ),
    ("linear_probe", "positive", 0.4): (
        "linear_probe", "decoder_only", 1, 4.0,
        ((32,), 24.190061569213867), 1, -0.4, ((32,), 23.355911254882812),
    ),
}


def _cases() -> list[tuple[str, str, float]]:
    cases: list[tuple[str, str, float]] = []
    for (family, mode) in sorted(_METHOD_BY_FAMILY_MODE):
        for magnitude in (0.0, 0.4):
            cases.append((family, mode, magnitude))
    return cases


def test_scoring_steered_intervention_matches_golden_fingerprints() -> None:
    """``scoring._steered_intervention`` is byte-identical to the captured golden
    across families/modes/magnitudes on the real mini fixture."""
    for family, mode, magnitude in _cases():
        method_name = _METHOD_BY_FAMILY_MODE[(family, mode)]
        iv = _scoring_intervention(
            method_name=method_name, mode=mode, magnitude=magnitude
        )
        assert _fingerprint(iv) == _SCORING_GOLDEN[(family, mode, magnitude)], (
            f"scoring fingerprint drift for {family}/{mode}/mag={magnitude}"
        )


def test_general_val_combo_intervention_matches_scoring_fingerprints() -> None:
    """``general_val._build_combo_intervention`` produces the SAME fingerprint as
    ``scoring._steered_intervention`` for the same identity (both share the golden).

    The base RNG seed differs between the two (``general_val|...`` vs the scoring
    ``method|concept|core_prefix|coef`` seed), but for the families covered here
    the direction-load is RNG-independent, so the fingerprints coincide. The
    counter add-on is identical.
    """
    for family, mode, magnitude in _cases():
        method_name = _METHOD_BY_FAMILY_MODE[(family, mode)]
        iv = _general_val_intervention(
            method_name=method_name, mode=mode, magnitude=magnitude
        )
        assert _fingerprint(iv) == _SCORING_GOLDEN[(family, mode, magnitude)], (
            f"general_val fingerprint drift for {family}/{mode}/mag={magnitude}"
        )


def test_scoring_and_general_val_agree_pairwise() -> None:
    """Direct cross-check: the two call sites agree fingerprint-for-fingerprint
    (independent of the pinned constants)."""
    for family, mode, magnitude in _cases():
        method_name = _METHOD_BY_FAMILY_MODE[(family, mode)]
        scoring_fp = _fingerprint(
            _scoring_intervention(method_name=method_name, mode=mode, magnitude=magnitude)
        )
        general_fp = _fingerprint(
            _general_val_intervention(
                method_name=method_name, mode=mode, magnitude=magnitude
            )
        )
        assert scoring_fp == general_fp, (
            f"scoring vs general_val disagree for {family}/{mode}/mag={magnitude}"
        )


def test_generation_builder_matches_scoring_fingerprints() -> None:
    """GENERATION CORRECTNESS ANCHOR: ``build_steered_intervention`` (the one-call
    entry point completion generation will use) yields the SAME fingerprint as
    ``scoring._steered_intervention`` for the same row identity.

    This is what makes the forthcoming generation feature provably steer the way
    the eval scores: both go through the shared ``apply_additive_counter``, and
    the one-call builder reproduces the scorer's base + counter exactly.
    """
    for family, mode, magnitude in _cases():
        method_name = _METHOD_BY_FAMILY_MODE[(family, mode)]
        scoring_fp = _fingerprint(
            _scoring_intervention(method_name=method_name, mode=mode, magnitude=magnitude)
        )
        generation_fp = _fingerprint(
            _generation_intervention(
                method_name=method_name, mode=mode, magnitude=magnitude
            )
        )
        assert generation_fp == scoring_fp, (
            f"build_steered_intervention diverges from scoring for "
            f"{family}/{mode}/mag={magnitude}"
        )
        # And it equals the pinned golden, so generation is byte-identity-pinned too.
        assert generation_fp == _SCORING_GOLDEN[(family, mode, magnitude)]
