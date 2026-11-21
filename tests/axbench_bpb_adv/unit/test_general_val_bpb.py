"""Unit tests for the general_val_bpb feature (the `score_general_val` stage).

Covers the pure logic that does NOT require loading models / GPU:
* config: ``GeneralValBpbConfig`` defaults + ``validate_config`` rules.
* intervention-signature dedup: faster_gcg is now DISTINCT per k under the
  universal-trigger attack (was all ``base``); additive>0 distinct;
  additive-magnitude-0 and gepa stay ``base``.
* ``max_interventions`` cap selection.
* the aggregation left-join + macro concept-averaging of ``general_val_bpb``
  (the join keys on the intervention identity so per-k values land).
* the ``general_val_bpb`` scatter x-axis branch.
* ``build_general_val_docs`` BOS-additive slicing with a fake tokenizer (the
  prefix is BOS-prefixed; the scored OUTPUT continuation is unchanged).
* the pure doc-attack helpers (``_attack_doc`` last-k substitution, the
  universal-trigger lookup) that make ``general_val_bpb`` non-flat across the
  GCG axis (k=0 == clean baseline; a missing trigger raises).

The model-backed scoring (``run_general_val_bpb_stage``) is exercised by the
functional CLI tests, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb_adv import general_val, general_val_roofline
from steering.eval.axbench_bpb_adv.aggregation import (
    _join_general_val_bpb,
    _scatter_baseline_x,
    compute_macro_aggregation,
)
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    RawClusteredFeatureSegment,
    RawClusteredPrefix,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    ScatterBaselineX,
    validate_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.general_val import (
    _attack_doc,
    _byte_weighted_bpb,
    _corpus_signature,
    _distinct_combos,
    _GeneralValDoc,
    _intervention_signature,
    _InterventionCombo,
    _is_attacked_decoder_combo,
    _lookup_universal_trigger,
    _roofline_bag_config_metadata,
    _RooflineGeneralValState,
    _select_scored_signatures,
    build_general_val_docs,
)
from steering.eval.axbench_bpb_adv.general_val_roofline import GeneralValSignalBudget
from steering.eval.axbench_bpb_adv.schemas import (
    GENERAL_VAL_BPB_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.statistics_doubles import fixed_statistics_double


# --------------------------------------------------------------------------- #
# config                                                                       #
# --------------------------------------------------------------------------- #
def _valid_config(**overrides: object) -> AxBenchBpbAdvConfig:
    base = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


def test_general_val_bpb_enabled_by_default() -> None:
    cfg = AxBenchBpbAdvConfig()
    assert isinstance(cfg.general_val_bpb, GeneralValBpbConfig)
    assert cfg.general_val_bpb.enabled is True


def test_general_val_bpb_can_be_disabled_when_scatter_baseline_x_moves_off_it() -> None:
    validate_config(
        _valid_config(
            base_methods=["diffmean_encdec"],
            general_val_bpb=GeneralValBpbConfig(enabled=False),
            scatter_baseline_x="baseline_loss",
        )
    )


def test_general_val_bpb_valid_when_enabled() -> None:
    validate_config(
        _valid_config(
            base_methods=["diffmean_encdec"],
            general_val_bpb=GeneralValBpbConfig(
                enabled=True, n_documents=4, prefix_tokens=32, output_tokens=16
            ),
        )
    )


@pytest.mark.parametrize("field", ["n_documents", "prefix_tokens", "output_tokens"])
def test_general_val_bpb_rejects_non_positive_when_enabled(field: str) -> None:
    cfg = GeneralValBpbConfig(enabled=True, **{field: 0})
    with pytest.raises(AxBenchBpbAdvError, match=f"general_val_bpb.{field}"):
        validate_config(_valid_config(general_val_bpb=cfg))


def test_general_val_bpb_non_positive_ignored_when_disabled() -> None:
    # When disabled the n_documents/prefix/output knobs are inert. Disabling
    # general_val_bpb also requires moving scatter_baseline_x off its default
    # (general_val_bpb) -- see test_scatter_general_val_requires_enabled.
    validate_config(
        _valid_config(
            general_val_bpb=GeneralValBpbConfig(enabled=False, n_documents=0),
            scatter_baseline_x="baseline_loss",
        )
    )


def test_general_val_bpb_rejects_negative_max_interventions() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="max_interventions"):
        GeneralValBpbConfig(max_interventions=-1)


def test_scatter_general_val_requires_enabled() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="general_val_bpb.enabled"):
        validate_config(
            _valid_config(
                scatter_baseline_x="general_val_bpb",
                general_val_bpb=GeneralValBpbConfig(enabled=False),
            )
        )
    # OK once enabled.
    validate_config(
        _valid_config(
            base_methods=["diffmean_encdec"],
            scatter_baseline_x="general_val_bpb",
            general_val_bpb=GeneralValBpbConfig(enabled=True),
        )
    )


def test_scatter_baseline_x_enum_has_general_val_bpb() -> None:
    assert ScatterBaselineX.coerce("general_val_bpb") is ScatterBaselineX.GENERAL_VAL_BPB


def test_strength_family_plot_requires_general_val_enabled() -> None:
    # issue #635 review [1]: the frontier x-axis is general_val_bpb, so enabling
    # the strength-family plots without general_val would crash the plot stage at
    # the very end of a run. validate_config rejects it FAST.
    with pytest.raises(AxBenchBpbAdvError, match="plot_steering_strength_families"):
        validate_config(
            _valid_config(
                plot_steering_strength_families=True,
                general_val_bpb=GeneralValBpbConfig(enabled=False),
                # Off general_val_bpb so the earlier scatter_baseline_x check
                # (also gated on general_val_bpb.enabled) doesn't fire first.
                scatter_baseline_x="baseline_loss",
            )
        )
    # OK once general_val is enabled.
    validate_config(
        _valid_config(
            base_methods=["diffmean_encdec"],
            plot_steering_strength_families=True,
            general_val_bpb=GeneralValBpbConfig(enabled=True),
        )
    )


def test_validate_config_rejects_prefix_shorter_than_largest_k() -> None:
    # The universal trigger is substituted into the last-k tokens of each
    # BOS-prefixed decoder-only validation doc, which needs >= k non-BOS prefix
    # tokens. validate_config must fail FAST when prefix_tokens < max(k) instead
    # of crashing deep inside the general_val stage.
    with pytest.raises(AxBenchBpbAdvError, match="prefix_tokens"):
        validate_config(
            _valid_config(
                base_methods=["diffmean_decoder_only"],
                target_models=["decoder_only"],
                adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0, 1, 32))],
                general_val_bpb=GeneralValBpbConfig(enabled=True, prefix_tokens=16),
            )
        )
    # OK once prefix_tokens covers the largest k.
    validate_config(
        _valid_config(
            base_methods=["diffmean_decoder_only"],
            target_models=["decoder_only"],
            adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0, 1, 32))],
            general_val_bpb=GeneralValBpbConfig(enabled=True, prefix_tokens=32),
        )
    )


def test_validate_config_prefix_guard_inert_without_decoder_only() -> None:
    # encdec faster_gcg is deferred (clean docs, no substitution), so a short
    # prefix is harmless when decoder_only is not a target -> no guard fires.
    validate_config(
        _valid_config(
            base_methods=["diffmean_encdec"],
            target_models=["encdec"],
            adversarial_methods=[FasterGcgConfig(optimized_token_counts=(0, 1, 32))],
            general_val_bpb=GeneralValBpbConfig(enabled=True, prefix_tokens=16),
        )
    )


# --------------------------------------------------------------------------- #
# intervention signature dedup                                                 #
# --------------------------------------------------------------------------- #
def _combo(
    adversarial_method: str,
    strength: float,
    *,
    strength_index: int = 0,
    target_model: str = "decoder_only",
) -> _InterventionCombo:
    return _InterventionCombo(
        target_model=target_model,
        method_name="diffmean_decoder_only",
        mode="positive",
        concept_id=7,
        coefficient_str="1.0",
        adversarial_method=adversarial_method,
        adversarial_strength=strength,
        strength_index=strength_index,
    )


def test_signature_collapses_gcg_and_additive_zero_to_base() -> None:
    # Under the UNIVERSAL trigger attack, faster_gcg triggers differ per k -- so
    # the signature now VARIES with strength_index (it is NOT collapsed to
    # ``base`` any more). Inverting the old collapse assertion is the C1-existing
    # fix.
    k0 = _intervention_signature(_combo("faster_gcg", 0.0, strength_index=0))
    assert _intervention_signature(_combo("faster_gcg", 8.0, strength_index=1)) != k0
    assert _intervention_signature(_combo("faster_gcg", 32.0, strength_index=2)) != k0
    # The additive-magnitude-0 control is still the pure baseline -> ``base``,
    # and the k=0 faster_gcg control (no trigger substituted) is ALSO base
    # (it scores the clean BOS-prefixed docs).
    additive_zero = _intervention_signature(_combo("additive_direction", 0.0))
    assert additive_zero.endswith("|base")
    assert k0.endswith("|base")


def test_signature_collapse_mutation_teeth() -> None:
    # Mutation teeth: if a refactor collapsed faster_gcg back to ``base`` (the
    # old behavior), two distinct k combos would share a signature and
    # general_val_bpb would broadcast the k=0 value to every k. Pin that they
    # DIFFER so that regression fails loudly here.
    a = _intervention_signature(_combo("faster_gcg", 8.0, strength_index=1))
    b = _intervention_signature(_combo("faster_gcg", 16.0, strength_index=2))
    assert a != b
    # And each carries the faster_gcg adv part (not the literal ``base``).
    assert not a.endswith("|base")
    assert not b.endswith("|base")


def test_signature_gepa_stays_base() -> None:
    # GEPA has no universal trigger (it is not a universal-GCG member), so its
    # signature stays ``base`` regardless of strength_index (m7): the doc attack
    # never substitutes a trigger for gepa.
    a = _intervention_signature(_combo("gepa", 8.0, strength_index=1))
    b = _intervention_signature(_combo("gepa", 32.0, strength_index=3))
    assert a.endswith("|base")
    assert a == b


def test_signature_distinguishes_additive_magnitudes() -> None:
    base = _intervention_signature(_combo("additive_direction", 0.0))
    sig_1 = _intervention_signature(_combo("additive_direction", 1.0))
    sig_2 = _intervention_signature(_combo("additive_direction", 2.0))
    assert sig_1 != base
    assert sig_2 != base
    assert sig_1 != sig_2


def test_signature_additive_unchanged_by_k() -> None:
    # The additive path keys only on its magnitude (adversarial_strength); the
    # strength_index field must NOT alter the additive signature (only faster_gcg
    # forks on it).
    a = _intervention_signature(_combo("additive_direction", 1.0, strength_index=0))
    b = _intervention_signature(_combo("additive_direction", 1.0, strength_index=2))
    assert a == b


def test_signature_distinguishes_method_concept_coefficient() -> None:
    a = _intervention_signature(_combo("faster_gcg", 0.0))
    other = _InterventionCombo(
        target_model="decoder_only",
        method_name="diffmean_decoder_only",
        mode="positive",
        concept_id=8,  # different concept
        coefficient_str="1.0",
        adversarial_method="faster_gcg",
        adversarial_strength=0.0,
        strength_index=0,
    )
    assert _intervention_signature(other) != a


# --------------------------------------------------------------------------- #
# non-roofline signature goldens (D7): pin the CURRENT literal output so a     #
# future refactor of _intervention_signature must consciously update this      #
# test rather than silently reshaping the general_val dedup key.               #
# --------------------------------------------------------------------------- #
def test_non_roofline_signature_strings_are_pinned() -> None:
    combo = _InterventionCombo(
        target_model="decoder_only",
        method_name="diffmean_decoder_only",
        mode="positive",
        concept_id=3,
        coefficient_str="2.0",
        adversarial_method="additive_direction",
        adversarial_strength=0.5,
        strength_index=2,
    )
    assert _intervention_signature(combo) == "decoder_only|diffmean_decoder_only|3|2.0|additive:0.5"
    control = replace(combo, adversarial_strength=0.0, strength_index=0)
    assert _intervention_signature(control) == "decoder_only|diffmean_decoder_only|3|2.0|base"


def _roofline_combo(
    method_name: str = "ours_gcg_roofline_encdec",
    *,
    concept_id: int = 3,
    coefficient_str: str = "0.0",
    adversarial_method: str = "faster_gcg",
    adversarial_strength: float = 0.0,
) -> _InterventionCombo:
    return _InterventionCombo(
        target_model="encdec",
        method_name=method_name,
        mode="positive",
        concept_id=concept_id,
        coefficient_str=coefficient_str,
        adversarial_method=adversarial_method,
        adversarial_strength=adversarial_strength,
        strength_index=0,
    )


def test_roofline_signature_is_concept_free_without_counter() -> None:
    a = _intervention_signature(_roofline_combo(concept_id=3))
    b = _intervention_signature(_roofline_combo(concept_id=9))
    assert a == b
    assert a == "encdec|ours_gcg_roofline_encdec|__roofline_doc__|0.0|base"


def test_roofline_feature_diffmean_signature_is_concept_free() -> None:
    sig = _intervention_signature(
        _roofline_combo(
            "feature_diffmean_roofline_encdec",
            coefficient_str="None",
            concept_id=42,
        )
    )
    assert sig == "encdec|feature_diffmean_roofline_encdec|__roofline_doc__|None|base"


def test_roofline_additive_counter_signature_keeps_concept() -> None:
    a = _intervention_signature(
        _roofline_combo(
            concept_id=3,
            adversarial_method="additive_direction",
            adversarial_strength=1.0,
        )
    )
    b = _intervention_signature(
        _roofline_combo(
            concept_id=9,
            adversarial_method="additive_direction",
            adversarial_strength=1.0,
        )
    )
    assert a != b
    assert a == "encdec|ours_gcg_roofline_encdec|3|0.0|additive:1.0"
    assert b == "encdec|ours_gcg_roofline_encdec|9|0.0|additive:1.0"


# --------------------------------------------------------------------------- #
# attacked-combo gating keys on k (adversarial_strength), NOT the grid ordinal #
# (regression guards for the strength_index-vs-k decoupling bug)               #
# --------------------------------------------------------------------------- #
def test_is_attacked_decoder_combo_gates_on_k_not_strength_index() -> None:
    # `strength_index` is the optimized_token_counts ORDINAL; the real trigger
    # length k == adversarial_strength. They coincide ONLY when the grid leads
    # with 0. The attack gate must key on k, so a non-leading-0 grid is handled.
    # idx=0 but k=1 (grid like [1, 2, 4]) -> a REAL attack.
    assert _is_attacked_decoder_combo(_combo("faster_gcg", 1.0, strength_index=0)) is True
    # idx=1 but k=0 (grid like [2, 0, 4]) -> a CONTROL (no trigger).
    assert _is_attacked_decoder_combo(_combo("faster_gcg", 0.0, strength_index=1)) is False
    # encdec never gets the decoder-only doc attack, whatever k.
    assert (
        _is_attacked_decoder_combo(
            _combo("faster_gcg", 8.0, strength_index=2, target_model="encdec")
        )
        is False
    )


def test_signature_forks_attacked_combos_by_k_not_ordinal() -> None:
    # No-leading-0 grid: idx0/k1 is a real attack -> must NOT collapse to base,
    # and the fork must label by k (adversarial_strength), so k1 and k2 differ.
    sig_k1 = _intervention_signature(_combo("faster_gcg", 1.0, strength_index=0))
    sig_k2 = _intervention_signature(_combo("faster_gcg", 2.0, strength_index=1))
    assert not sig_k1.endswith("|base")
    assert sig_k1 != sig_k2
    # A k=0 control carrying a NONZERO ordinal (grid like [2, 0, 4]) collapses
    # to base -- the OLD strength_index>0 gate would have (wrongly) forked it and
    # then failed the trigger lookup.
    assert _intervention_signature(_combo("faster_gcg", 0.0, strength_index=1)).endswith("|base")


def test_signature_k0_control_collapses_to_base() -> None:
    # The k=0 decoder-only control substitutes NO trigger, so its general_val
    # docs are clean; it must collapse to the base signature rather than forking
    # (which would re-score identical clean docs).
    a0 = _intervention_signature(_combo("faster_gcg", 0.0, strength_index=0))
    assert a0.endswith("|base")


def test_signature_encdec_faster_gcg_collapses_to_base() -> None:
    # encdec faster_gcg is deferred (clean docs, no trigger substituted), so it
    # must NOT fork per k -- every encdec faster_gcg combo shares the base
    # intervention instead of redundantly re-scoring identical clean docs.
    e1 = _intervention_signature(_combo("faster_gcg", 8.0, strength_index=1, target_model="encdec"))
    e2 = _intervention_signature(
        _combo(
            "faster_gcg",
            32.0,
            strength_index=3,
            target_model="encdec",
        )
    )
    assert e1.endswith("|base")
    assert e1 == e2


# --------------------------------------------------------------------------- #
# max_interventions cap                                                        #
# --------------------------------------------------------------------------- #
def test_select_signatures_no_cap() -> None:
    sigs = [f"s{i}" for i in range(5)]
    assert _select_scored_signatures(sigs, max_interventions=0, seed=0) == set(sigs)
    assert _select_scored_signatures(sigs, max_interventions=10, seed=0) == set(sigs)


def test_select_signatures_cap_is_deterministic_subset() -> None:
    sigs = [f"s{i}" for i in range(20)]
    a = _select_scored_signatures(sigs, max_interventions=5, seed=3)
    b = _select_scored_signatures(sigs, max_interventions=5, seed=3)
    assert a == b
    assert len(a) == 5
    assert a.issubset(set(sigs))


# --------------------------------------------------------------------------- #
# byte-weighted corpus BPB                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class _FakeScoreRow:
    bpb_intervened: float
    n_output_bytes: int


def test_byte_weighted_bpb_is_corpus_bpb() -> None:
    # bpb = sum(bpb_i * bytes_i) / sum(bytes_i)
    rows = [_FakeScoreRow(1.0, 10), _FakeScoreRow(3.0, 30)]
    bpb, total = _byte_weighted_bpb(rows)
    assert total == 40
    assert bpb == pytest.approx((1.0 * 10 + 3.0 * 30) / 40)


def test_byte_weighted_bpb_rejects_zero_bytes() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="zero scored bytes"):
        _byte_weighted_bpb([_FakeScoreRow(1.0, 0)])


# --------------------------------------------------------------------------- #
# aggregation join + macro averaging                                          #
# --------------------------------------------------------------------------- #
def _per_candidate_row(*, concept_id: int, strength: float, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "concept_id": concept_id,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "additive_direction",
        "adversarial_strength": strength,
        "strength_unit": "magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 2.0,
        "primary_margin_y": 1.0,
        "baseline_primary_margin": 1.0,
        "general_val_bpb": None,
        "general_val_corpus": None,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 5,
        "n_avoid_output_tokens": 5,
        "n_rich_output_bytes": 20,
        "n_avoid_output_bytes": 20,
        "is_control": strength == 0.0,
    }
    row.update(over)
    return row


def _per_candidate_table() -> pa.Table:
    return pa.Table.from_pylist(
        [_per_candidate_row(concept_id=1, strength=0.0)],
        schema=PER_CANDIDATE_AGG_SCHEMA,
    )


def _general_val_row(
    *,
    concept_id: int,
    strength: float,
    bpb: float,
    adversarial_method: str = "additive_direction",
    general_val_corpus: str | None = "nd=4;nb=100;h=0000000000000000",
) -> dict[str, Any]:
    return {
        "target_model": "decoder_only",
        "method_name": "diffmean_decoder_only",
        "mode": "positive",
        "concept_id": concept_id,
        "coefficient_str": "1.0",
        "adversarial_method": adversarial_method,
        "adversarial_strength": strength,
        "general_val_bpb": bpb,
        "n_documents": 4,
        "n_val_bytes": 100,
        "intervention_signature": f"sig-{concept_id}-{strength}",
        "general_val_corpus": general_val_corpus,
    }


def test_join_general_val_bpb_fills_by_intervention_key() -> None:
    per_candidate = pa.Table.from_pylist(
        [
            _per_candidate_row(concept_id=1, strength=0.0),
            _per_candidate_row(concept_id=1, strength=2.0),
        ],
        schema=PER_CANDIDATE_AGG_SCHEMA,
    )
    general_val = pa.Table.from_pylist(
        [
            _general_val_row(concept_id=1, strength=0.0, bpb=1.5),
            _general_val_row(concept_id=1, strength=2.0, bpb=2.5),
        ],
        schema=GENERAL_VAL_BPB_SCHEMA,
    )
    joined = _join_general_val_bpb(per_candidate, general_val).to_pylist()
    by_strength = {r["adversarial_strength"]: r for r in joined}
    assert by_strength[0.0]["general_val_bpb"] == pytest.approx(1.5)
    assert by_strength[0.0]["general_val_corpus"] == "nd=4;nb=100;h=0000000000000000"
    assert by_strength[2.0]["general_val_bpb"] == pytest.approx(2.5)
    assert by_strength[2.0]["general_val_corpus"] == "nd=4;nb=100;h=0000000000000000"


def test_join_general_val_bpb_missing_key_stays_null() -> None:
    per_candidate = pa.Table.from_pylist(
        [_per_candidate_row(concept_id=1, strength=4.0)],  # no matching gv row
        schema=PER_CANDIDATE_AGG_SCHEMA,
    )
    general_val = pa.Table.from_pylist(
        [_general_val_row(concept_id=1, strength=0.0, bpb=1.5)],
        schema=GENERAL_VAL_BPB_SCHEMA,
    )
    joined = _join_general_val_bpb(per_candidate, general_val).to_pylist()
    assert joined[0]["general_val_bpb"] is None
    assert joined[0]["general_val_corpus"] is None


def test_join_general_val_bpb_lands_per_k() -> None:
    # The join must assign each (faster_gcg, k) per_candidate row its OWN
    # general_val value -- the universal trigger differs per k so the values
    # differ. A join that dropped the strength key would cross-contaminate the ks.
    per_candidate = pa.Table.from_pylist(
        [
            _per_candidate_row(
                concept_id=1,
                strength=8.0,
                adversarial_method="faster_gcg",
                strength_index=1,
                is_control=False,
            ),
            _per_candidate_row(
                concept_id=1,
                strength=16.0,
                adversarial_method="faster_gcg",
                strength_index=2,
                is_control=False,
            ),
        ],
        schema=PER_CANDIDATE_AGG_SCHEMA,
    )
    general_val = pa.Table.from_pylist(
        [
            _general_val_row(
                concept_id=1,
                strength=8.0,
                bpb=2.0,
                adversarial_method="faster_gcg",
            ),
            _general_val_row(
                concept_id=1,
                strength=16.0,
                bpb=4.0,
                adversarial_method="faster_gcg",
            ),
        ],
        schema=GENERAL_VAL_BPB_SCHEMA,
    )
    joined = _join_general_val_bpb(per_candidate, general_val).to_pylist()
    by_key = {r["adversarial_strength"]: r for r in joined}
    assert by_key[8.0]["general_val_bpb"] == pytest.approx(2.0)
    assert by_key[8.0]["general_val_corpus"] == "nd=4;nb=100;h=0000000000000000"
    assert by_key[16.0]["general_val_bpb"] == pytest.approx(4.0)
    assert by_key[16.0]["general_val_corpus"] == "nd=4;nb=100;h=0000000000000000"


def test_join_rejects_general_val_value_without_corpus() -> None:
    gv_row = _general_val_row(concept_id=1, strength=0.0, bpb=3.0, general_val_corpus=None)
    with pytest.raises(AxBenchBpbAdvError, match="without a corpus signature"):
        _join_general_val_bpb(
            _per_candidate_table(),
            pa.Table.from_pylist([gv_row], schema=GENERAL_VAL_BPB_SCHEMA),
        )


def test_old_schema_general_val_parquet_is_rejected_at_join() -> None:
    # Pre-635 artifacts lack the general_val_corpus column entirely (schema v1).
    # The join must reject that at the schema gate rather than KeyError deep
    # inside the per-row loop or silently treating the corpus as always-null.
    old_schema = pa.schema([f for f in GENERAL_VAL_BPB_SCHEMA if f.name != "general_val_corpus"])
    old_table = pa.Table.from_pylist([], schema=old_schema)
    with pytest.raises(AxBenchBpbAdvError, match="GENERAL_VAL_BPB_SCHEMA"):
        _join_general_val_bpb(_per_candidate_table(), old_table)


def test_macro_averages_general_val_over_concepts() -> None:
    # Two concepts at the same intervention coordinate -> macro averages their
    # joined general_val_bpb (both concepts scored the SAME corpus).
    per_candidate = pa.Table.from_pylist(
        [
            _per_candidate_row(
                concept_id=1,
                strength=2.0,
                general_val_bpb=2.0,
                general_val_corpus="nd=4;nb=100;h=0000000000000000",
            ),
            _per_candidate_row(
                concept_id=2,
                strength=2.0,
                general_val_bpb=4.0,
                general_val_corpus="nd=4;nb=100;h=0000000000000000",
            ),
        ],
        schema=PER_CANDIDATE_AGG_SCHEMA,
    )
    macro = compute_macro_aggregation(per_candidate).to_pylist()
    assert len(macro) == 1
    assert macro[0]["general_val_bpb"] == pytest.approx(3.0)
    assert macro[0]["general_val_corpus"] == "nd=4;nb=100;h=0000000000000000"
    assert macro[0]["n_concepts"] == 2


def test_macro_general_val_null_when_all_missing() -> None:
    per_candidate = pa.Table.from_pylist(
        [_per_candidate_row(concept_id=1, strength=2.0, general_val_bpb=None)],
        schema=PER_CANDIDATE_AGG_SCHEMA,
    )
    macro = compute_macro_aggregation(per_candidate).to_pylist()
    assert macro[0]["general_val_bpb"] is None
    assert macro[0]["general_val_corpus"] is None


# --------------------------------------------------------------------------- #
# scatter x-axis branch                                                        #
# --------------------------------------------------------------------------- #
def test_scatter_baseline_x_general_val_branch() -> None:
    row = {"mode": "positive", "general_val_bpb": 2.7}
    assert _scatter_baseline_x(row, scatter_baseline_x="general_val_bpb") == pytest.approx(2.7)
    assert _scatter_baseline_x({"mode": "positive"}, scatter_baseline_x="general_val_bpb") is None


# --------------------------------------------------------------------------- #
# build_general_val_docs slicing (fake tokenizer, no model)                    #
# --------------------------------------------------------------------------- #
# A BOS id that does not collide with any ascii code the test texts produce
# (x=120, y=121, q=113, w/x/y/z=119..122), so a leading BOS is unambiguous.
_FAKE_BOS_ID = 255


class _FakeTokenizer:
    """Whitespace tokenizer-ish stub: encode -> ascii codes, decode -> chars.

    Supports the BOS-additive contract the universal-trigger general_val needs:
    ``get_bos_token_id()`` returns a fixed id and ``encode(text, prepend=bos)``
    prepends it (mirroring the real nanochat tokenizer's ``prepend`` kwarg).
    """

    def get_bos_token_id(self) -> int:
        return _FAKE_BOS_ID

    def encode(self, text: str, *, prepend: int | None = None) -> list[int]:
        ids = [ord(c) % 256 for c in text]
        if prepend is not None:
            return [int(prepend), *ids]
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_build_general_val_docs_slices_and_skips_short() -> None:
    tok = _FakeTokenizer()
    token_bytes = t.ones(256, dtype=t.int64)  # 1 byte per token id
    # The too-short text comes FIRST so the loop must actually take the skip
    # branch before finding a survivor (long-first would break at n_documents=1
    # without ever reaching the short text, leaving the skip untested).
    texts = ["y" * 5, "x" * 50]  # first too short for prefix+output
    docs = build_general_val_docs(
        texts=iter(texts),
        tokenizer=tok,
        token_bytes=token_bytes,
        n_documents=1,
        prefix_tokens=20,
        output_tokens=10,
    )
    assert len(docs) == 1
    doc = docs[0]
    assert isinstance(doc, _GeneralValDoc)
    # The survivor is the SECOND text: the first was skipped, and sample_id
    # carries the original stream index (not a post-skip renumbering).
    assert doc.sample_id == "general_val_1"
    # Preserve main's BOS-additive contract while adding the stricter single-BOS
    # guard: prefix == [BOS] + first 20 real tokens, with no BOS in content.
    assert len(doc.prefix_tokens) == 21
    assert doc.prefix_tokens[0] == _FAKE_BOS_ID
    assert _FAKE_BOS_ID not in doc.prefix_tokens[1:]
    # The scored OUTPUT continuation is the SAME real tokens that follow the
    # prefix's real span -- BOS must NOT shift it (M4).
    assert len(doc.output_tokens) == 10
    real = tok.encode("x" * 50)  # no BOS
    assert doc.prefix_tokens[1:] == real[:20]
    assert doc.output_tokens == real[20:30]
    assert int(doc.output_byte_weights.sum().item()) == 10


def test_build_general_val_docs_output_window_unshifted_by_bos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The OUTPUT contract (M4): prepending BOS must not move the scored
    # continuation. With distinct per-position tokens we pin that output ==
    # real[prefix_tokens : prefix_tokens + output_tokens], independent of BOS.
    monkeypatch.setattr(general_val, "logger", MagicMock())
    tok = _FakeTokenizer()
    token_bytes = t.ones(256, dtype=t.int64)
    # 12 distinct ascii chars 'a'..'l' (97..108), none equal to the BOS id.
    text = "abcdefghijkl"
    docs = build_general_val_docs(
        texts=iter([text]),
        tokenizer=tok,
        token_bytes=token_bytes,
        n_documents=1,
        prefix_tokens=4,
        output_tokens=3,
    )
    assert len(docs) == 1
    doc = docs[0]
    real = tok.encode(text)  # [97, 98, ..., 108], no BOS
    assert doc.prefix_tokens == [_FAKE_BOS_ID, *real[:4]]  # BOS + a b c d
    assert doc.output_tokens == real[4:7]  # e f g  (unshifted)


def test_build_general_val_docs_rejects_raw_double_bos_source() -> None:
    tok = _FakeTokenizer()
    token_bytes = t.ones(256, dtype=t.int64)

    with pytest.raises(AxBenchBpbAdvError, match="already starts with BOS"):
        build_general_val_docs(
            texts=iter([chr(_FAKE_BOS_ID) + "abcdef"]),
            tokenizer=tok,
            token_bytes=token_bytes,
            n_documents=1,
            prefix_tokens=3,
            output_tokens=2,
        )


def test_build_general_val_docs_rejects_raw_extra_bos_in_prefix() -> None:
    tok = _FakeTokenizer()
    token_bytes = t.ones(256, dtype=t.int64)

    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        build_general_val_docs(
            texts=iter(["a" + chr(_FAKE_BOS_ID) + "bcdef"]),
            tokenizer=tok,
            token_bytes=token_bytes,
            n_documents=1,
            prefix_tokens=4,
            output_tokens=1,
        )


def test_build_general_val_docs_raises_when_all_too_short() -> None:
    tok = _FakeTokenizer()
    token_bytes = t.ones(256, dtype=t.int64)
    with pytest.raises(AxBenchBpbAdvError, match="long enough"):
        build_general_val_docs(
            texts=iter(["short"]),
            tokenizer=tok,
            token_bytes=token_bytes,
            n_documents=1,
            prefix_tokens=20,
            output_tokens=10,
        )


def test_build_general_val_docs_selects_first_n_survivors() -> None:
    tokenizer = _FakeTokenizer()
    # texts[1] too short to supply prefix+output; the rest survive.
    texts = ["a b c d e f", "a", "g h i j k l", "m n o p q r", "s t u v w x"]
    docs = build_general_val_docs(
        texts=iter(texts),
        tokenizer=tokenizer,
        token_bytes=t.ones(256, dtype=t.int64),
        n_documents=3,
        prefix_tokens=2,
        output_tokens=2,
    )
    assert [d.sample_id for d in docs] == ["general_val_0", "general_val_2", "general_val_3"]
    assert all(len(d.output_tokens) == 2 for d in docs)


def test_build_general_val_docs_raises_on_exhaustion() -> None:
    tokenizer = _FakeTokenizer()
    with pytest.raises(AxBenchBpbAdvError, match="found 1 of 2"):
        build_general_val_docs(
            texts=iter(["a b c d e f", "a"]),
            tokenizer=tokenizer,
            token_bytes=t.ones(256, dtype=t.int64),
            n_documents=2,
            prefix_tokens=2,
            output_tokens=2,
        )


def test_build_general_val_docs_empty_corpus_actionable_message() -> None:
    # A completely empty stream means base_data has no val shard at all: the
    # error must carry the download instructions, not the generic found-0-of-N.
    tokenizer = _FakeTokenizer()
    with pytest.raises(AxBenchBpbAdvError, match="Download a shard"):
        build_general_val_docs(
            texts=iter([]),
            tokenizer=tokenizer,
            token_bytes=t.ones(256, dtype=t.int64),
            n_documents=1,
            prefix_tokens=2,
            output_tokens=2,
        )


# --------------------------------------------------------------------------- #
# _corpus_signature: self-describing per-target corpus identity (D2)          #
# --------------------------------------------------------------------------- #
def test_corpus_signature_format_and_sensitivity() -> None:
    tokenizer = _FakeTokenizer()
    docs = build_general_val_docs(
        texts=iter(["a b c d e f", "g h i j k l"]),
        tokenizer=tokenizer,
        token_bytes=t.ones(256, dtype=t.int64),
        n_documents=2,
        prefix_tokens=2,
        output_tokens=2,
    )
    sig = _corpus_signature(docs, prefix_tokens=2, output_tokens=2)
    total = sum(int(d.output_byte_weights.sum().item()) for d in docs)
    assert sig.startswith(f"nd=2;nb={total};h=")
    assert len(sig.split("h=")[1]) == 32
    # order/content sensitivity
    assert _corpus_signature(list(reversed(docs)), prefix_tokens=2, output_tokens=2) != sig
    assert _corpus_signature(docs, prefix_tokens=3, output_tokens=2) != sig
    # attacked forks share the CLEAN corpus: the signature is doc-identity only,
    # so the same docs always hash identically.
    assert _corpus_signature(docs, prefix_tokens=2, output_tokens=2) == sig


def test_corpus_signature_is_pure_function_of_doc_content() -> None:
    # The comparability guarantee ("every plotted method scored on the SAME
    # corpus") rests on the signature being a pure function of the docs'
    # CONTENT -- independent of object identity, intervention, or method. Two
    # independently-built doc lists with byte-identical content hash the same;
    # changing a single output token id (same length) changes the hash.
    def _doc(sample_id: str, prefix: list[int], output: list[int]) -> _GeneralValDoc:
        return _GeneralValDoc(
            sample_id=sample_id,
            prefix_tokens=prefix,
            output_tokens=output,
            prefix_text="ignored-by-signature",
            output_byte_weights=t.ones(len(output), dtype=t.int64),
        )

    docs_a = [_doc("general_val_0", [1, 2, 3], [4, 5]), _doc("general_val_1", [6, 7], [8, 9])]
    docs_b = [_doc("general_val_0", [1, 2, 3], [4, 5]), _doc("general_val_1", [6, 7], [8, 9])]
    sig_a = _corpus_signature(docs_a, prefix_tokens=3, output_tokens=2)
    # Distinct objects, identical content -> identical signature (method-independent).
    assert _corpus_signature(docs_b, prefix_tokens=3, output_tokens=2) == sig_a

    # Flip ONE output token id (same length) -> different corpus.
    docs_c = [_doc("general_val_0", [1, 2, 3], [4, 5]), _doc("general_val_1", [6, 7], [8, 99])]
    assert _corpus_signature(docs_c, prefix_tokens=3, output_tokens=2) != sig_a
    # Change a sample_id (same tokens) -> different corpus (doc identity matters).
    docs_d = [_doc("general_val_0", [1, 2, 3], [4, 5]), _doc("general_val_2", [6, 7], [8, 9])]
    assert _corpus_signature(docs_d, prefix_tokens=3, output_tokens=2) != sig_a


# --------------------------------------------------------------------------- #
# _extract_encdec_segments: per-doc raise on empty feature-segment tuple (D7)  #
#                                                                             #
# extract_prefix_features is stubbed directly (no real Gemma model); the       #
# surrounding plumbing (parse_neuronpedia_source, load_statistics,             #
# build_boundaries_from_stats) is either given real-but-inert inputs (the      #
# default neuronpedia_source parses fine) or stubbed via SimpleNamespace,      #
# mirroring tests/steering/test_build_eval_context_cache.py's                  #
# ``load_statistics`` stub convention.                                         #
# --------------------------------------------------------------------------- #
def _seg() -> ContextFeatureSegment:
    return ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=1)


def _stub_runtime(*, expected_tail_after: int | None = None) -> Any:
    # Duck-typed RuntimeState stand-in: _extract_encdec_segments only calls
    # .base_gemma_and_sae(). base_model=None short-circuits the post-loop
    # `base_model.to("cpu")` offload (guarded by `base_model is not None`), so
    # the stub base_sae only needs `.cfg.d_sae` (consumed by
    # build_boundaries_from_stats to size the boundaries tensor).
    def base_gemma_and_sae(
        *, offload_transformer_tail_after: int | None = None
    ) -> tuple[None, SimpleNamespace]:
        assert offload_transformer_tail_after == expected_tail_after
        return None, SimpleNamespace(cfg=SimpleNamespace(d_sae=4))

    return SimpleNamespace(base_gemma_and_sae=base_gemma_and_sae)


# ``_extract_encdec_segments`` takes its snapshot from the caller: production
# hands it the exact object the stage already validated, never a re-read file.
_STATISTICS = fixed_statistics_double()


def _cfg() -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats.safetensors",
    )


def test_general_val_extraction_handoff_orders_unload_release_and_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    restored_decoder = object()
    restored_encdec = object()

    class State:
        def release_target_models(self) -> None:
            assert bundles == {}
            events.append("release_targets")

        def release_base_gemma_and_sae(self) -> None:
            events.append("release_extractor_and_sae")

        def decoder_only(self) -> object:
            events.append("restore_decoder")
            return restored_decoder

        def encdec(self) -> object:
            events.append("restore_encdec")
            return restored_encdec

    monkeypatch.setattr(
        general_val,
        "release_cuda_iteration_memory",
        lambda: events.append("release_allocator"),
    )
    bundles: dict[str, Any] = {
        "decoder_only": object(),
        "encdec": object(),
    }

    result = general_val._extract_with_target_memory_handoff(
        runtime_state=cast(RuntimeState, State()),
        target_bundles=cast(dict[str, TargetModelBundle], bundles),
        extract=lambda: events.append("extract") or {"sample": (_seg(),)},
    )

    assert result == {"sample": (_seg(),)}
    assert bundles == {
        "decoder_only": restored_decoder,
        "encdec": restored_encdec,
    }
    assert events == [
        "release_targets",
        "release_allocator",
        "extract",
        "release_extractor_and_sae",
        "release_allocator",
        "restore_decoder",
        "restore_encdec",
    ]


def test_general_val_extraction_handoff_cleans_up_without_restore_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class State:
        def release_target_models(self) -> None:
            events.append("release_targets")

        def release_base_gemma_and_sae(self) -> None:
            events.append("release_extractor_and_sae")

        def decoder_only(self) -> object:
            raise AssertionError("failed extraction must not restore targets")

        def encdec(self) -> object:
            raise AssertionError("failed extraction must not restore targets")

    def fail_extraction() -> dict[str, tuple[ContextFeatureSegment, ...]]:
        events.append("extract")
        raise RuntimeError("injected extraction failure")

    monkeypatch.setattr(
        general_val,
        "release_cuda_iteration_memory",
        lambda: events.append("release_allocator"),
    )

    with pytest.raises(RuntimeError, match="injected extraction failure"):
        general_val._extract_with_target_memory_handoff(
            runtime_state=cast(RuntimeState, State()),
            target_bundles=cast(
                dict[str, TargetModelBundle],
                {"decoder_only": object(), "encdec": object()},
            ),
            extract=fail_extraction,
        )

    assert events == [
        "release_targets",
        "release_allocator",
        "extract",
        "release_extractor_and_sae",
        "release_allocator",
    ]


def test_roofline_bag_config_metadata_includes_model_identity() -> None:
    cfg = _cfg()
    cfg.neuronpedia_model_id = "gemma-2-2b"
    cfg.neuronpedia_source = "8-gemmascope-res-16k"
    boundaries = t.tensor([[0.5], [1.0]], dtype=t.float32)

    metadata = _roofline_bag_config_metadata(cfg, boundaries=boundaries, sae_layer=8)

    assert metadata["neuronpedia_model_id"] == "gemma-2-2b"
    assert metadata["neuronpedia_source"] == "8-gemmascope-res-16k"
    changed = _roofline_bag_config_metadata(
        replace(cfg, neuronpedia_model_id="gemma-3-4b-it"),
        boundaries=boundaries,
        sae_layer=8,
    )
    assert changed["neuronpedia_model_id"] == "gemma-3-4b-it"
    assert changed["neuronpedia_model_id"] != metadata["neuronpedia_model_id"]


def test_extract_encdec_segments_raises_on_empty_segment_doc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Every scored doc must condition (D7): a doc whose extract_prefix_features
    # returns an empty tuple used to be silently warned-and-skipped, forking the
    # encdec corpus from the decoder corpus. It must now raise immediately,
    # naming the offending doc's sample_id.
    tokenizer = _FakeTokenizer()
    docs = build_general_val_docs(
        texts=iter(["a b c d e f", "g h i j k l"]),
        tokenizer=tokenizer,
        token_bytes=t.ones(256, dtype=t.int64),
        n_documents=2,
        prefix_tokens=2,
        output_tokens=2,
    )
    assert len(docs) == 2
    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda _path: pytest.fail("extraction must use the caller's statistics"),
    )
    monkeypatch.setattr(
        general_val,
        "extract_prefix_features",
        lambda **kw: () if kw["prefix_text"] == docs[1].prefix_text else (_seg(),),
    )
    with pytest.raises(AxBenchBpbAdvError, match="produced no encdec feature segments"):
        general_val._extract_encdec_segments(
            docs,
            runtime_state=_stub_runtime(),
            cfg=_cfg(),
            statistics=_STATISTICS,
        )


def test_clustered_general_val_context_replays_all_docs_and_segments_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    docs = [
        _GeneralValDoc(
            sample_id="general_val_0",
            prefix_tokens=[1, 2],
            output_tokens=[3],
            prefix_text="first",
            output_byte_weights=t.ones(1, dtype=t.int64),
        ),
        _GeneralValDoc(
            sample_id="general_val_1",
            prefix_tokens=[4, 5],
            output_tokens=[6],
            prefix_text="second",
            output_byte_weights=t.ones(1, dtype=t.int64),
        ),
    ]
    raw_by_prefix = {
        "first": RawClusteredPrefix(
            segments=(
                RawClusteredFeatureSegment(
                    feature_ids=t.tensor([1]),
                    feature_magnitudes=t.tensor([1.0]),
                    char_start=0,
                    char_end=2,
                ),
                RawClusteredFeatureSegment(
                    feature_ids=t.tensor([2]),
                    feature_magnitudes=t.tensor([2.0]),
                    char_start=2,
                    char_end=5,
                ),
            ),
            n_tokens=2,
        ),
        "second": RawClusteredPrefix(
            segments=(
                RawClusteredFeatureSegment(
                    feature_ids=t.tensor([3]),
                    feature_magnitudes=t.tensor([3.0]),
                    char_start=0,
                    char_end=6,
                ),
            ),
            n_tokens=2,
        ),
    }

    class _Assertions:
        def assert_representatives_only(self, ids: t.Tensor, *, surface: str) -> None:
            assert surface == "clustered general_val context segment"
            assert ids.dtype == t.int64

    class _Mask:
        def assert_no_masked(self, ids: t.Tensor) -> None:
            assert ids.dtype == t.int64

    contract: Any = SimpleNamespace(
        feature_directory=SimpleNamespace(segment_size=128),
        selection=SimpleNamespace(num_buckets=2),
        consolidation=_Assertions(),
        feature_mask=_Mask(),
    )
    cfg = _cfg()
    cfg.extraction_threshold = 0.0
    cfg.context_cache.segment_size = 128
    cfg.context_cache.num_buckets = 2
    replay_states: list[object] = []
    selection_calls: list[tuple[object, int, float]] = []

    def _reconstruct(_contract: Any) -> object:
        state = object()
        replay_states.append(state)
        return state

    def _select(raw_segment: RawClusteredFeatureSegment, *, replay_state: Any, contract: Any):
        del contract
        selection_calls.append(
            (
                replay_state,
                raw_segment.char_start,
                float(raw_segment.feature_magnitudes.item()),
            )
        )
        positive_id = int(raw_segment.feature_ids.item())
        return SimpleNamespace(
            positive_ids=t.tensor([positive_id], dtype=t.int64),
            negative_ids=t.tensor([7], dtype=t.int64),
            positive_buckets=t.tensor([0], dtype=t.int64),
        )

    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda _path: pytest.fail("extraction must use the caller's statistics"),
    )
    monkeypatch.setattr(
        general_val,
        "build_boundaries_from_stats",
        lambda *args, **kwargs: t.zeros((8, 1), dtype=t.float64),
    )
    monkeypatch.setattr(
        general_val,
        "reconstruct_clustered_replay_state",
        _reconstruct,
    )
    monkeypatch.setattr(
        general_val,
        "extract_raw_clustered_prefix",
        lambda **kwargs: raw_by_prefix[kwargs["prefix_text"]],
    )
    monkeypatch.setattr(general_val, "select_raw_clustered_segment", _select)

    first = general_val._extract_encdec_segments(
        docs,
        runtime_state=_stub_runtime(expected_tail_after=8),
        cfg=cfg,
        statistics=_STATISTICS,
        cluster_contract=contract,
        corpus_signature="corpus-v1",
    )
    second = general_val._extract_encdec_segments(
        docs,
        runtime_state=_stub_runtime(expected_tail_after=8),
        cfg=cfg,
        statistics=_STATISTICS,
        cluster_contract=contract,
        corpus_signature="corpus-v1",
    )

    assert first == second
    assert first == {
        "general_val_0": (
            ContextFeatureSegment((1, 7), (0, 1), 0, 2),
            ContextFeatureSegment((2, 7), (0, 1), 2, 5),
        ),
        "general_val_1": (ContextFeatureSegment((3, 7), (0, 1), 0, 6),),
    }
    assert len(replay_states) == 2
    assert selection_calls == [
        (replay_states[0], 0, 1.0),
        (replay_states[0], 2, 2.0),
        (replay_states[0], 0, 3.0),
        (replay_states[1], 0, 1.0),
        (replay_states[1], 2, 2.0),
        (replay_states[1], 0, 3.0),
    ]


# --------------------------------------------------------------------------- #
# _extract_encdec_segments: allowed_ids threads into derive_real_output_signal #
# (issue #695 PR-1 review, C2) -- proves the WIRING run_general_val_bpb_stage  #
# relies on: the production call site passes                                  #
# feature_vocab_allowed_ids(target_bundles["encdec"].model) as allowed_ids,    #
# and this must reach derive_real_output_signal's real selection so a         #
# compacted model's V is actually enforced (derive_real_output_signal's OWN   #
# allowed_ids handling is already covered end-to-end in                       #
# test_general_val_roofline_vocab.py; this only proves the caller wires it    #
# through). _max_pooled_sae_activations is stubbed at the lowest real seam    #
# (mirroring test_general_val_roofline_vocab.py) so the real                  #
# derive_real_output_signal + _select_signal_indices ranking code runs.       #
# --------------------------------------------------------------------------- #
def _uniform_percentile_table(n_features: int) -> t.Tensor:
    """``[n_features, 99]`` table where percentile ranking == magnitude ranking."""
    row = t.linspace(1.0 / 99.0, 1.0, 99, dtype=t.float64)
    return row.unsqueeze(0).repeat(n_features, 1)


def test_extract_encdec_segments_threads_allowed_ids_into_derive_real_output_signal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _FakeTokenizer()
    docs = build_general_val_docs(
        texts=iter(["a b c d e f"]),
        tokenizer=tokenizer,
        token_bytes=t.ones(256, dtype=t.int64),
        n_documents=1,
        prefix_tokens=2,
        output_tokens=2,
    )
    assert len(docs) == 1
    cfg = _cfg()
    cfg.extraction_threshold = 0.1
    monkeypatch.setattr(
        general_val,
        "load_statistics",
        lambda _path: pytest.fail("extraction must use the caller's statistics"),
    )
    monkeypatch.setattr(general_val, "extract_prefix_features", lambda **kw: (_seg(),))
    monkeypatch.setattr(
        general_val,
        "build_boundaries_from_stats",
        lambda *args, **kwargs: t.tensor([[1.0]] * 4, dtype=t.float64),
    )
    monkeypatch.setattr(
        general_val,
        "build_percentile_table",
        lambda *args, **kwargs: _uniform_percentile_table(4),
    )
    monkeypatch.setattr(
        general_val_roofline,
        "_max_pooled_sae_activations",
        # feature 3 fires highest (0.9) -- the unrestricted top -- but sits
        # OUTSIDE V below. feature 2 fires second-highest (0.7) and IS in V.
        lambda *args, **kwargs: t.tensor([[0.3, 0.5, 0.7, 0.9]], dtype=t.float32),
    )
    signal_budget = GeneralValSignalBudget(
        k_pos=1, k_neg=0, positive_bucket=0, negative_bucket=1
    )

    # Unrestricted (allowed_ids omitted): sanity-check the global top wins.
    unrestricted_state = _RooflineGeneralValState.for_budget(signal_budget)
    general_val._extract_encdec_segments(
        docs,
        runtime_state=_stub_runtime(),
        cfg=cfg,
        statistics=_STATISTICS,
        roofline_state=unrestricted_state,
        tokenizer=tokenizer,
        device=t.device("cpu"),
    )
    unrestricted_bag = dict(unrestricted_state.real_bags[docs[0].sample_id].entries)
    assert unrestricted_bag == {3: 0}

    # Restricted: the production shape -- run_general_val_bpb_stage passes
    # feature_vocab_allowed_ids(target_bundles["encdec"].model) as allowed_ids.
    allowed = t.tensor([0, 1, 2], dtype=t.int64)  # V excludes feature 3
    restricted_state = _RooflineGeneralValState.for_budget(signal_budget)
    general_val._extract_encdec_segments(
        docs,
        runtime_state=_stub_runtime(),
        cfg=cfg,
        statistics=_STATISTICS,
        roofline_state=restricted_state,
        tokenizer=tokenizer,
        device=t.device("cpu"),
        allowed_ids=allowed,
    )
    restricted_bag = dict(restricted_state.real_bags[docs[0].sample_id].entries)
    # (a) the selected id stays within V.
    assert set(restricted_bag) <= {0, 1, 2}
    # (b) non-vacuous: the excluded global top (feature 3) never appears, and a
    # DIFFERENT (lower-ranked, in-V) feature is chosen in its place.
    assert 3 not in restricted_bag
    assert restricted_bag == {2: 0}
    assert restricted_bag != unrestricted_bag


# --------------------------------------------------------------------------- #
# universal-trigger lookup (decoder-only doc attack)                           #
#                                                                             #
# general_val substitutes the SAME per-concept universal trigger into the      #
# validation docs so general_val_bpb is measured under the attack. The trigger #
# is keyed by the 7-tuple (method_name, target_model, concept_id,              #
# coefficient_str, adversarial_method, strength_index, mode)                   #
# -- the group identity the candidate stage stamps on each trigger row, which  #
# INCLUDES method_name + target_model (the default config writes several rows  #
# that share the other five fields but differ on method/target).              #
# --------------------------------------------------------------------------- #
def _trigger_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=UNIVERSAL_TRIGGERS_SCHEMA)


def _trigger_row(
    *,
    concept_id: int,
    coefficient_str: str,
    strength_index: int,
    mode: str,
    trigger_tokens: list[int],
    adversarial_method: str = "faster_gcg",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": adversarial_method,
        "strength_index": strength_index,
        "adversarial_strength": float(len(trigger_tokens)),
        "trigger_tokens": trigger_tokens,
        "trigger_text": "".join(chr(tok) for tok in trigger_tokens),
        "objective_value": 0.0,
        "max_steps": 4,
        "accepted_steps": 1,
        "visited_sequences": 2,
        "n_pairs": 3,
    }


def _trigger_combo(
    *,
    concept_id: int = 1,
    coefficient_str: str = "1.0",
    strength_index: int = 1,
    mode: str = "positive",
    adversarial_method: str = "faster_gcg",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
) -> _InterventionCombo:
    return _InterventionCombo(
        target_model=target_model,
        method_name=method_name,
        mode=mode,
        concept_id=concept_id,
        coefficient_str=coefficient_str,
        adversarial_method=adversarial_method,
        adversarial_strength=float(strength_index),
        strength_index=strength_index,
    )


def test_lookup_universal_trigger_resolves_unique_per_7tuple() -> None:
    index = general_val._build_trigger_index(
        _trigger_table(
            [
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[10, 11],
                ),
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=2,
                    mode="positive",
                    trigger_tokens=[30, 31, 32],
                ),
                _trigger_row(
                    concept_id=2,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[40, 41],
                ),
            ]
        )
    )
    # Each (concept, coeff, adv, k, mode) resolves its OWN trigger.
    assert _lookup_universal_trigger(index, _trigger_combo(concept_id=1, strength_index=1)) == [
        10,
        11,
    ]
    assert _lookup_universal_trigger(index, _trigger_combo(concept_id=1, strength_index=2)) == [
        30,
        31,
        32,
    ]
    assert _lookup_universal_trigger(index, _trigger_combo(concept_id=2, strength_index=1)) == [
        40,
        41,
    ]


def test_lookup_universal_trigger_missing_raises() -> None:
    index = general_val._build_trigger_index(
        _trigger_table(
            [
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[10, 11],
                ),
            ]
        )
    )
    # A faster_gcg k>0 combo with no matching trigger row is a fail-loud error.
    with pytest.raises(AxBenchBpbAdvError, match="no universal trigger"):
        _lookup_universal_trigger(
            index,
            _trigger_combo(concept_id=1, strength_index=2),
        )


def test_build_trigger_index_rejects_duplicate_7tuple() -> None:
    # Two trigger rows colliding on the full 7-key (these helpers default the
    # same method_name/target_model) would make the substitution ambiguous; the
    # index build must reject that loudly.
    with pytest.raises(AxBenchBpbAdvError, match="duplicate universal trigger"):
        general_val._build_trigger_index(
            _trigger_table(
                [
                    _trigger_row(
                        concept_id=1,
                        coefficient_str="1.0",
                        strength_index=1,
                        mode="positive",
                        trigger_tokens=[10, 11],
                    ),
                    _trigger_row(
                        concept_id=1,
                        coefficient_str="1.0",
                        strength_index=1,
                        mode="positive",
                        trigger_tokens=[99, 98],
                    ),
                ]
            )
        )


def test_build_trigger_index_accepts_cross_method_same_5tuple() -> None:
    # REGRESSION (high): the artifact writes ONE trigger row per concept GROUP,
    # whose identity is the 7-tuple INCLUDING method_name + target_model
    # (UNIVERSAL_TRIGGERS_SCHEMA). With the default config's multiple
    # decoder-only steering families (diffmean_decoder_only,
    # linear_probe_decoder_only, ...) two distinct groups at the SAME
    # (concept, coeff, adv, k, mode) but different method_name are
    # legitimately separate trigger rows. Indexing on only the 5-tuple
    # mis-rejected them as a "duplicate"; the index must key on method_name too
    # and resolve each method's own trigger.
    index = general_val._build_trigger_index(
        _trigger_table(
            [
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[10, 11],
                    method_name="diffmean_decoder_only",
                ),
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[20, 21],
                    method_name="linear_probe_decoder_only",
                ),
            ]
        )
    )
    assert _lookup_universal_trigger(
        index, _trigger_combo(method_name="diffmean_decoder_only")
    ) == [10, 11]
    assert _lookup_universal_trigger(
        index, _trigger_combo(method_name="linear_probe_decoder_only")
    ) == [20, 21]


def test_build_trigger_index_accepts_cross_target_same_5tuple() -> None:
    # REGRESSION (high): the default config sets target_models=[decoder_only,
    # encdec], so the candidate stage loops per target and stamps target_model
    # on each group's trigger row. A decoder_only and an encdec group at the
    # otherwise-identical (concept, coeff, adv, k, mode) are distinct
    # groups (distinct triggers); the 5-tuple index collided them and aborted
    # the whole general_val stage at _build_trigger_index (called BEFORE the
    # decoder-only combo filter). The index must key on target_model too.
    index = general_val._build_trigger_index(
        _trigger_table(
            [
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[10, 11],
                    method_name="diffmean_decoder_only",
                    target_model="decoder_only",
                ),
                _trigger_row(
                    concept_id=1,
                    coefficient_str="1.0",
                    strength_index=1,
                    mode="positive",
                    trigger_tokens=[30, 31],
                    method_name="diffmean_encdec",
                    target_model="encdec",
                ),
            ]
        )
    )
    assert _lookup_universal_trigger(
        index,
        _trigger_combo(method_name="diffmean_decoder_only", target_model="decoder_only"),
    ) == [10, 11]
    assert _lookup_universal_trigger(
        index,
        _trigger_combo(method_name="diffmean_encdec", target_model="encdec"),
    ) == [30, 31]


def test_build_trigger_index_still_rejects_full_7tuple_duplicate() -> None:
    # Two rows colliding on the FULL group identity (incl. method_name +
    # target_model) remain a corruption of the artifact and must raise: the
    # candidate stage writes exactly one trigger row per group, so a 7-tuple
    # collision means a last-write-wins would silently substitute the wrong
    # trigger.
    with pytest.raises(AxBenchBpbAdvError, match="duplicate universal trigger"):
        general_val._build_trigger_index(
            _trigger_table(
                [
                    _trigger_row(
                        concept_id=1,
                        coefficient_str="1.0",
                        strength_index=1,
                        mode="positive",
                        trigger_tokens=[10, 11],
                        method_name="diffmean_decoder_only",
                        target_model="decoder_only",
                    ),
                    _trigger_row(
                        concept_id=1,
                        coefficient_str="1.0",
                        strength_index=1,
                        mode="positive",
                        trigger_tokens=[99, 98],
                        method_name="diffmean_decoder_only",
                        target_model="decoder_only",
                    ),
                ]
            )
        )


# --------------------------------------------------------------------------- #
# _attack_doc: last-k trigger substitution into a BOS-prefixed val doc          #
# --------------------------------------------------------------------------- #
def _make_doc(prefix_tokens: list[int], output_tokens: list[int]) -> _GeneralValDoc:
    tok = _FakeTokenizer()
    return _GeneralValDoc(
        sample_id="general_val_0",
        prefix_tokens=prefix_tokens,
        output_tokens=output_tokens,
        prefix_text=tok.decode(prefix_tokens),
        output_byte_weights=t.ones(len(output_tokens), dtype=t.int64),
    )


def test_attack_doc_substitutes_last_k() -> None:
    # The trigger replaces the LAST k tokens of the BOS-prefixed prefix; BOS at
    # position 0 and the earlier real tokens are untouched; the scored OUTPUT is
    # unchanged; prefix_text is re-decoded from the attacked tokens.
    tok = _FakeTokenizer()
    doc = _make_doc([_FAKE_BOS_ID, 97, 98, 99, 100], [101, 102])  # BOS a b c d / e f
    attacked = _attack_doc(doc, trigger=[111, 112], tokenizer=tok)  # 'o','p'
    assert attacked.prefix_tokens == [_FAKE_BOS_ID, 97, 98, 111, 112]
    assert attacked.prefix_tokens[0] == _FAKE_BOS_ID  # BOS preserved
    assert attacked.output_tokens == [101, 102]  # output untouched
    assert attacked.prefix_text == tok.decode([_FAKE_BOS_ID, 97, 98, 111, 112])
    assert attacked.sample_id == doc.sample_id
    # The byte-weights tensor for the (unchanged) output carries over.
    assert int(attacked.output_byte_weights.sum().item()) == 2


def test_attack_doc_k_equals_all_real_tokens_keeps_bos() -> None:
    # k == every real token (prefix has BOS + k reals): the whole real span is
    # replaced but BOS at 0 stays (select_final_k_editable_span never edits BOS).
    tok = _FakeTokenizer()
    doc = _make_doc([_FAKE_BOS_ID, 97, 98], [99])
    attacked = _attack_doc(doc, trigger=[120, 121], tokenizer=tok)
    assert attacked.prefix_tokens == [_FAKE_BOS_ID, 120, 121]


def test_attack_doc_rejects_trigger_that_would_create_double_bos() -> None:
    tok = _FakeTokenizer()
    doc = _make_doc([_FAKE_BOS_ID, 97, 98], [99])

    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        _attack_doc(doc, trigger=[_FAKE_BOS_ID, 120], tokenizer=tok)


def test_attack_doc_rejects_non_adjacent_trigger_bos() -> None:
    tok = _FakeTokenizer()
    doc = _make_doc([_FAKE_BOS_ID, 97, 98, 99, 100], [101])

    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        _attack_doc(doc, trigger=[120, _FAKE_BOS_ID], tokenizer=tok)


def test_attack_doc_length_preserving() -> None:
    tok = _FakeTokenizer()
    doc = _make_doc([_FAKE_BOS_ID, 97, 98, 99, 100, 101], [102, 103])
    attacked = _attack_doc(doc, trigger=[200, 201, 202], tokenizer=tok)
    assert len(attacked.prefix_tokens) == len(doc.prefix_tokens)
    assert len(attacked.output_tokens) == len(doc.output_tokens)


def test_attack_doc_distinct_triggers_give_distinct_prefixes() -> None:
    # Per-k distinct triggers ⇒ distinct attacked docs ⇒ the downstream
    # general_val_bpb is non-flat across the GCG axis (the whole point). Pin that
    # two different triggers (different k included) produce different prefixes.
    tok = _FakeTokenizer()
    doc = _make_doc([_FAKE_BOS_ID, 97, 98, 99, 100], [101])
    a = _attack_doc(doc, trigger=[111, 112], tokenizer=tok)
    b = _attack_doc(doc, trigger=[113, 114], tokenizer=tok)
    c = _attack_doc(doc, trigger=[115, 116, 117], tokenizer=tok)  # k=3
    assert a.prefix_tokens != b.prefix_tokens
    assert a.prefix_tokens != c.prefix_tokens
    assert b.prefix_tokens != c.prefix_tokens
    # k=0 (empty trigger) is the clean baseline: the doc is returned unchanged.
    clean = _attack_doc(doc, trigger=[], tokenizer=tok)
    assert clean.prefix_tokens == doc.prefix_tokens
    assert clean.output_tokens == doc.output_tokens


# --------------------------------------------------------------------------- #
# _distinct_combos: roofline precomputed bags now flow through                #
# --------------------------------------------------------------------------- #
# Real method-name strings from PartitionModifier (StrEnum values).
_ROOFLINE_METHOD = "ours_gcg_roofline_encdec"
_ROOFLINE_FEATURE_DIFFMEAN_METHOD = "feature_diffmean_roofline_encdec"
_FAIR_GCG_METHOD = "ours_gcg_encdec"
_FAIR_FEATURE_DIFFMEAN_METHOD = "feature_diffmean_encdec"
_NON_GCG_METHOD = "diffmean_encdec"

_DISTINCT_COMBOS_SCHEMA = pa.schema(
    [
        ("target_model", pa.string()),
        ("method_name", pa.string()),
        ("mode", pa.string()),
        ("concept_id", pa.int32()),
        ("coefficient_str", pa.string()),
        ("adversarial_method", pa.string()),
        ("adversarial_strength", pa.float64()),
        ("strength_index", pa.int32()),
    ]
)


def _dc_row(method_name: str, *, concept_id: int = 1) -> dict[str, Any]:
    return {
        "target_model": "encdec",
        "method_name": method_name,
        "mode": "positive",
        "concept_id": concept_id,
        "coefficient_str": "1.0",
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_index": 0,
    }


def _dc_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=_DISTINCT_COMBOS_SCHEMA)


@pytest.mark.parametrize(
    "method_name",
    [_ROOFLINE_METHOD, _ROOFLINE_FEATURE_DIFFMEAN_METHOD],
)
def test_distinct_combos_keeps_roofline(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
) -> None:
    monkeypatch.setattr(general_val, "logger", MagicMock())
    table = _dc_table([_dc_row(method_name)])
    combos = _distinct_combos(table)
    assert [combo.method_name for combo in combos] == [method_name]


def test_distinct_combos_keeps_roofline_mixed_with_fair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(general_val, "logger", MagicMock())
    table = _dc_table(
        [
            _dc_row(_FAIR_GCG_METHOD),
            _dc_row(_FAIR_FEATURE_DIFFMEAN_METHOD),
            _dc_row(_NON_GCG_METHOD),
            _dc_row(_ROOFLINE_FEATURE_DIFFMEAN_METHOD),
        ]
    )
    combos = _distinct_combos(table)
    method_names = {combo.method_name for combo in combos}
    assert method_names == {
        _FAIR_GCG_METHOD,
        _FAIR_FEATURE_DIFFMEAN_METHOD,
        _NON_GCG_METHOD,
        _ROOFLINE_FEATURE_DIFFMEAN_METHOD,
    }


def test_distinct_combos_keeps_fair_and_non_gcg_without_roofline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fair precomputed-bag and non-GCG combos pass through when no roofline is
    present (only roofline is unimplemented, so those must not be affected)."""
    monkeypatch.setattr(general_val, "logger", MagicMock())
    table = _dc_table(
        [
            _dc_row(_FAIR_GCG_METHOD),
            _dc_row(_FAIR_FEATURE_DIFFMEAN_METHOD),
            _dc_row(_NON_GCG_METHOD),
        ]
    )
    combos = _distinct_combos(table)
    method_names = {c.method_name for c in combos}
    assert method_names == {
        _FAIR_GCG_METHOD,
        _FAIR_FEATURE_DIFFMEAN_METHOD,
        _NON_GCG_METHOD,
    }
