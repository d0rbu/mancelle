from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pa_pq
import pytest
import torch as t
from safetensors.torch import save_file

import steering.eval.axbench_bpb_adv.aggregation as agg_module
import steering.eval.axbench_bpb_adv.intervention_build as intervention_build
import steering.eval.axbench_bpb_adv.scoring as scoring
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import (
    BpbScoreRow,
    Intervention,
    InterventionKind,
    ScoringInputs,
    SteeringMode,
)
from steering.eval.axbench_bpb_adv.aggregation import (
    _filter_rows_to_representative_coefficient,
    _select_representative_coefficient,
    _signed_margin,
    aggregate_bpb_adv_scores,
    compute_macro_aggregation,
    compute_per_candidate_aggregation,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.intervention_build import (
    _load_concept_direction,
    additive_counter_coefficient,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    GENERAL_VAL_BPB_SCHEMA,
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
    SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import (
    ModelBackedDualCompletionScorer,
    build_bpb_adv_score_table,
    candidate_row_to_scoring_inputs,
    run_model_backed_bpb_adv_score_stage,
)


class _TinyTokenizer:
    def get_vocab_size(self) -> int:
        return 256

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _candidate_rows(*, mode: str = "positive") -> list[dict[str, Any]]:
    source = [0, 97, 98]
    rich_output = [99, 100]
    avoid_output = [101, 102, 103]
    base = {
        "candidate_id": "c0",
        "attack_row_id": "r0",
        "core_prefix_id": "core0",
        "concept_rich_sample_id": "rich_sample",
        "concept_avoiding_sample_id": "avoid_sample",
        "concept_id": 1,
        "mode": mode,
        "method_name": "diffmean_encdec",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 0.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 0,
        "context_signal_mode": "fixed_context",
        "prefix_text": "ab",
        "prefix_hash": "hash",
        "source_prompt_tokens": source,
        "source_prompt_token_digest": token_sequence_digest(source),
        "attacked_prompt_text": "ab",
        "attacked_prompt_tokens": source,
        "attacked_prompt_token_digest": token_sequence_digest(source),
        "concept_rich_output_tokens": rich_output,
        "concept_avoiding_output_tokens": avoid_output,
        "concept_rich_output_digest": token_sequence_digest(rich_output),
        "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
        "editable_token_start": len(source),
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": None,
        "candidate_metadata_json": '{"candidate":"no_adversary_control"}',
    }
    attacked = [0, 97, 120]
    second = {
        **base,
        "candidate_id": "c1",
        "adversarial_strength": 1.0,
        "strength_index": 1,
        "attacked_prompt_text": "ax",
        "attacked_prompt_tokens": attacked,
        "attacked_prompt_token_digest": token_sequence_digest(attacked),
        "editable_token_start": 2,
        "editable_token_count": 1,
        "replacement_tokens": [120],
        "replacement_text": "x",
    }
    return [base, second]


def _candidate_table(*, mode: str = "positive") -> pa.Table:
    return pa.Table.from_pylist(_candidate_rows(mode=mode), schema=CANDIDATES_SCHEMA)


def _make_score_row(
    candidate_row: Mapping[str, Any],
    scoring_input: ScoringInputs,
    *,
    bpb: float,
    n_output_bytes: int = 2,
    truncated: bool = False,
) -> BpbScoreRow:
    return BpbScoreRow(
        method_name=str(candidate_row["method_name"]),
        concept_id=int(candidate_row["concept_id"]),
        mode=str(candidate_row["mode"]),
        content_class=scoring_input.content_class,
        target=str(candidate_row["target_model"]),
        coefficient_str=str(candidate_row["coefficient_str"]),
        sample_id=scoring_input.sample_id,
        bpb_intervened=bpb,
        n_output_tokens=scoring_input.n_output,
        n_output_bytes=n_output_bytes,
        truncated=truncated,
    )


def _dual_score_pair(
    candidate_row: Mapping[str, Any],
    rich_input: ScoringInputs,
    avoid_input: ScoringInputs,
) -> tuple[BpbScoreRow, BpbScoreRow]:
    # rich loss decreases with strength; avoid loss stays put, so the
    # positive-mode margin (avoid - rich) GROWS as strength rises only if
    # rich loss falls. We instead make rich loss INCREASE with strength so the
    # margin SHRINKS as the adversary degrades steering (the realistic shape).
    strength = float(candidate_row["adversarial_strength"])
    rich = _make_score_row(candidate_row, rich_input, bpb=1.0 + strength, n_output_bytes=2)
    avoid = _make_score_row(candidate_row, avoid_input, bpb=3.0, n_output_bytes=5)
    return rich, avoid


def test_candidate_row_to_scoring_inputs_shares_prompt_differs_only_in_output() -> None:
    candidate_row = _candidate_rows()[1]
    rich_input, avoid_input = candidate_row_to_scoring_inputs(
        candidate_row=candidate_row,
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len=32,
    )

    # Shared prompt / context.
    assert rich_input.prefix_tokens == avoid_input.prefix_tokens
    assert rich_input.prefix_tokens == candidate_row["attacked_prompt_tokens"]
    assert rich_input.prefix_text == avoid_input.prefix_text
    assert rich_input.prefix_hash == avoid_input.prefix_hash
    assert rich_input.prefix_context_segments == avoid_input.prefix_context_segments

    # Differ only in output side / identity.
    assert rich_input.content_class == "concept_rich"
    assert avoid_input.content_class == "concept_avoiding"
    assert rich_input.output_tokens == candidate_row["concept_rich_output_tokens"]
    assert avoid_input.output_tokens == candidate_row["concept_avoiding_output_tokens"]
    assert rich_input.sample_id == candidate_row["concept_rich_sample_id"]
    assert avoid_input.sample_id == candidate_row["concept_avoiding_sample_id"]


def test_scoring_positive_margin_is_avoid_minus_rich() -> None:
    scores = build_bpb_adv_score_table(
        candidates=_candidate_table(mode="positive"),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=_dual_score_pair,
    )

    assert scores.schema.equals(SCORE_SCHEMA)
    rows = scores.to_pylist()
    control = rows[0]
    attacked = rows[1]

    # positive margin = BPB(avoid) - BPB(rich). The SCORE table no longer carries
    # a write-only primary_margin_y column; recompute it from the two side losses
    # via the shared signed_class_margin helper.
    assert control["loss_concept_rich"] == 1.0
    assert control["loss_concept_avoiding"] == 3.0
    assert "primary_margin_y" not in control
    assert signed_class_margin(
        mode="positive",
        rich=control["loss_concept_rich"],
        avoid=control["loss_concept_avoiding"],
    ) == pytest.approx(2.0)
    # k=0 control row carries is_control True and a real margin.
    assert control["is_control"] is True

    # As adversarial strength rises, rich loss rises -> margin shrinks.
    assert attacked["loss_concept_rich"] == 2.0
    assert signed_class_margin(
        mode="positive",
        rich=attacked["loss_concept_rich"],
        avoid=attacked["loss_concept_avoiding"],
    ) == pytest.approx(1.0)
    assert attacked["is_control"] is False
    assert attacked["attacked_prompt_tokens"] == [0, 97, 120]


def test_scoring_negative_margin_is_rich_minus_avoid() -> None:
    scores = build_bpb_adv_score_table(
        candidates=_candidate_table(mode="negative"),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=_dual_score_pair,
    )

    rows = scores.to_pylist()
    control = rows[0]
    # negative margin = BPB(rich) - BPB(avoid) = 1.0 - 3.0 = -2.0 (sign flip).
    assert control["loss_concept_rich"] == 1.0
    assert control["loss_concept_avoiding"] == 3.0
    assert "primary_margin_y" not in control
    assert signed_class_margin(
        mode="negative",
        rich=control["loss_concept_rich"],
        avoid=control["loss_concept_avoiding"],
    ) == pytest.approx(-2.0)


def test_score_schema_drops_write_only_primary_margin_y() -> None:
    # F7: primary_margin_y had no production reader at the SCORE level; the
    # aggregation recomputes it from the two side losses. The column is dropped
    # from SCORE_SCHEMA (it is retained in the per_candidate / macro agg schemas,
    # which ARE read downstream).
    assert "primary_margin_y" not in SCORE_SCHEMA.names
    assert "loss_concept_rich" in SCORE_SCHEMA.names
    assert "loss_concept_avoiding" in SCORE_SCHEMA.names
    assert "primary_margin_y" in PER_CANDIDATE_AGG_SCHEMA.names
    assert "primary_margin_y" in MACRO_AGG_SCHEMA.names


def test_general_val_corpus_sits_directly_after_general_val_bpb() -> None:
    # Issue #635 (D2): the column POSITION is load-bearing — every fixture and
    # consumer validates these schemas with strict schema.equals, so pin
    # general_val_corpus immediately after general_val_bpb in all four aggregate
    # schemas, and LAST in GENERAL_VAL_BPB_SCHEMA (appended in schema v2).
    for schema in (
        PER_CANDIDATE_AGG_SCHEMA,
        MACRO_AGG_SCHEMA,
        ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
        ROUNDTRIP_MACRO_AGG_SCHEMA,
    ):
        names = schema.names
        assert names.index("general_val_corpus") == names.index("general_val_bpb") + 1
    assert GENERAL_VAL_BPB_SCHEMA.names[-1] == "general_val_corpus"


def test_scoring_does_not_emit_primary_margin_y_column() -> None:
    scores = build_bpb_adv_score_table(
        candidates=_candidate_table(mode="positive"),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=_dual_score_pair,
    )
    assert "primary_margin_y" not in scores.column_names
    # The margin is fully recomputable from the persisted side losses.
    row = scores.to_pylist()[0]
    assert signed_class_margin(
        mode="positive",
        rich=row["loss_concept_rich"],
        avoid=row["loss_concept_avoiding"],
    ) == pytest.approx(2.0)


def test_scoring_records_per_side_bytes_and_truncation_without_equality() -> None:
    def _uneven_pair(
        candidate_row: Mapping[str, Any],
        rich_input: ScoringInputs,
        avoid_input: ScoringInputs,
    ) -> tuple[BpbScoreRow, BpbScoreRow]:
        rich = _make_score_row(
            candidate_row, rich_input, bpb=1.0, n_output_bytes=2, truncated=False
        )
        avoid = _make_score_row(
            candidate_row, avoid_input, bpb=3.0, n_output_bytes=7, truncated=True
        )
        return rich, avoid

    scores = build_bpb_adv_score_table(
        candidates=_candidate_table(),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=_uneven_pair,
    )

    row = scores.to_pylist()[0]
    # Per-side bytes recorded; no cross-side equality assertion is enforced.
    assert row["n_rich_output_bytes"] == 2
    assert row["n_avoid_output_bytes"] == 7
    assert row["n_rich_output_tokens"] == 2
    assert row["n_avoid_output_tokens"] == 3
    # Candidate is truncated if EITHER side is truncated.
    assert row["rich_truncated"] is False
    assert row["avoid_truncated"] is True


def test_scoring_rejects_zero_byte_output_side() -> None:
    def _zero_byte_pair(
        candidate_row: Mapping[str, Any],
        rich_input: ScoringInputs,
        avoid_input: ScoringInputs,
    ) -> tuple[BpbScoreRow, BpbScoreRow]:
        rich = _make_score_row(candidate_row, rich_input, bpb=1.0, n_output_bytes=0)
        avoid = _make_score_row(candidate_row, avoid_input, bpb=3.0, n_output_bytes=5)
        return rich, avoid

    with pytest.raises(AxBenchBpbAdvError, match="bytes"):
        build_bpb_adv_score_table(
            candidates=_candidate_table(),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            token_bytes=t.ones(256, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 32},
            score_pair=_zero_byte_pair,
        )


def test_write_csv_delegates_to_atomic_write_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The per_candidate.csv / macro.csv writes route through the canonical
    atomic_write_text (tmp+fsync+replace) rather than a hand-rolled writer, so the
    durability guarantee matches the atomic parquets."""
    captured: dict[str, Any] = {}

    def _fake_atomic_write_text(text: str, final_path: Any, *, encoding: str = "utf-8") -> None:
        captured["text"] = text
        captured["path"] = Path(final_path)

    monkeypatch.setattr(agg_module, "atomic_write_text", _fake_atomic_write_text)
    target = tmp_path / "out.csv"
    agg_module._write_csv([{"a": 1, "b": 2}], target)

    assert captured["path"] == target
    assert "a,b" in captured["text"] and "1,2" in captured["text"]


def test_additive_counter_coefficient_sign_per_mode() -> None:
    # The counter is a STANDALONE coefficient on the specified counter-direction,
    # applied on top of the full baseline (no base coefficient folded in).
    # positive mode steers TOWARD the concept -> counter subtracts (-magnitude).
    assert additive_counter_coefficient(mode="positive", magnitude=1.5) == pytest.approx(-1.5)
    # negative mode steers AWAY -> counter adds (+magnitude).
    assert additive_counter_coefficient(mode="negative", magnitude=1.5) == pytest.approx(1.5)
    # magnitude 0 = pure-baseline control (no counter), both modes.
    assert additive_counter_coefficient(mode="positive", magnitude=0.0) == 0.0
    assert additive_counter_coefficient(mode="negative", magnitude=0.0) == 0.0


def test_additive_branch_composes_counter_direction_on_full_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The additive branch builds the baseline's NORMAL steered intervention and
    attaches the counter-direction add-on (specified family, signed magnitude),
    for ANY baseline -- decoder direction/lora AND encdec ours."""
    direction = t.ones(4)
    load_calls: list[dict[str, Any]] = []

    def _fake_load(**kwargs: Any) -> tuple[t.Tensor, int, dict[str, str]]:
        load_calls.append(kwargs)
        return (direction, 6, {})

    monkeypatch.setattr(intervention_build, "_load_concept_direction", _fake_load)

    def _make_scorer() -> ModelBackedDualCompletionScorer:
        scorer = ModelBackedDualCompletionScorer(
            target_bundles=cast(Any, {}),
            target_model_metas={},
            concepts_by_id={1: cast(Any, SimpleNamespace(concept_id=1))},
            source_paths=cast(Any, SimpleNamespace()),
            tokenizer=None,
            device=t.device("cpu"),
            diffmean_layer=6,
            inactive_sampler=None,
            apply_max_act_scale=False,
            counter_direction_family="diffmean",
        )
        return scorer

    def _row(target: str) -> dict[str, Any]:
        return {
            "target_model": target,
            "adversarial_method": "additive_direction",
            "adversarial_strength": 2.0,
        }

    # diffmean baseline (decoder): base steering preserved + counter on top.
    diffmean_scorer = _make_scorer()
    base_diffmean = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=t.zeros(4),
        direction_layer=6,
        coefficient=3.0,
    )
    diffmean_scorer._base_steered_intervention = cast(Any, lambda **_kw: base_diffmean)
    iv = diffmean_scorer._steered_intervention(
        method=cast(
            Any,
            SimpleNamespace(
                baseline="diffmean", mode=SteeringMode.POSITIVE, target_model="decoder_only"
            ),
        ),
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        coefficient=3.0,
        candidate_row=_row("decoder_only"),
    )
    assert iv.kind is InterventionKind.DIFFMEAN
    assert iv.direction_coefficient == 3.0  # full baseline steering preserved
    assert t.equal(cast(t.Tensor, iv.counter_direction), direction)
    assert iv.counter_direction_layer == 6
    assert iv.counter_direction_coefficient == pytest.approx(-2.0)  # positive -> -mag
    # The counter direction is loaded for the BASELINE's own model/target.
    assert load_calls[-1]["target"] == "decoder_only"

    # encdec ours baseline: no longer rejected; counter add-on attached, +mag (neg).
    ours_scorer = _make_scorer()
    base_ours = Intervention.ours(feature_conditions={1: 0})
    ours_scorer._base_steered_intervention = cast(Any, lambda **_kw: base_ours)
    iv_ours = ours_scorer._steered_intervention(
        method=cast(
            Any, SimpleNamespace(baseline="ours", mode=SteeringMode.NEGATIVE, target_model="encdec")
        ),
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        coefficient=None,
        candidate_row=_row("encdec"),
    )
    assert iv_ours.kind is InterventionKind.OURS
    assert iv_ours.feature_conditions == {1: 0}
    assert iv_ours.counter_direction_coefficient == pytest.approx(2.0)  # negative -> +mag
    # CRITICAL: the encdec baseline's counter must load the ENCDEC direction
    # (its own residual basis), NOT the decoder_only one.
    assert load_calls[-1]["target"] == "encdec"


def test_non_additive_branch_returns_plain_base_intervention() -> None:
    scorer = ModelBackedDualCompletionScorer(
        target_bundles=cast(Any, {}),
        target_model_metas={},
        concepts_by_id={1: cast(Any, SimpleNamespace(concept_id=1))},
        source_paths=cast(Any, SimpleNamespace()),
        tokenizer=None,
        device=t.device("cpu"),
        diffmean_layer=6,
        inactive_sampler=None,
        apply_max_act_scale=False,
    )
    base = Intervention.lora(adapter_path=Path("/tmp/a.pt"))
    scorer._base_steered_intervention = cast(Any, lambda **_kw: base)
    iv = scorer._steered_intervention(
        method=cast(
            Any,
            SimpleNamespace(
                baseline="lora", mode=SteeringMode.POSITIVE, target_model="decoder_only"
            ),
        ),
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        coefficient=None,
        candidate_row={
            "target_model": "decoder_only",
            "adversarial_method": "faster_gcg",
            "adversarial_strength": 0.0,
        },
    )
    assert iv is base  # no counter add-on for non-additive rows
    assert iv.counter_direction is None


def test_additive_control_intervention_uses_shared_counter_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    direction = t.ones(4)
    load_calls: list[dict[str, Any]] = []

    def _fake_load(**kwargs: Any) -> tuple[t.Tensor, int, dict[str, str]]:
        load_calls.append(kwargs)
        return (direction, 6, {})

    monkeypatch.setattr(intervention_build, "_load_concept_direction", _fake_load)
    scorer = ModelBackedDualCompletionScorer(
        target_bundles=cast(Any, {}),
        target_model_metas={},
        concepts_by_id={1: cast(Any, SimpleNamespace(concept_id=1))},
        source_paths=cast(Any, SimpleNamespace()),
        tokenizer=None,
        device=t.device("cpu"),
        diffmean_layer=6,
        inactive_sampler=None,
        apply_max_act_scale=False,
        counter_direction_family="linear_probe",
    )

    iv = scorer._control_intervention(
        method=cast(Any, SimpleNamespace(mode=SteeringMode.POSITIVE)),
        concept=cast(Any, SimpleNamespace(concept_id=1)),
        candidate_row={
            "target_model": "decoder_only",
            "adversarial_method": "additive_direction",
            "adversarial_strength": 2.0,
        },
    )

    assert iv.kind is InterventionKind.NONE
    assert t.equal(cast(t.Tensor, iv.counter_direction), direction)
    assert iv.counter_direction_layer == 6
    assert iv.counter_direction_coefficient == pytest.approx(-2.0)
    assert load_calls[-1]["family"] == "linear_probe"
    assert load_calls[-1]["target"] == "decoder_only"


def test_scoring_rejects_replacement_tokens_that_do_not_match_attacked_span() -> None:
    rows = _candidate_rows()
    rows[1] = {**rows[1], "replacement_tokens": [121]}

    with pytest.raises(AxBenchBpbAdvError, match="replacement_tokens"):
        build_bpb_adv_score_table(
            candidates=pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            token_bytes=t.ones(256, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 32},
            score_pair=_dual_score_pair,
        )


def test_scoring_rejects_candidate_changes_outside_editable_span() -> None:
    rows = _candidate_rows()
    attacked = [0, 88, 120]
    rows[1] = {
        **rows[1],
        "attacked_prompt_tokens": attacked,
        "attacked_prompt_token_digest": token_sequence_digest(attacked),
    }

    with pytest.raises(AxBenchBpbAdvError, match="outside the editable span"):
        build_bpb_adv_score_table(
            candidates=pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA),
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            token_bytes=t.ones(256, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 32},
            score_pair=_dual_score_pair,
        )


def test_dual_completion_scorer_scores_both_under_one_steered_intervention() -> None:
    captured: list[dict[str, Any]] = []

    def _fake_score_sample(
        *,
        inputs: ScoringInputs,
        target_bundle: Any,
        intervention: Any,
        device: Any,
        coefficient_str: str,
        method_name: str,
        lora: Any = None,
    ) -> BpbScoreRow:
        captured.append(
            {
                "content_class": inputs.content_class,
                "output_tokens": list(inputs.output_tokens),
                "intervention": intervention,
            }
        )
        return BpbScoreRow(
            method_name=method_name,
            concept_id=int(inputs.concept_id),
            mode=inputs.mode.value,
            content_class=inputs.content_class,
            target=target_bundle.target_name,
            coefficient_str=coefficient_str,
            sample_id=inputs.sample_id,
            bpb_intervened=1.0,
            n_output_tokens=inputs.n_output,
            n_output_bytes=2,
            truncated=False,
        )

    sentinel_intervention = Intervention.none()

    scorer = ModelBackedDualCompletionScorer(
        target_bundles=cast(Any, {"decoder_only": SimpleNamespace(target_name="decoder_only")}),
        target_model_metas={"decoder_only": object()},
        concepts_by_id={1: cast(Any, SimpleNamespace(concept_id=1))},
        source_paths=cast(Any, SimpleNamespace()),
        tokenizer=_TinyTokenizer(),
        device=t.device("cpu"),
        diffmean_layer=8,
        inactive_sampler=None,
        apply_max_act_scale=False,
        lora=None,
    )
    # Stub the steered-intervention builder + the scoring primitive.
    scorer._base_steered_intervention = cast(Any, lambda **_kwargs: sentinel_intervention)
    scoring_score_sample = scoring.score_sample
    try:
        scoring.score_sample = cast(Any, _fake_score_sample)
        candidate_row = _candidate_rows()[1]
        rich_input, avoid_input = candidate_row_to_scoring_inputs(
            candidate_row=candidate_row,
            context_segments_by_prefix_hash={},
            tokenizer=_TinyTokenizer(),
            token_bytes=t.ones(256, dtype=t.int64),
            max_seq_len=32,
        )
        rich_score, avoid_score = scorer(candidate_row, rich_input, avoid_input)
    finally:
        scoring.score_sample = scoring_score_sample

    assert rich_score.content_class == "concept_rich"
    assert avoid_score.content_class == "concept_avoiding"
    # Both completions scored under the SAME intervention instance.
    assert len(captured) == 2
    assert captured[0]["intervention"] is sentinel_intervention
    assert captured[1]["intervention"] is sentinel_intervention
    # Two DIFFERENT completions.
    assert captured[0]["output_tokens"] != captured[1]["output_tokens"]
    assert {c["content_class"] for c in captured} == {
        "concept_rich",
        "concept_avoiding",
    }


def test_direction_intervention_rejects_layer_metadata_mismatch(tmp_path: Path) -> None:
    direction_path = (
        tmp_path
        / "baselines"
        / "diffmean"
        / "decoder_only"
        / "positive"
        / "concept_0001.safetensors"
    )
    direction_path.parent.mkdir(parents=True)
    save_file(
        {"direction": t.ones(4)},
        str(direction_path),
        metadata={
            "concept_id": "1",
            "layer": "6",
            "mode": "positive",
            "target_model": "decoder_only",
        },
    )

    with pytest.raises(AxBenchBpbAdvError, match="layer mismatch"):
        _load_concept_direction(
            family="diffmean",
            mode=SteeringMode.POSITIVE.value,
            concept=cast(Any, SimpleNamespace(concept_id=1)),
            source_paths=SourcePaths(tmp_path),
            expected_layer=8,
        )


def test_score_stage_rejects_empty_candidates_before_loading_runtime() -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )

    with pytest.raises(AxBenchBpbAdvError, match="no candidates"):
        run_model_backed_bpb_adv_score_stage(
            cfg=cfg,
            paths=cast(Any, SimpleNamespace()),
            candidates=pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA),
        )


# ---------------------------------------------------------------------------
# Aggregation (Part 2.7): per-candidate + macro class-margin rollups.
# ---------------------------------------------------------------------------


def _score_row(
    *,
    concept_id: int = 1,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    coefficient_str: str = "1.0",
    adversarial_method: str = "faster_gcg",
    adversarial_strength: float = 0.0,
    strength_unit: str = "optimized_tokens",
    strength_index: int = 0,
    context_signal_mode: str = "fixed_context",
    core_prefix_id: str = "core0",
    loss_concept_rich: float | None = 1.0,
    loss_concept_avoiding: float | None = 3.0,
    is_control: bool = False,
    rich_truncated: bool = False,
    avoid_truncated: bool = False,
    n_rich_output_tokens: int = 2,
    n_avoid_output_tokens: int = 3,
    n_rich_output_bytes: int = 2,
    n_avoid_output_bytes: int = 5,
    candidate_suffix: str = "",
) -> dict[str, Any]:
    return {
        "candidate_id": f"cand_{core_prefix_id}_{strength_index}{candidate_suffix}",
        "core_prefix_id": core_prefix_id,
        "concept_id": concept_id,
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": adversarial_method,
        "adversarial_strength": adversarial_strength,
        "strength_unit": strength_unit,
        "strength_index": strength_index,
        "context_signal_mode": context_signal_mode,
        "source_prompt_tokens": [0, 97, 98],
        "source_prompt_token_digest": "d",
        "attacked_prompt_tokens": [0, 97, 98],
        "attacked_prompt_token_digest": "d",
        "editable_token_start": 3,
        "editable_token_count": 0,
        "replacement_tokens": [],
        "loss_concept_rich": loss_concept_rich,
        "loss_concept_avoiding": loss_concept_avoiding,
        "is_control": is_control,
        "n_rich_output_tokens": n_rich_output_tokens,
        "n_avoid_output_tokens": n_avoid_output_tokens,
        "n_rich_output_bytes": n_rich_output_bytes,
        "n_avoid_output_bytes": n_avoid_output_bytes,
        "rich_truncated": rich_truncated,
        "avoid_truncated": avoid_truncated,
    }


def _score_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)


def test_per_candidate_positive_margin_is_avoid_minus_rich() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
        ]
    )

    per_candidate = compute_per_candidate_aggregation(scores)

    assert per_candidate.schema.equals(PER_CANDIDATE_AGG_SCHEMA)
    row = per_candidate.to_pylist()[0]
    assert row["mean_loss_concept_rich"] == pytest.approx(1.0)
    assert row["mean_loss_concept_avoiding"] == pytest.approx(3.0)
    # positive primary margin = avoid - rich
    assert row["primary_margin_y"] == pytest.approx(2.0)
    # control group fills its own baseline margin.
    assert row["baseline_primary_margin"] == pytest.approx(2.0)


def test_per_candidate_negative_margin_is_sign_flipped() -> None:
    scores = _score_table(
        [
            _score_row(
                mode="negative",
                core_prefix_id="core_a",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
        ]
    )

    row = compute_per_candidate_aggregation(scores).to_pylist()[0]
    # negative primary margin = rich - avoid = 1.0 - 3.0 = -2.0
    assert row["primary_margin_y"] == pytest.approx(-2.0)


def test_per_candidate_averages_each_side_then_recomputes_margin() -> None:
    # Two core prefixes in the same concept/strength group; averaging EACH side
    # then taking the margin differs from averaging per-prefix margins only when
    # the operation order matters, so use distinct per-side values.
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                loss_concept_rich=1.0,
                loss_concept_avoiding=5.0,
                is_control=True,
            ),
            _score_row(
                core_prefix_id="core_b",
                loss_concept_rich=3.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
        ]
    )

    row = compute_per_candidate_aggregation(scores).to_pylist()[0]
    # mean rich = 2.0, mean avoid = 4.0 -> margin = 2.0
    assert row["mean_loss_concept_rich"] == pytest.approx(2.0)
    assert row["mean_loss_concept_avoiding"] == pytest.approx(4.0)
    assert row["primary_margin_y"] == pytest.approx(2.0)
    assert row["n_samples"] == 2


def _cartesian_pair_score_rows(
    *,
    mode: str,
    rich_losses: list[float],
    avoid_losses: list[float],
) -> list[dict[str, Any]]:
    """One score row per (rich_i, avoid_j) Cartesian pair, all in one group.

    Cartesian explosion (Gap B) keeps each row owning exactly one rich and one
    avoid completion. Every pair shares the same core prefix / method / strength
    so they land in ONE aggregation group, where each side is averaged
    separately and the margin re-derived via ``_signed_margin``.
    """

    rows: list[dict[str, Any]] = []
    for i, rich in enumerate(rich_losses):
        for j, avoid in enumerate(avoid_losses):
            rows.append(
                _score_row(
                    mode=mode,
                    core_prefix_id="core_a",
                    candidate_suffix=f"_p{i}x{j}",
                    loss_concept_rich=rich,
                    loss_concept_avoiding=avoid,
                    is_control=True,
                )
            )
    return rows


def test_cartesian_pairs_aggregate_via_signed_margin_of_side_means_positive() -> None:
    # 2 rich x 2 avoid = 4 pairs in ONE group. Each rich appears with every avoid.
    rich_losses = [1.0, 3.0]
    avoid_losses = [4.0, 8.0]
    scores = _score_table(
        _cartesian_pair_score_rows(
            mode="positive", rich_losses=rich_losses, avoid_losses=avoid_losses
        )
    )
    row = compute_per_candidate_aggregation(scores).to_pylist()[0]

    mean_rich = sum(rich_losses) / len(rich_losses)  # 2.0
    mean_avoid = sum(avoid_losses) / len(avoid_losses)  # 6.0
    assert row["n_samples"] == 4
    # Side means equal the per-side means (each rich appears M times, each avoid N).
    assert row["mean_loss_concept_rich"] == pytest.approx(mean_rich)
    assert row["mean_loss_concept_avoiding"] == pytest.approx(mean_avoid)
    # CRITICAL: margin is _signed_margin(side means), NOT the mean of per-pair margins.
    assert row["primary_margin_y"] == pytest.approx(
        _signed_margin(mode="positive", mean_rich=mean_rich, mean_avoid=mean_avoid)
    )
    assert row["primary_margin_y"] == pytest.approx(mean_avoid - mean_rich)


def test_cartesian_pairs_aggregate_via_signed_margin_of_side_means_negative() -> None:
    rich_losses = [1.0, 3.0]
    avoid_losses = [4.0, 8.0]
    scores = _score_table(
        _cartesian_pair_score_rows(
            mode="negative", rich_losses=rich_losses, avoid_losses=avoid_losses
        )
    )
    row = compute_per_candidate_aggregation(scores).to_pylist()[0]

    mean_rich = sum(rich_losses) / len(rich_losses)
    mean_avoid = sum(avoid_losses) / len(avoid_losses)
    # Negative mode: sign-flipped margin, routed through _signed_margin.
    assert row["primary_margin_y"] == pytest.approx(
        _signed_margin(mode="negative", mean_rich=mean_rich, mean_avoid=mean_avoid)
    )
    assert row["primary_margin_y"] == pytest.approx(mean_rich - mean_avoid)


def test_cartesian_pairs_margin_recomputes_signed_margin_from_side_means_uneven() -> None:
    # Uneven sides (3 rich x 2 avoid = 6 cartesian pairs). This asserts the
    # SIDE-MEAN RECOMPUTATION IDENTITY: the reported margin equals
    # _signed_margin(mean_rich, mean_avoid) computed from the raw side means.
    #
    # NOTE: this test cannot, by construction, distinguish the side-mean path
    # from the forbidden "mean of per-pair margins" path. The margin is LINEAR
    # (avoid - rich), so mean(per-pair avoid - rich) == mean_avoid - mean_rich
    # ALWAYS, even for uneven sides. Detecting the forbidden path is therefore
    # impossible numerically; it is guarded by the production code comment in
    # compute_per_candidate_aggregation (compute side means first, then ONE
    # _signed_margin call), not by this assertion.
    rich_losses = [1.0, 2.0, 9.0]
    avoid_losses = [4.0, 5.0]
    scores = _score_table(
        _cartesian_pair_score_rows(
            mode="positive", rich_losses=rich_losses, avoid_losses=avoid_losses
        )
    )
    row = compute_per_candidate_aggregation(scores).to_pylist()[0]
    mean_rich = sum(rich_losses) / len(rich_losses)
    mean_avoid = sum(avoid_losses) / len(avoid_losses)
    assert row["n_samples"] == 6
    assert row["mean_loss_concept_rich"] == pytest.approx(mean_rich)
    assert row["mean_loss_concept_avoiding"] == pytest.approx(mean_avoid)
    assert row["primary_margin_y"] == pytest.approx(
        _signed_margin(mode="positive", mean_rich=mean_rich, mean_avoid=mean_avoid)
    )


def test_per_candidate_refuses_group_missing_a_side() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                loss_concept_rich=1.0,
                loss_concept_avoiding=None,
            ),
        ]
    )

    with pytest.raises(AxBenchBpbAdvError, match="concept=1"):
        compute_per_candidate_aggregation(scores)


def test_per_candidate_does_not_collapse_across_strength() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                strength_index=0,
                adversarial_strength=0.0,
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                core_prefix_id="core_a",
                strength_index=1,
                adversarial_strength=1.0,
                loss_concept_rich=2.0,
                loss_concept_avoiding=3.0,
                is_control=False,
            ),
        ]
    )

    per_candidate = compute_per_candidate_aggregation(scores)
    rows = sorted(per_candidate.to_pylist(), key=lambda r: int(r["strength_index"]))
    # Two distinct strength rows survive -> strengths are NOT merged.
    assert [r["strength_index"] for r in rows] == [0, 1]
    assert [r["adversarial_strength"] for r in rows] == [0.0, 1.0]
    assert rows[0]["primary_margin_y"] == pytest.approx(2.0)
    assert rows[1]["primary_margin_y"] == pytest.approx(1.0)


def test_fixed_context_and_not_applicable_stay_separate_groups() -> None:
    # Two rows identical EXCEPT context_signal_mode (fixed_context vs
    # not_applicable -- the only two surviving values after #506 removed
    # recomputed_context). context_signal_mode is part of _full_key, so the two
    # rows must NOT merge into one per-candidate group.
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                context_signal_mode="fixed_context",
                candidate_suffix="_fixed",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                core_prefix_id="core_a",
                context_signal_mode="not_applicable",
                candidate_suffix="_na",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
        ]
    )

    per_candidate = compute_per_candidate_aggregation(scores)
    rows = per_candidate.to_pylist()
    # Two distinct per-candidate groups survive -> the contexts are NOT merged.
    assert len(rows) == 2
    modes = {row["context_signal_mode"] for row in rows}
    assert modes == {"fixed_context", "not_applicable"}


def test_per_candidate_baseline_primary_margin_from_control() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                strength_index=0,
                adversarial_strength=0.0,
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                core_prefix_id="core_a",
                strength_index=1,
                adversarial_strength=1.0,
                loss_concept_rich=2.0,
                loss_concept_avoiding=3.0,
                is_control=False,
            ),
        ]
    )

    rows = sorted(
        compute_per_candidate_aggregation(scores).to_pylist(),
        key=lambda r: int(r["strength_index"]),
    )
    # Both rows carry the control (k=0) margin as baseline.
    assert rows[0]["baseline_primary_margin"] == pytest.approx(2.0)
    assert rows[1]["baseline_primary_margin"] == pytest.approx(2.0)


def test_per_candidate_baseline_is_null_when_control_absent() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                strength_index=1,
                adversarial_strength=1.0,
                loss_concept_rich=2.0,
                loss_concept_avoiding=3.0,
                is_control=False,
            ),
        ]
    )

    row = compute_per_candidate_aggregation(scores).to_pylist()[0]
    assert row["baseline_primary_margin"] is None


def test_macro_carries_baseline_and_averages_across_concepts() -> None:
    scores = _score_table(
        [
            _score_row(
                concept_id=1,
                core_prefix_id="c1",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                concept_id=2,
                core_prefix_id="c2",
                loss_concept_rich=1.0,
                loss_concept_avoiding=5.0,
                is_control=True,
            ),
        ]
    )

    per_candidate = compute_per_candidate_aggregation(scores)
    macro = compute_macro_aggregation(per_candidate)

    assert macro.schema.equals(MACRO_AGG_SCHEMA)
    row = macro.to_pylist()[0]
    assert row["n_concepts"] == 2
    # concept-1 margin = 2.0, concept-2 margin = 4.0 -> macro = 3.0
    assert row["primary_margin_y"] == pytest.approx(3.0)
    # baseline carried through (both concepts control margin == their margin).
    assert row["baseline_primary_margin"] == pytest.approx(3.0)


# --- Issue #635 (D2): macro rollup carries a single general-val corpus signature
# per group, or raises when the group's rows disagree / lack one -------------


def _per_candidate_row(
    *,
    concept_id: int,
    general_val_bpb: float | None = None,
    general_val_corpus: str | None = None,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 2.0,
        "primary_margin_y": 1.0,
        "baseline_primary_margin": 1.0,
        "general_val_bpb": general_val_bpb,
        "general_val_corpus": general_val_corpus,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 5,
        "n_avoid_output_tokens": 5,
        "n_rich_output_bytes": 20,
        "n_avoid_output_bytes": 20,
        "is_control": True,
    }


def test_macro_carries_single_corpus_and_raises_on_mix() -> None:
    a = _per_candidate_row(concept_id=1, general_val_bpb=3.0, general_val_corpus="nd=2;nb=10;h=aa")
    b = _per_candidate_row(concept_id=2, general_val_bpb=4.0, general_val_corpus="nd=2;nb=10;h=aa")
    macro = compute_macro_aggregation(pa.Table.from_pylist([a, b], schema=PER_CANDIDATE_AGG_SCHEMA))
    assert macro.to_pylist()[0]["general_val_corpus"] == "nd=2;nb=10;h=aa"

    c = _per_candidate_row(concept_id=2, general_val_bpb=4.0, general_val_corpus="nd=9;nb=99;h=bb")
    with pytest.raises(AxBenchBpbAdvError, match="mixes general-val corpus signatures"):
        compute_macro_aggregation(pa.Table.from_pylist([a, c], schema=PER_CANDIDATE_AGG_SCHEMA))


def test_macro_rejects_value_without_corpus() -> None:
    row = _per_candidate_row(concept_id=1, general_val_bpb=3.0, general_val_corpus=None)
    with pytest.raises(AxBenchBpbAdvError, match="general_val_bpb without a corpus signature"):
        compute_macro_aggregation(pa.Table.from_pylist([row], schema=PER_CANDIDATE_AGG_SCHEMA))

    # An EMPTY-STRING corpus is just as unusable as a null one: the macro check
    # mirrors the joins' falsy (`not corpus`) predicate, per the schema comment's
    # "null iff general_val_bpb is null" invariant (issue #635 fail-loud).
    empty = _per_candidate_row(concept_id=1, general_val_bpb=3.0, general_val_corpus="")
    with pytest.raises(AxBenchBpbAdvError, match="general_val_bpb without a corpus signature"):
        compute_macro_aggregation(pa.Table.from_pylist([empty], schema=PER_CANDIDATE_AGG_SCHEMA))


def test_macro_empty_corpus_on_null_value_is_not_a_distinct_signature() -> None:
    # issue #635 review [4]: the COLLECTION side must also treat "" as missing
    # (falsy), so a null-value/empty-corpus row does not enter the corpora set and
    # spuriously trip the >1 mixing raise against a real-corpus row in the group.
    null_empty = _per_candidate_row(concept_id=1, general_val_bpb=None, general_val_corpus="")
    real = _per_candidate_row(concept_id=2, general_val_bpb=3.0, general_val_corpus="nd=2;nb=10;h=aa")
    macro = compute_macro_aggregation(
        pa.Table.from_pylist([null_empty, real], schema=PER_CANDIDATE_AGG_SCHEMA)
    )
    assert macro.to_pylist()[0]["general_val_corpus"] == "nd=2;nb=10;h=aa"


# --- F6: is_control flag propagated through both agg schemas, never re-derived
# from adversarial_strength -----------------------------------------------------


def _representative_key() -> tuple[Any, ...]:
    return (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )


def test_per_candidate_carries_is_control_flag_for_control_group() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
        ]
    )

    per_candidate = compute_per_candidate_aggregation(scores)
    assert per_candidate.schema.equals(PER_CANDIDATE_AGG_SCHEMA)
    row = per_candidate.to_pylist()[0]
    # per_candidate is_control == all(score is_control) -> True for a control group.
    assert row["is_control"] is True


def test_per_candidate_is_control_false_for_non_control_group() -> None:
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                strength_index=1,
                adversarial_strength=1.0,
                loss_concept_rich=2.0,
                loss_concept_avoiding=3.0,
                is_control=False,
            ),
        ]
    )

    row = compute_per_candidate_aggregation(scores).to_pylist()[0]
    assert row["is_control"] is False


def test_per_candidate_is_control_false_for_mixed_group() -> None:
    # A group containing one control and one non-control score row is NOT a
    # control group (all(...) over the score-level flag is False).
    scores = _score_table(
        [
            _score_row(
                core_prefix_id="core_a",
                candidate_suffix="_ctrl",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                core_prefix_id="core_a",
                candidate_suffix="_adv",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=False,
            ),
        ]
    )

    row = compute_per_candidate_aggregation(scores).to_pylist()[0]
    assert row["is_control"] is False


def test_macro_carries_is_control_as_all_of_per_candidate() -> None:
    scores = _score_table(
        [
            _score_row(
                concept_id=1,
                core_prefix_id="c1",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                concept_id=2,
                core_prefix_id="c2",
                loss_concept_rich=1.0,
                loss_concept_avoiding=5.0,
                is_control=True,
            ),
        ]
    )

    macro = compute_macro_aggregation(compute_per_candidate_aggregation(scores))
    assert macro.schema.equals(MACRO_AGG_SCHEMA)
    row = macro.to_pylist()[0]
    # macro is_control == all(per_candidate is_control) -> True here.
    assert row["is_control"] is True


def test_macro_is_control_false_when_any_concept_non_control() -> None:
    scores = _score_table(
        [
            _score_row(
                concept_id=1,
                core_prefix_id="c1",
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,
                is_control=True,
            ),
            _score_row(
                concept_id=2,
                core_prefix_id="c2",
                loss_concept_rich=1.0,
                loss_concept_avoiding=5.0,
                is_control=False,
            ),
        ]
    )

    macro = compute_macro_aggregation(compute_per_candidate_aggregation(scores))
    row = macro.to_pylist()[0]
    assert row["is_control"] is False


def test_strength_zero_mutated_prompt_is_not_treated_as_control() -> None:
    """The divergent class: a strength-0 row whose prompt was MUTATED carries
    ``is_control=False`` from scoring. Aggregation must PRESERVE that flag and
    NOT re-derive control status from ``adversarial_strength == 0``.

    Two rows in the SAME representative family at different coefficients:
    * coeff "1.0": a genuine k=0 control (is_control=True, strength 0).
    * coeff "2.0": a strength-0 surrogate whose prompt is mutated
      (is_control=False, strength 0) with a SPURIOUSLY higher baseline margin.

    If ``_is_control_row`` re-derived control from strength==0, the "2.0" family
    would be selectable as a control and could win representative selection. The
    fix reads only the persisted flag, so only the "1.0" family is a control.
    """

    scores = _score_table(
        [
            # Genuine control at coeff "1.0": strength 0, prompt unmodified.
            _score_row(
                coefficient_str="1.0",
                core_prefix_id="core_a",
                strength_index=0,
                adversarial_strength=0.0,
                loss_concept_rich=1.0,
                loss_concept_avoiding=3.0,  # margin 2.0
                is_control=True,
            ),
            # Strength-0 but MUTATED prompt at coeff "2.0": is_control False, and
            # a much higher margin (would win argmax IF mis-classed as control).
            _score_row(
                coefficient_str="2.0",
                core_prefix_id="core_a",
                strength_index=0,
                adversarial_strength=0.0,
                loss_concept_rich=1.0,
                loss_concept_avoiding=99.0,  # margin 98.0
                is_control=False,
            ),
        ]
    )

    per_candidate = compute_per_candidate_aggregation(scores)
    macro = compute_macro_aggregation(per_candidate)
    by_coeff = {row["coefficient_str"]: row for row in macro.to_pylist()}

    # (a) The strength-0-mutated config keeps is_control == False (flag preserved,
    # NOT re-derived from strength).
    assert by_coeff["2.0"]["is_control"] is False
    assert by_coeff["1.0"]["is_control"] is True

    # (b) Representative selection treats ONLY the is_control=True coefficient as a
    # control: despite "2.0" having a far larger margin, it is not a control row,
    # so "1.0" (the genuine control) is selected.
    macro_rows = macro.to_pylist()
    feature_diffmean_target = {
        **macro_rows[0],
        "method_name": "feature_diffmean_encdec",
        "target_model": "encdec",
        "coefficient_str": "None",
        "baseline_primary_margin": 2.0,
        "primary_margin_y": 2.0,
    }
    baseline_rows = [*macro_rows, feature_diffmean_target]
    selected = _select_representative_coefficient(
        macro_rows,
        baseline_rows=baseline_rows,
    )
    assert selected[_representative_key()] == "1.0"

    # And the rep filter keeps only the "1.0" family's rows.
    kept = _filter_rows_to_representative_coefficient(
        macro_rows,
        baseline_rows=baseline_rows,
    )
    assert {row["coefficient_str"] for row in kept} == {"1.0"}


# --- F15: atomic aggregate parquet writes -----------------------------------


def _written_scores_path(tmp_path: Path) -> Path:
    """Write a minimal valid SCORE_SCHEMA table to disk and return its path."""
    scores = _score_table([_score_row(is_control=True)])
    scores_path = tmp_path / "scores.parquet"
    pa_pq.write_table(scores, scores_path)
    return scores_path


def _general_val_table(*, bpb: float) -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "target_model": "decoder_only",
                "method_name": "diffmean_decoder_only",
                "mode": "positive",
                "concept_id": 1,
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": 0.0,
                "general_val_bpb": bpb,
                "n_documents": 2,
                "n_val_bytes": 32,
                "intervention_signature": "sig",
                "general_val_corpus": "nd=2;nb=32;h=0000000000000000",
            }
        ],
        schema=GENERAL_VAL_BPB_SCHEMA,
    )


def test_aggregate_uses_validated_scores_after_path_is_replaced(tmp_path: Path) -> None:
    scores_path = _written_scores_path(tmp_path)
    validated_scores = pa_pq.read_table(scores_path)
    assert validated_scores.schema.equals(SCORE_SCHEMA)

    mutated_scores = _score_table(
        [
            _score_row(
                loss_concept_rich=1.0,
                loss_concept_avoiding=9.0,
                is_control=True,
            )
        ]
    )
    pa_pq.write_table(mutated_scores, scores_path)

    macro = aggregate_bpb_adv_scores(
        scores_path,
        tmp_path / "out",
        scores_table=validated_scores,
    )

    assert macro.to_pylist()[0]["primary_margin_y"] == pytest.approx(2.0), (
        "aggregation must consume the already-validated score table, not "
        "replacement content read from its path"
    )


def test_aggregate_uses_validated_general_val_after_path_is_replaced(
    tmp_path: Path,
) -> None:
    scores_path = _written_scores_path(tmp_path)
    general_val_path = tmp_path / "general_val_bpb.parquet"
    pa_pq.write_table(_general_val_table(bpb=2.25), general_val_path)
    validated_general_val = pa_pq.read_table(general_val_path)
    assert validated_general_val.schema.equals(GENERAL_VAL_BPB_SCHEMA)

    pa_pq.write_table(_general_val_table(bpb=9.75), general_val_path)

    macro = aggregate_bpb_adv_scores(
        scores_path,
        tmp_path / "out",
        general_val_path=general_val_path,
        general_val_table=validated_general_val,
    )

    assert macro.to_pylist()[0]["general_val_bpb"] == pytest.approx(2.25), (
        "aggregation must consume the already-validated general-val table, not "
        "replacement content read from its path"
    )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            pa.Table.from_pydict({"wrong": [1]}),
            "schema mismatch in .*scores.parquet: expected SCORE_SCHEMA",
        ),
        (
            pa.Table.from_pylist([], schema=SCORE_SCHEMA),
            "no axbench-bpb-adv scores in .*scores.parquet",
        ),
    ],
)
def test_aggregate_none_scores_table_preserves_legacy_validation_messages(
    tmp_path: Path,
    table: pa.Table,
    message: str,
) -> None:
    scores_path = tmp_path / "scores.parquet"
    pa_pq.write_table(table, scores_path)

    with pytest.raises(AxBenchBpbAdvError, match=message):
        aggregate_bpb_adv_scores(
            scores_path,
            tmp_path / "out",
            scores_table=None,
        )


@pytest.mark.parametrize(
    ("table", "message"),
    [
        (
            pa.Table.from_pydict({"wrong": [1]}),
            "schema mismatch in .*scores.parquet: expected SCORE_SCHEMA",
        ),
        (
            pa.Table.from_pylist([], schema=SCORE_SCHEMA),
            "no axbench-bpb-adv scores in .*scores.parquet",
        ),
    ],
)
def test_aggregate_supplied_scores_table_keeps_schema_and_empty_validation(
    tmp_path: Path,
    table: pa.Table,
    message: str,
) -> None:
    scores_path = tmp_path / "scores.parquet"

    with pytest.raises(AxBenchBpbAdvError, match=message):
        aggregate_bpb_adv_scores(
            scores_path,
            tmp_path / "out",
            scores_table=table,
        )


def test_aggregate_writes_route_through_atomic_write_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both aggregate parquets go through the canonical atomic_write_parquet
    (tmp+fsync+replace) -- the durable, reuse-of-canonical-helper guarantee F15 is
    about -- so a crash mid-write can never land a torn table at the canonical
    path. (The crash-atomicity itself is owned and tested by nanochat.atomic_io.)"""
    scores_path = _written_scores_path(tmp_path)
    out_dir = tmp_path / "out"
    routed: list[str] = []
    real_atomic_write_parquet = agg_module.atomic_write_parquet

    def _spy(table: pa.Table, final_path: Any) -> None:
        routed.append(Path(final_path).name)
        real_atomic_write_parquet(table, final_path)

    monkeypatch.setattr(agg_module, "atomic_write_parquet", _spy)
    aggregate_bpb_adv_scores(scores_path, out_dir)

    assert set(routed) == {"per_candidate.parquet", "macro.parquet"}
    assert (out_dir / "per_candidate.parquet").exists()
    assert (out_dir / "macro.parquet").exists()


def test_aggregate_writes_complete_readable_parquets(tmp_path: Path) -> None:
    """Happy path: both aggregate parquets are complete, readable, schema-valid,
    and no stray `.tmp` files remain."""
    scores_path = _written_scores_path(tmp_path)
    out_dir = tmp_path / "out"

    aggregate_bpb_adv_scores(scores_path, out_dir)

    per_candidate_path = out_dir / "per_candidate.parquet"
    macro_path = out_dir / "macro.parquet"
    assert per_candidate_path.exists()
    assert macro_path.exists()

    per_candidate = pa_pq.read_table(per_candidate_path)
    macro = pa_pq.read_table(macro_path)
    assert per_candidate.schema.equals(PER_CANDIDATE_AGG_SCHEMA)
    assert macro.schema.equals(MACRO_AGG_SCHEMA)
    assert per_candidate.num_rows >= 1
    assert macro.num_rows >= 1

    # No leftover temp files after a clean run.
    assert list(out_dir.glob("*.tmp")) == []
