"""End-to-end CLI stage-chain test for the class-margin bpb-adv eval.

Runs the real CLI stage dispatch VALIDATE_SOURCE -> ... -> PLOT on a tiny
both-classes-per-prefix source fixture. The two model-heavy stages (candidate
build, score) are driven through the real artifact-producing code paths with a
tiny tokenizer / embedding and a deterministic dual-completion stub scorer, so
the test asserts the NEW schemas at every artifact and a finite, signed
``primary_margin_y`` end to end.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb.types import BpbScoreRow, ScoringInputs, SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackWorkSpec
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    Stage,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import (
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table
from tests.axbench_bpb_adv.source_run_builders import (
    write_scalar_direction_baselines,
    write_seed_contract,
)
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)

_VOCAB = 128

_CLASS_MARGIN_OUTPUT_INVENTORY = {
    "aggregates/aggregation_report.yaml",
    "aggregates/macro.csv",
    "aggregates/macro.parquet",
    "aggregates/per_candidate.csv",
    "aggregates/per_candidate.parquet",
    "attack_rows/attack_rows.parquet",
    "candidates/candidates.parquet",
    "evaluation_population.json",
    "plots/negative/baseline_primary_margin_vs_strength.png",
    "plots/negative/by_coefficient/coeff_1/steerability_vs_baseline.png",
    "plots/negative/by_coefficient/coeff_none/steerability_vs_baseline.png",
    "plots/negative/by_coefficient/plot_index.csv",
    "plots/negative/mean_loss_concept_avoiding_vs_strength.png",
    "plots/negative/mean_loss_concept_rich_vs_strength.png",
    "plots/negative/primary_margin_vs_strength.png",
    "plots/negative/steerability_vs_baseline.png",
    "plots/plot_index.csv",
    "plots/plot_series_legend.csv",
    "plots/positive/baseline_primary_margin_vs_strength.png",
    "plots/positive/by_coefficient/coeff_1/steerability_vs_baseline.png",
    "plots/positive/by_coefficient/coeff_none/steerability_vs_baseline.png",
    "plots/positive/by_coefficient/plot_index.csv",
    "plots/positive/mean_loss_concept_avoiding_vs_strength.png",
    "plots/positive/mean_loss_concept_rich_vs_strength.png",
    "plots/positive/primary_margin_vs_strength.png",
    "plots/positive/steerability_vs_baseline.png",
    "scores.parquet",
    "source_contract.json",
    "universal_triggers.parquet",
}

_ROUNDTRIP_OUTPUT_INVENTORY = {
    "attack_rows/attack_rows.parquet",
    "candidates/candidates.parquet",
    "evaluation_population.json",
    "roundtrip_aggregates/aggregation_report.yaml",
    "roundtrip_aggregates/macro.csv",
    "roundtrip_aggregates/macro.parquet",
    "roundtrip_aggregates/per_candidate.csv",
    "roundtrip_aggregates/per_candidate.parquet",
    "roundtrip_plots/negative/control_primary_roundtrip_y_vs_strength.png",
    "roundtrip_plots/negative/primary_roundtrip_y_vs_strength.png",
    "roundtrip_plots/negative/roundtrip_percentile_mean_vs_strength.png",
    "roundtrip_plots/negative/roundtrip_post_act_mean_vs_strength.png",
    "roundtrip_plots/negative/roundtrip_pre_act_mean_vs_strength.png",
    "roundtrip_plots/negative/roundtrip_steerability_vs_baseline.png",
    "roundtrip_plots/plot_index.csv",
    "roundtrip_plots/plot_series_legend.csv",
    "roundtrip_plots/positive/control_primary_roundtrip_y_vs_strength.png",
    "roundtrip_plots/positive/primary_roundtrip_y_vs_strength.png",
    "roundtrip_plots/positive/roundtrip_percentile_mean_vs_strength.png",
    "roundtrip_plots/positive/roundtrip_post_act_mean_vs_strength.png",
    "roundtrip_plots/positive/roundtrip_pre_act_mean_vs_strength.png",
    "roundtrip_plots/positive/roundtrip_steerability_vs_baseline.png",
    "roundtrip_scores.parquet",
    "source_contract.json",
    "universal_triggers.parquet",
}


def _output_inventory(root: Path) -> set[str]:
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return _VOCAB

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % _VOCAB for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=None,
        n_sae_features=8,
        negative_selection=None,
        feature_selection=None,
        feature_mask=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=6,
        segment_size=256,
    )


_SYNTH_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("content_class", pa.string()),
        ("prefix_text", pa.string()),
        ("output_text", pa.string()),
        ("qc_accepted", pa.bool_()),
        ("prefix_source_shard", pa.int32()),
        ("prefix_source_doc", pa.int32()),
    ]
)


def _both_class_rows(prefix: str, *, rich: str, avoid: str) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": 1,
            "content_class": "concept_rich",
            "prefix_text": prefix,
            "output_text": rich,
            "qc_accepted": True,
            "prefix_source_shard": 0,
            "prefix_source_doc": 0,
        },
        {
            "concept_id": 1,
            "content_class": "concept_avoiding",
            "prefix_text": prefix,
            "output_text": avoid,
            "qc_accepted": True,
            "prefix_source_shard": 0,
            "prefix_source_doc": 0,
        },
    ]


def _write_source_run(root: Path) -> None:
    root.mkdir(parents=True)
    concepts_payload = json.dumps(
        [{"concept_id": 1, "description": "concept"}]
    )
    (root / "concepts.json").write_text(concepts_payload, encoding="utf-8")
    (root / "concepts_stage1c.json").write_text(
        concepts_payload,
        encoding="utf-8",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                *_both_class_rows("abc", rich="rich", avoid="avoid"),
                *_both_class_rows("abd", rich="rich2", avoid="avoid2"),
            ],
            schema=schema_with_stage1_continuation_contract(
                schema_with_positive_qc_contract(
                    _SYNTH_SCHEMA,
                    accept_threshold=0.3,
                ),
                role="stage1b_positive",
            ),
        ),
        root / "synth_training_positive.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                *_both_class_rows("xyz", rich="neg-rich", avoid="neg-avoid"),
                *_both_class_rows("xyy", rich="neg-rich2", avoid="neg-avoid2"),
            ],
            schema=schema_with_stage1_continuation_contract(
                _SYNTH_SCHEMA,
                role="stage1c_negative",
            ),
        ),
        root / "synth_training_negative.parquet",
    )
    for name in (
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
    ):
        (root / name).write_bytes(b"x")
    write_seed_contract(root)
    write_scalar_direction_baselines(root / "baselines")


def _dual_score_pair(
    candidate_row: Mapping[str, Any],
    rich_input: ScoringInputs,
    avoid_input: ScoringInputs,
) -> tuple[BpbScoreRow, BpbScoreRow]:
    # Same prompt, one intervention, two distinct completions.
    assert rich_input.prefix_tokens == avoid_input.prefix_tokens
    strength = float(candidate_row["adversarial_strength"])

    def _row(scoring_input: ScoringInputs, *, bpb: float, n_bytes: int) -> BpbScoreRow:
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
            n_output_bytes=n_bytes,
            truncated=False,
        )

    # rich loss rises with adversarial strength -> positive-mode margin shrinks.
    rich = _row(rich_input, bpb=1.0 + strength, n_bytes=2)
    avoid = _row(avoid_input, bpb=3.0, n_bytes=4)
    return rich, avoid


def _roundtrip_scores_from_bpb_stub(
    candidates: pa.Table,
    *,
    tokenizer: _TinyTokenizer,
) -> tuple[pa.Table, pa.Table]:
    bpb_scores = build_bpb_adv_score_table(
        candidates=candidates,
        context_segments_by_prefix_hash={},
        tokenizer=tokenizer,
        token_bytes=t.ones(_VOCAB, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=_dual_score_pair,
    )
    bpb_by_candidate = {str(row["candidate_id"]): row for row in bpb_scores.to_pylist()}
    roundtrip_rows: list[dict[str, Any]] = []
    for candidate in candidates.to_pylist():
        candidate_id = str(candidate["candidate_id"])
        bpb_row = bpb_by_candidate[candidate_id]
        strength = float(candidate["adversarial_strength"])
        strength_index = int(candidate["strength_index"])
        mode = str(candidate["mode"])
        percentile = 65.0 - 5.0 * strength if mode == "positive" else 35.0 + 5.0 * strength
        control_percentile = 55.0 if mode == "positive" else 45.0
        roundtrip_rows.append(
            {
                "candidate_id": candidate_id,
                "core_prefix_id": str(candidate["core_prefix_id"]),
                "concept_id": int(candidate["concept_id"]),
                "mode": mode,
                "method_name": str(candidate["method_name"]),
                "target_model": str(candidate["target_model"]),
                "coefficient_str": str(candidate["coefficient_str"]),
                "adversarial_method": str(candidate["adversarial_method"]),
                "adversarial_strength": strength,
                "strength_unit": str(candidate["strength_unit"]),
                "strength_index": strength_index,
                "context_signal_mode": str(candidate["context_signal_mode"]),
                "source_prompt_tokens": list(candidate["source_prompt_tokens"]),
                "source_prompt_token_digest": str(candidate["source_prompt_token_digest"]),
                "attacked_prompt_tokens": list(candidate["attacked_prompt_tokens"]),
                "attacked_prompt_token_digest": str(candidate["attacked_prompt_token_digest"]),
                "roundtrip_prompt_tokens": list(candidate["attacked_prompt_tokens"]),
                "roundtrip_prompt_token_digest": str(candidate["attacked_prompt_token_digest"]),
                "roundtrip_prompt_text": str(candidate["attacked_prompt_text"]),
                "control_roundtrip_prompt_tokens": list(candidate["attacked_prompt_tokens"]),
                "control_roundtrip_prompt_token_digest": str(
                    candidate["attacked_prompt_token_digest"]
                ),
                "control_roundtrip_prompt_text": str(candidate["attacked_prompt_text"]),
                "editable_token_start": int(candidate["editable_token_start"]),
                "editable_token_count": int(candidate["editable_token_count"]),
                "replacement_tokens": list(candidate["replacement_tokens"]),
                "loss_concept_rich": float(bpb_row["loss_concept_rich"]),
                "loss_concept_avoiding": float(bpb_row["loss_concept_avoiding"]),
                "sample_index": 0,
                "reference_segment_token_count": 3,
                "generation_seed": 1000 + strength_index,
                "generated_text": f"roundtrip generated {candidate_id}",
                "n_generated_tokens": 3,
                "stop_reason": "max_tokens",
                "control_generation_seed": 2000 + strength_index,
                "control_generated_text": f"roundtrip control {candidate_id}",
                "control_n_generated_tokens": 2,
                "control_stop_reason": "eos",
                "is_control": strength_index == 0,
                "target_feature_id": 42,
                "n_roundtrip_features": 1,
                "roundtrip_pre_act": percentile / 10.0,
                "roundtrip_post_act": percentile / 20.0,
                "roundtrip_percentile": percentile,
                "control_n_roundtrip_features": 1,
                "control_roundtrip_pre_act": control_percentile / 10.0,
                "control_roundtrip_post_act": control_percentile / 20.0,
                "control_roundtrip_percentile": control_percentile,
            }
        )
    return pa.Table.from_pylist(roundtrip_rows, schema=ROUNDTRIP_SCORE_SCHEMA), bpb_scores


def test_cli_stage_chain_validate_through_plot(tmp_path: Path, monkeypatch) -> None:
    source_root = tmp_path / "source"
    _write_source_run(source_root)
    output_root = tmp_path / "out"

    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_root),
        output_root=str(output_root),
        run_tag="run",
        device="cpu",
        stage=Stage.ALL.value,
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0, 1),
                max_steps=0,
            )
        ],
        # score_general_val isn't mocked in this chain; general_val_bpb now
        # defaults to enabled=True (needs a real model), so keep it off.
        general_val_bpb=GeneralValBpbConfig(enabled=False),
        scatter_baseline_x="baseline_loss",
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    tokenizer = _TinyTokenizer()
    expected_bpb_losses: dict[str, tuple[float, float]] = {}

    # Tiny stand-ins for the model-heavy runtime pieces. Everything else (source
    # validation, attack-row pairing/schema, aggregation, plotting, dispatch)
    # runs for real.
    monkeypatch.setattr(
        bpb_adv_cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg: ({"decoder_only": _meta()}, tokenizer),
    )

    def _fake_work_specs(**_kwargs: Any) -> list[BpbAttackWorkSpec]:
        return [
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
            BpbAttackWorkSpec(
                method_name="neg_diffmean_decoder_only",
                mode=SteeringMode.NEGATIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
        ]

    monkeypatch.setattr(bpb_adv_cli, "work_specs_from_method_registry", _fake_work_specs)

    monkeypatch.setattr(
        bpb_adv_cli,
        "_stage_prepare_context_cache",
        lambda *_args, **_kwargs: None,
    )

    def _fake_candidate_stage(*, cfg: AxBenchBpbAdvConfig, paths: AxBenchBpbAdvPaths) -> int:
        return run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=tokenizer,
            embedding_weight=t.zeros(_VOCAB, 1),
            test_objective=zero_objective,
            test_gradient_provider_batch=zero_gradient_batch,
        )

    monkeypatch.setattr(bpb_adv_cli, "run_model_backed_bpb_candidate_stage", _fake_candidate_stage)

    def _fake_score_stage(
        *,
        cfg: AxBenchBpbAdvConfig,
        paths: AxBenchBpbAdvPaths,
        candidates: pa.Table,
        stats: dict[str, int] | None = None,
    ) -> pa.Table:
        scores = build_bpb_adv_score_table(
            candidates=candidates,
            context_segments_by_prefix_hash={},
            tokenizer=tokenizer,
            token_bytes=t.ones(_VOCAB, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 32},
            score_pair=_dual_score_pair,
        )
        rows = scores.to_pylist()
        for mode, method_name in (
            ("positive", "feature_diffmean_encdec"),
            ("negative", "neg_feature_diffmean_encdec"),
        ):
            base = next(row for row in rows if row["mode"] == mode and row["is_control"])
            candidate_id = f"feature-diffmean-target-{mode}"
            expected_bpb_losses[candidate_id] = (
                float(base["loss_concept_rich"]),
                float(base["loss_concept_avoiding"]),
            )
            rows.append(
                {
                    **base,
                    "candidate_id": candidate_id,
                    "method_name": method_name,
                    "target_model": "encdec",
                    "coefficient_str": "None",
                }
            )
        return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)

    monkeypatch.setattr(bpb_adv_cli, "run_model_backed_bpb_adv_score_stage", _fake_score_stage)

    assert (
        bpb_adv_cli._run_resolved(cfg, ddp_rank=0, ddp_world_size=1)
        is cfg
    )
    assert _output_inventory(paths.root) == _CLASS_MARGIN_OUTPUT_INVENTORY

    # Every artifact matches the NEW schemas.
    attack_rows = pq.read_table(paths.attack_rows_path())
    assert attack_rows.schema.equals(ATTACK_ROWS_SCHEMA)
    candidates = pq.read_table(paths.candidates_path())
    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    scores = pq.read_table(paths.scores_path())
    assert scores.schema.equals(SCORE_SCHEMA)
    # The clustered dispatch wiring is a strict no-op for legacy runs: it must
    # neither create a root marker nor add clustered parquet metadata.
    assert not (paths.root / "feature_contract.json").exists()
    for table in (attack_rows, candidates, scores):
        metadata = table.schema.metadata or {}
        assert b"feature_contract_fingerprint" not in metadata
        assert b"source_contract_hash" not in metadata

    per_candidate = pq.read_table(paths.aggregates_dir() / "per_candidate.parquet")
    assert per_candidate.schema.equals(PER_CANDIDATE_AGG_SCHEMA)
    macro = pq.read_table(paths.aggregates_dir() / "macro.parquet")
    assert macro.schema.equals(MACRO_AGG_SCHEMA)

    # primary_margin_y is finite (not NaN) at every AGGREGATE level. The SCORE
    # table no longer carries the column (it is recomputable from the two side
    # losses), so it is excluded here.
    assert "primary_margin_y" not in scores.column_names
    for table in (per_candidate, macro):
        margins = table.column("primary_margin_y").to_pylist()
        assert margins
        assert all(value == value for value in margins)

    # Positive-mode control (k=0) margin is BPB(avoid) - BPB(rich) = 2.0 > 0.
    score_rows = scores.to_pylist()
    positive_control = next(
        row for row in score_rows if row["mode"] == "positive" and row["is_control"]
    )
    assert (
        signed_class_margin(
            mode="positive",
            rich=positive_control["loss_concept_rich"],
            avoid=positive_control["loss_concept_avoiding"],
        )
        == 2.0
    )

    # Headline behavioral claim: as adversarial strength rises, the positive-mode
    # class margin DEGRADES (the stub makes the concept_rich loss climb with
    # strength while the avoid loss is fixed). per_candidate keeps one row per
    # strength (it never collapses across adversarial_strength).
    pos_margin_by_strength = {
        float(row["adversarial_strength"]): float(row["primary_margin_y"])
        for row in per_candidate.to_pylist()
        if row["mode"] == "positive"
    }
    assert set(pos_margin_by_strength) == {0.0, 1.0}
    # k=0 control is the no-adversary baseline; strength 1 is strictly worse.
    assert pos_margin_by_strength[0.0] == 2.0
    assert pos_margin_by_strength[1.0] == 1.0
    assert pos_margin_by_strength[1.0] < pos_margin_by_strength[0.0]
    # The aggregated baseline_primary_margin equals the k=0 control margin.
    pos_rows = [r for r in per_candidate.to_pylist() if r["mode"] == "positive"]
    assert all(r["baseline_primary_margin"] == 2.0 for r in pos_rows)

    # Plots rendered.
    plots = list(paths.plots_dir().rglob("*.png"))
    assert plots


def test_cli_stage_chain_default_roundtrip_validate_through_plot(
    tmp_path: Path, monkeypatch
) -> None:
    source_root = tmp_path / "source"
    _write_source_run(source_root)
    output_root = tmp_path / "out"

    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_root),
        output_root=str(output_root),
        run_tag="run",
        device="cpu",
        stage=Stage.ALL.value,
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0, 1),
                max_steps=0,
            )
        ],
        # score_general_val isn't mocked in this chain; general_val_bpb now
        # defaults to enabled=True (needs a real model), so keep it off.
        general_val_bpb=GeneralValBpbConfig(enabled=False),
        scatter_baseline_x="baseline_loss",
    )
    assert cfg.steering_metric == "roundtrip"
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    tokenizer = _TinyTokenizer()
    expected_bpb_losses: dict[str, tuple[float, float]] = {}

    monkeypatch.setattr(
        bpb_adv_cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg: ({"decoder_only": _meta()}, tokenizer),
    )

    def _fake_work_specs(**_kwargs: Any) -> list[BpbAttackWorkSpec]:
        return [
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
            BpbAttackWorkSpec(
                method_name="neg_diffmean_decoder_only",
                mode=SteeringMode.NEGATIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
        ]

    monkeypatch.setattr(bpb_adv_cli, "work_specs_from_method_registry", _fake_work_specs)
    monkeypatch.setattr(
        bpb_adv_cli,
        "_stage_prepare_context_cache",
        lambda *_args, **_kwargs: None,
    )

    def _fake_candidate_stage(*, cfg: AxBenchBpbAdvConfig, paths: AxBenchBpbAdvPaths) -> int:
        return run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=tokenizer,
            embedding_weight=t.zeros(_VOCAB, 1),
            test_objective=zero_objective,
            test_gradient_provider_batch=zero_gradient_batch,
        )

    monkeypatch.setattr(bpb_adv_cli, "run_model_backed_bpb_candidate_stage", _fake_candidate_stage)

    def _fake_roundtrip_score_stage(
        *,
        cfg: AxBenchBpbAdvConfig,
        paths: AxBenchBpbAdvPaths,
        candidates: pa.Table,
        stats: dict[str, int] | None = None,
    ) -> pa.Table:
        del cfg, paths
        roundtrip_scores, bpb_scores = _roundtrip_scores_from_bpb_stub(
            candidates,
            tokenizer=tokenizer,
        )
        for bpb_row in bpb_scores.to_pylist():
            expected_bpb_losses[str(bpb_row["candidate_id"])] = (
                float(bpb_row["loss_concept_rich"]),
                float(bpb_row["loss_concept_avoiding"]),
            )
        if stats is not None:
            stats.update(
                {
                    "cache_hits": 0,
                    "scored_unique": bpb_scores.num_rows,
                    "generation_cache_hits": 0,
                    "generated_unique": roundtrip_scores.num_rows,
                    "generated_tokens": 3 * roundtrip_scores.num_rows,
                    "control_generation_cache_hits": 0,
                    "control_generated_unique": roundtrip_scores.num_rows,
                    "control_generated_tokens": 2 * roundtrip_scores.num_rows,
                    "probe_rows": roundtrip_scores.num_rows,
                }
            )
        rows = roundtrip_scores.to_pylist()
        for mode, method_name in (
            ("positive", "feature_diffmean_encdec"),
            ("negative", "neg_feature_diffmean_encdec"),
        ):
            base = next(row for row in rows if row["mode"] == mode and row["is_control"])
            candidate_id = f"feature-diffmean-target-{mode}"
            expected_bpb_losses[candidate_id] = (
                float(base["loss_concept_rich"]),
                float(base["loss_concept_avoiding"]),
            )
            rows.append(
                {
                    **base,
                    "candidate_id": candidate_id,
                    "method_name": method_name,
                    "target_model": "encdec",
                    "coefficient_str": "None",
                }
            )
        return pa.Table.from_pylist(rows, schema=ROUNDTRIP_SCORE_SCHEMA)

    monkeypatch.setattr(
        bpb_adv_cli,
        "run_model_backed_roundtrip_score_stage",
        _fake_roundtrip_score_stage,
    )

    assert (
        bpb_adv_cli._run_resolved(cfg, ddp_rank=0, ddp_world_size=1)
        is cfg
    )
    assert _output_inventory(paths.root) == _ROUNDTRIP_OUTPUT_INVENTORY

    attack_rows = pq.read_table(paths.attack_rows_path())
    assert attack_rows.schema.equals(ATTACK_ROWS_SCHEMA)
    candidates = pq.read_table(paths.candidates_path())
    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    roundtrip_scores = pq.read_table(paths.roundtrip_scores_path())
    assert roundtrip_scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert not paths.scores_path().exists()
    assert not (paths.root / "feature_contract.json").exists()
    for table in (attack_rows, candidates, roundtrip_scores):
        metadata = table.schema.metadata or {}
        assert b"feature_contract_fingerprint" not in metadata
        assert b"source_contract_hash" not in metadata

    assert expected_bpb_losses
    for row in roundtrip_scores.to_pylist():
        assert (
            row["loss_concept_rich"],
            row["loss_concept_avoiding"],
        ) == expected_bpb_losses[str(row["candidate_id"])]

    per_candidate = pq.read_table(paths.roundtrip_aggregates_dir() / "per_candidate.parquet")
    assert per_candidate.schema.equals(ROUNDTRIP_PER_CANDIDATE_AGG_SCHEMA)
    macro = pq.read_table(paths.roundtrip_aggregates_dir() / "macro.parquet")
    assert macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)
    assert set(macro.column("adversarial_strength").to_pylist()) == {0.0, 1.0}
    assert all(
        0.0 <= float(value) <= 100.0 for value in macro.column("primary_roundtrip_y").to_pylist()
    )

    plots = list(paths.roundtrip_plots_dir().rglob("*.png"))
    assert plots
    assert (paths.roundtrip_plots_dir() / "plot_index.csv").exists()


def test_stage_plot_threads_plot_methods_and_scatter_baseline_x(
    tmp_path: Path, monkeypatch
) -> None:
    """The PLOT driver seam: _stage_plot must thread cfg.plot_methods AND
    cfg.scatter_baseline_x into plot_bpb_adv_summaries. Pins the wiring so a
    dropped/mis-routed kwarg (e.g. cfg.base_methods instead of cfg.plot_methods)
    fails -- the PR#503/#504 'plot-path config knob silently disconnected' class.
    """

    output_root = tmp_path / "out"
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "src"),
        output_root=str(output_root),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        steering_metric="bpb_class_margin",
        plot_methods=["diffmean_decoder_only"],
        scatter_baseline_x="min",
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    # Pre-write an (empty) macro so _stage_plot skips aggregation and goes
    # straight to plotting; the recording stub never inspects the contents.
    pq.write_table(
        pa.Table.from_pylist([], schema=MACRO_AGG_SCHEMA),
        paths.aggregates_dir() / "macro.parquet",
    )

    captured: dict[str, Any] = {}

    def _spy(
        macro: pa.Table,
        plots_dir: Path,
        *,
        scatter_baseline_x: str = "baseline_loss",
        plot_methods: list[str] | None = None,
    ) -> list[Path]:
        captured["scatter_baseline_x"] = scatter_baseline_x
        captured["plot_methods"] = plot_methods
        return []

    monkeypatch.setattr(bpb_adv_cli, "plot_bpb_adv_summaries", _spy)

    bpb_adv_cli._stage_plot(cfg)

    assert captured["plot_methods"] == ["diffmean_decoder_only"]
    assert captured["scatter_baseline_x"] == "min"


def _plot_cfg(tmp_path: Path) -> tuple[AxBenchBpbAdvConfig, AxBenchBpbAdvPaths]:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "src"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        steering_metric="bpb_class_margin",
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    return cfg, paths


def test_stage_plot_reaggregates_stale_schema_macro(tmp_path: Path, monkeypatch) -> None:
    """F6 added is_control to MACRO_AGG_SCHEMA, and _is_control_row now hard-indexes
    it. A macro.parquet written by PRE-F6 code lacks the column; on a PLOT-stage
    resume _stage_plot must DETECT the stale schema and re-aggregate, not feed the
    stale table to the plot (which would KeyError('is_control'))."""
    cfg, paths = _plot_cfg(tmp_path)
    # Stale macro = the current schema MINUS is_control (the pre-F6 shape).
    stale_schema = MACRO_AGG_SCHEMA.remove(MACRO_AGG_SCHEMA.get_field_index("is_control"))
    pq.write_table(
        pa.Table.from_pylist([], schema=stale_schema),
        paths.aggregates_dir() / "macro.parquet",
    )

    reaggregated = {"called": False}

    def _fake_aggregate(
        scores_path: Path, out_dir: Path, *, general_val_path: Path | None = None
    ) -> None:
        reaggregated["called"] = True
        pq.write_table(
            pa.Table.from_pylist([], schema=MACRO_AGG_SCHEMA),
            Path(out_dir) / "macro.parquet",
        )

    monkeypatch.setattr(bpb_adv_cli, "aggregate_bpb_adv_scores", _fake_aggregate)
    captured: dict[str, Any] = {}

    def _spy_plot(
        macro: pa.Table,
        plots_dir: Path,
        *,
        scatter_baseline_x: str = "baseline_loss",
        plot_methods: Any = None,
    ) -> list[Path]:
        captured["schema"] = macro.schema
        return []

    monkeypatch.setattr(bpb_adv_cli, "plot_bpb_adv_summaries", _spy_plot)

    bpb_adv_cli._stage_plot(cfg)

    assert reaggregated["called"] is True
    # The plot received the FRESH (current-schema) macro, not the stale one.
    assert captured["schema"].equals(MACRO_AGG_SCHEMA)


def test_stage_plot_keeps_current_schema_macro_without_reaggregating(
    tmp_path: Path, monkeypatch
) -> None:
    """The converse guard: a current-schema macro.parquet is plotted as-is (no
    wasteful re-aggregation)."""
    cfg, paths = _plot_cfg(tmp_path)
    pq.write_table(
        pa.Table.from_pylist([], schema=MACRO_AGG_SCHEMA),
        paths.aggregates_dir() / "macro.parquet",
    )

    reaggregated = {"called": False}

    def _fake_aggregate(
        scores_path: Path, out_dir: Path, *, general_val_path: Path | None = None
    ) -> None:
        reaggregated["called"] = True

    monkeypatch.setattr(bpb_adv_cli, "aggregate_bpb_adv_scores", _fake_aggregate)
    monkeypatch.setattr(bpb_adv_cli, "plot_bpb_adv_summaries", lambda *a, **k: [])

    bpb_adv_cli._stage_plot(cfg)

    assert reaggregated["called"] is False


def _one_sided_rows(prefix: str, *, content_class: str, output: str) -> list[dict[str, Any]]:
    return [
        {
            "concept_id": 1,
            "content_class": content_class,
            "prefix_text": prefix,
            "output_text": output,
            "qc_accepted": True,
            "prefix_source_shard": 0,
            "prefix_source_doc": 0,
        }
    ]


def _write_incomplete_source_run(root: Path, *, n_incomplete: int) -> None:
    """Source with one complete prefix + ``n_incomplete`` one-sided prefixes."""

    root.mkdir(parents=True)
    concepts_payload = json.dumps(
        [{"concept_id": 1, "description": "concept"}]
    )
    (root / "concepts.json").write_text(concepts_payload, encoding="utf-8")
    (root / "concepts_stage1c.json").write_text(
        concepts_payload,
        encoding="utf-8",
    )
    positive = list(_both_class_rows("abc", rich="rich", avoid="avoid"))
    for idx in range(n_incomplete):
        # Alternate which side is present so we exercise both missing sides.
        side = "concept_rich" if idx % 2 == 0 else "concept_avoiding"
        positive += _one_sided_rows(f"bad{idx}", content_class=side, output="x")
    pq.write_table(
        pa.Table.from_pylist(
            positive,
            schema=schema_with_positive_qc_contract(
                _SYNTH_SCHEMA,
                accept_threshold=0.3,
            ),
        ),
        root / "synth_training_positive.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _both_class_rows("xyz", rich="neg-rich", avoid="neg-avoid"),
            schema=_SYNTH_SCHEMA,
        ),
        root / "synth_training_negative.parquet",
    )
    for name in (
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
    ):
        (root / name).write_bytes(b"x")
    write_seed_contract(root)
    prompting = root / "baselines" / "prompting"
    prompting.mkdir(parents=True)
    (prompting / "positive_descriptions.json").write_text("{}", encoding="utf-8")


@pytest.mark.slow
def test_cli_stage_chain_rejects_incomplete_prefixes(tmp_path: Path) -> None:
    """An incomplete-prefix source fails before any attack-row artifacts are written."""

    source_root = tmp_path / "source"
    _write_incomplete_source_run(source_root, n_incomplete=3)
    output_root = tmp_path / "out"

    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_root),
        output_root=str(output_root),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0,),
                max_steps=0,
            )
        ],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")

    with pytest.raises(AxBenchBpbAdvError, match="missing a class side"):
        bpb_adv_cli._stage_validate_source(cfg)

    assert not paths.source_contract().exists()
    assert not paths.attack_rows_path().exists()


def _row(
    prefix: str,
    *,
    content_class: str,
    output: str,
    concept_id: int = 1,
    qc_accepted: bool = True,
) -> dict[str, Any]:
    return {
        "concept_id": concept_id,
        "content_class": content_class,
        "prefix_text": prefix,
        "output_text": output,
        "qc_accepted": qc_accepted,
        "prefix_source_shard": 0,
        "prefix_source_doc": 0,
    }


def _write_filtered_population_source_run(root: Path) -> None:
    """Source whose incomplete prefixes are induced by the FILTERED population.

    The raw positive parquet has every IN-CONCEPTS prefix double-sided, but two
    prefixes become incomplete after ``filter_unaccepted_samples=True``. The
    validator must check the same filtered population that prepare_attack_rows
    would pair:

    - ``abc``    : both classes, qc-accepted, concept 1 -> survives (complete).
    - ``qcdrop1``: rich accepted + avoiding qc-REJECTED (concept 1) -> after
      ``filter_unaccepted_samples=True`` it loses its avoiding side -> error.
    - ``qcdrop2``: rich accepted + avoiding qc-REJECTED (concept 1) -> error.
    - ``rogue``  : a one-sided row whose concept_id is NOT in concepts.json ->
      the concept-id filter removes it entirely, so it is not an error.
    """

    root.mkdir(parents=True)
    concepts_payload = json.dumps(
        [{"concept_id": 1, "description": "concept"}]
    )
    (root / "concepts.json").write_text(concepts_payload, encoding="utf-8")
    (root / "concepts_stage1c.json").write_text(
        concepts_payload,
        encoding="utf-8",
    )
    positive = [
        *_both_class_rows("abc", rich="rich", avoid="avoid"),
        _row("qcdrop1", content_class="concept_rich", output="r", qc_accepted=True),
        _row("qcdrop1", content_class="concept_avoiding", output="a", qc_accepted=False),
        _row("qcdrop2", content_class="concept_rich", output="r", qc_accepted=True),
        _row("qcdrop2", content_class="concept_avoiding", output="a", qc_accepted=False),
        _row("rogue", content_class="concept_rich", output="x", concept_id=999),
    ]
    pq.write_table(
        pa.Table.from_pylist(
            positive,
            schema=schema_with_positive_qc_contract(
                _SYNTH_SCHEMA,
                accept_threshold=0.3,
            ),
        ),
        root / "synth_training_positive.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            _both_class_rows("xyz", rich="neg-rich", avoid="neg-avoid"),
            schema=_SYNTH_SCHEMA,
        ),
        root / "synth_training_negative.parquet",
    )
    for name in (
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
    ):
        (root / name).write_bytes(b"x")
    write_seed_contract(root)
    prompting = root / "baselines" / "prompting"
    prompting.mkdir(parents=True)
    (prompting / "positive_descriptions.json").write_text("{}", encoding="utf-8")


@pytest.mark.slow
def test_cli_stage_chain_rejects_qc_induced_incomplete_prefixes(
    tmp_path: Path,
) -> None:
    """Completeness is checked after QC and source-concept filtering."""

    source_root = tmp_path / "source"
    _write_filtered_population_source_run(source_root)
    output_root = tmp_path / "out"

    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_root),
        output_root=str(output_root),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        filter_unaccepted_samples=True,
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0,),
                max_steps=0,
            )
        ],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")

    with pytest.raises(AxBenchBpbAdvError, match="qcdrop"):
        bpb_adv_cli._stage_validate_source(cfg)

    assert not paths.source_contract().exists()
    assert not paths.attack_rows_path().exists()
