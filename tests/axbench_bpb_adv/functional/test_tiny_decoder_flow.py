from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from steering.eval.axbench_base.class_margin import signed_class_margin
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.types import BpbScoreRow, ScoringInputs, SteeringMode
from steering.eval.axbench_bpb_adv.aggregation import (
    aggregate_bpb_adv_scores,
    plot_bpb_adv_summaries,
)
from steering.eval.axbench_bpb_adv.attack_rows import (
    BpbAttackPrefix,
    BpbAttackWorkSpec,
    core_prefix_id,
    prepare_bpb_attack_rows,
)
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
)
from steering.eval.axbench_bpb_adv.evaluation_population import write_evaluation_population
from steering.eval.axbench_bpb_adv.model_optimize import (
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import SCORE_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return 128

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


_PREFIX_TEXT = "abc"
_PREFIX_TOKENS = [0, 97, 98, 99]
_RICH_OUTPUT = [100]
_AVOID_OUTPUT = [101]


def _prefix() -> BpbAttackPrefix:
    rich_id = make_sample_id(
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        content_class="concept_rich",
        prefix_text=_PREFIX_TEXT,
        output_text="rich",
    )
    avoid_id = make_sample_id(
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        content_class="concept_avoiding",
        prefix_text=_PREFIX_TEXT,
        output_text="avoid",
    )
    return BpbAttackPrefix(
        core_prefix_id=core_prefix_id(
            concept_id=1, mode=SteeringMode.POSITIVE, prefix_text=_PREFIX_TEXT
        ),
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        prefix_text=_PREFIX_TEXT,
        prefix_hash=make_prefix_hash(_PREFIX_TEXT),
        prefix_tokens=list(_PREFIX_TOKENS),
        concept_rich_sample_ids=[rich_id],
        concept_avoiding_sample_ids=[avoid_id],
        concept_rich_output_tokens=[list(_RICH_OUTPUT)],
        concept_avoiding_output_tokens=[list(_AVOID_OUTPUT)],
    )


def _multi_pair_prefix(
    *,
    rich_outputs: list[list[int]],
    avoid_outputs: list[list[int]],
) -> BpbAttackPrefix:
    rich_ids = [
        make_sample_id(
            concept_id=1,
            mode=SteeringMode.POSITIVE,
            content_class="concept_rich",
            prefix_text=_PREFIX_TEXT,
            output_text=f"rich-{idx}",
        )
        for idx in range(len(rich_outputs))
    ]
    avoid_ids = [
        make_sample_id(
            concept_id=1,
            mode=SteeringMode.POSITIVE,
            content_class="concept_avoiding",
            prefix_text=_PREFIX_TEXT,
            output_text=f"avoid-{idx}",
        )
        for idx in range(len(avoid_outputs))
    ]
    return BpbAttackPrefix(
        core_prefix_id=core_prefix_id(
            concept_id=1, mode=SteeringMode.POSITIVE, prefix_text=_PREFIX_TEXT
        ),
        concept_id=1,
        mode=SteeringMode.POSITIVE,
        prefix_text=_PREFIX_TEXT,
        prefix_hash=make_prefix_hash(_PREFIX_TEXT),
        prefix_tokens=list(_PREFIX_TOKENS),
        concept_rich_sample_ids=rich_ids,
        concept_avoiding_sample_ids=avoid_ids,
        concept_rich_output_tokens=[list(out) for out in rich_outputs],
        concept_avoiding_output_tokens=[list(out) for out in avoid_outputs],
    )


def _make_side_row(
    candidate_row: Mapping[str, Any],
    scoring_input: ScoringInputs,
    *,
    bpb: float,
    n_output_bytes: int,
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
        truncated=False,
    )


def _dual_score_pair(
    candidate_row: Mapping[str, Any],
    rich_input: ScoringInputs,
    avoid_input: ScoringInputs,
) -> tuple[BpbScoreRow, BpbScoreRow]:
    # Both completions share the prompt; only the output side differs.
    assert rich_input.prefix_tokens == avoid_input.prefix_tokens
    assert rich_input.prefix_tokens == candidate_row["attacked_prompt_tokens"]
    assert rich_input.sample_id == candidate_row["concept_rich_sample_id"]
    assert avoid_input.sample_id == candidate_row["concept_avoiding_sample_id"]
    # rich loss rises with adversarial strength -> positive-mode margin shrinks.
    strength = float(candidate_row["adversarial_strength"])
    rich = _make_side_row(candidate_row, rich_input, bpb=1.0 + strength, n_output_bytes=2)
    avoid = _make_side_row(candidate_row, avoid_input, bpb=3.0, n_output_bytes=4)
    return rich, avoid


def test_tiny_decoder_artifact_flow_is_idempotent(tmp_path: Path) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/unused/source",
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/unused/features",
        feature_statistics_path="/unused/stats",
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0, 1),
                max_steps=0,
            )
        ],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    attack_rows = prepare_bpb_attack_rows(
        prefixes=[_prefix()],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            )
        ],
        cfg=cfg,
    )
    paths.attack_rows_dir().mkdir(parents=True)
    pq.write_table(attack_rows, paths.attack_rows_path())
    write_evaluation_population(
        paths.evaluation_population(),
        attack_rows.to_pylist(),
        ours=cfg.ours_gcg,
    )

    tokenizer = _TinyTokenizer()
    first_count = run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=tokenizer,
        embedding_weight=t.zeros(128, 1),
        test_objective=zero_objective,
        test_gradient_provider_batch=zero_gradient_batch,
    )
    first_candidates = pq.read_table(paths.candidates_path()).to_pylist()
    second_count = run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=tokenizer,
        embedding_weight=t.zeros(128, 1),
        test_objective=zero_objective,
        test_gradient_provider_batch=zero_gradient_batch,
    )
    second_candidates = pq.read_table(paths.candidates_path()).to_pylist()

    assert first_count == second_count == 2
    assert first_candidates == second_candidates

    scores = build_bpb_adv_score_table(
        candidates=pq.read_table(paths.candidates_path()),
        context_segments_by_prefix_hash={},
        tokenizer=tokenizer,
        token_bytes=t.ones(128, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 16},
        score_pair=_dual_score_pair,
    )
    score_rows = scores.to_pylist()
    control_score = next(row for row in score_rows if row["is_control"])
    score_rows.append(
        {
            **control_score,
            "candidate_id": "feature-diffmean-target-positive",
            "method_name": "feature_diffmean_encdec",
            "target_model": "encdec",
            "coefficient_str": "None",
        }
    )
    scores = pa.Table.from_pylist(score_rows, schema=SCORE_SCHEMA)
    scores_path = tmp_path / "scores.parquet"
    pq.write_table(scores, scores_path)
    macro = aggregate_bpb_adv_scores(scores_path, tmp_path / "aggregate")
    plot_paths = plot_bpb_adv_summaries(macro, tmp_path / "plots")

    score_rows = [row for row in scores.to_pylist() if row["target_model"] == "decoder_only"]
    # Both completions scored under one steered intervention: real signed margin.
    assert len(score_rows) == 2
    control = next(row for row in score_rows if row["is_control"])
    assert control["loss_concept_rich"] == 1.0
    assert control["loss_concept_avoiding"] == 3.0
    # positive-mode margin = BPB(avoid) - BPB(rich) = 2.0, finite and > 0. The
    # SCORE table no longer carries primary_margin_y; recompute from side losses.
    assert "primary_margin_y" not in scores.column_names
    assert signed_class_margin(
        mode="positive",
        rich=control["loss_concept_rich"],
        avoid=control["loss_concept_avoiding"],
    ) == 2.0
    assert all(
        math.isfinite(
            signed_class_margin(
                mode=row["mode"],
                rich=row["loss_concept_rich"],
                avoid=row["loss_concept_avoiding"],
            )
        )
        for row in score_rows
    )

    assert macro.num_rows >= 1
    macro_rows = macro.to_pylist()
    assert all(
        row["primary_margin_y"] == row["primary_margin_y"] for row in macro_rows
    )
    assert plot_paths
    assert all(path.exists() for path in plot_paths)


def _bpb_by_output_token(
    candidate_row: Mapping[str, Any],
    rich_input: ScoringInputs,
    avoid_input: ScoringInputs,
) -> tuple[BpbScoreRow, BpbScoreRow]:
    """Distinct BPB per completion (keyed off its first output token).

    This lets the Cartesian (rich_i, avoid_j) pairs carry distinct per-side
    losses so the averaging identity is exercised end-to-end.
    """

    rich_bpb = float(rich_input.output_tokens[0])
    avoid_bpb = float(avoid_input.output_tokens[0])
    rich = _make_side_row(candidate_row, rich_input, bpb=rich_bpb, n_output_bytes=2)
    avoid = _make_side_row(candidate_row, avoid_input, bpb=avoid_bpb, n_output_bytes=2)
    return rich, avoid


def test_tiny_decoder_flow_explodes_cartesian_pairs_through_score_and_aggregate(
    tmp_path: Path,
) -> None:
    # 3 rich x 2 avoid completions of ONE core prefix -> 6 Cartesian pairs flow
    # through candidate -> score -> aggregate. pair_subsample=0 keeps them all.
    rich_outputs = [[100], [105], [110]]
    avoid_outputs = [[101], [106]]
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/unused/source",
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/unused/features",
        feature_statistics_path="/unused/stats",
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        pair_subsample=0,
        adversarial_methods=[AdditiveDirectionConfig(magnitudes=(0.0,))],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    attack_rows = prepare_bpb_attack_rows(
        prefixes=[
            _multi_pair_prefix(rich_outputs=rich_outputs, avoid_outputs=avoid_outputs)
        ],
        work_specs=[
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            )
        ],
        cfg=cfg,
    )
    # 3 rich x 2 avoid x 1 magnitude (additive, control only) = 6 attack rows.
    assert attack_rows.num_rows == 6
    assert len(set(attack_rows.column("attack_row_id").to_pylist())) == 6
    paths.attack_rows_dir().mkdir(parents=True)
    pq.write_table(attack_rows, paths.attack_rows_path())
    write_evaluation_population(
        paths.evaluation_population(),
        attack_rows.to_pylist(),
        ours=cfg.ours_gcg,
    )

    tokenizer = _TinyTokenizer()
    count = run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=tokenizer,
        embedding_weight=t.zeros(128, 1),
        test_objective=zero_objective,
        test_gradient_provider_batch=zero_gradient_batch,
    )
    assert count == 6
    candidates = pq.read_table(paths.candidates_path())
    # Distinct candidate_id per Cartesian pair (no dup-guard false-trip).
    assert len(set(candidates.column("candidate_id").to_pylist())) == 6

    scores = build_bpb_adv_score_table(
        candidates=candidates,
        context_segments_by_prefix_hash={},
        tokenizer=tokenizer,
        token_bytes=t.ones(128, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 16},
        score_pair=_bpb_by_output_token,
    )
    assert scores.num_rows == 6
    scores_path = tmp_path / "scores.parquet"
    pq.write_table(scores, scores_path)
    macro = aggregate_bpb_adv_scores(scores_path, tmp_path / "aggregate")

    # All 6 pairs share one (concept, mode, method, strength, alpha, context)
    # group -> a single macro row averaging each side, margin via _signed_margin.
    macro_rows = macro.to_pylist()
    assert len(macro_rows) == 1
    row = macro_rows[0]
    mean_rich = (100.0 + 105.0 + 110.0) / 3.0
    mean_avoid = (101.0 + 106.0) / 2.0
    assert row["mean_loss_concept_rich"] == mean_rich
    assert row["mean_loss_concept_avoiding"] == mean_avoid
    # positive-mode signed margin of the SIDE MEANS, not the mean of pair margins.
    assert row["primary_margin_y"] == mean_avoid - mean_rich
