from __future__ import annotations

import math
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
from steering.conditional.feature_gcg.candidate_search import (
    rank_replacements_from_gradients,
    sample_candidate_sequences,
)
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, FasterGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import write_evaluation_population
from steering.eval.axbench_bpb_adv.model_optimize import run_model_backed_bpb_candidate_stage
from steering.eval.axbench_bpb_adv.optimize import (
    build_control_candidate,
    build_gepa_candidates,
    objective_gradient,
    objective_value,
    parse_gepa_token_candidate,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA, CANDIDATES_SCHEMA
from tests.axbench_bpb_adv.faster_gcg_oracle import build_faster_gcg_candidates
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _attack_rows(*, editable_count: int = 1) -> pa.Table:
    tokens = [0, 97, 98, 99]
    start = len(tokens) - editable_count
    if editable_count == 0:
        start = len(tokens)
    rich_output = [100]
    avoid_output = [101]
    return pa.Table.from_pylist(
        [
            {
                "attack_row_id": "row",
                "core_prefix_id": "coreprefix000001",
                "concept_rich_sample_id": "rich-sample",
                "concept_avoiding_sample_id": "avoid-sample",
                "concept_id": 1,
                "mode": "positive",
                "method_name": "diffmean_encdec",
                "target_model": "encdec",
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": float(editable_count),
                "strength_unit": "optimized_tokens",
                "strength_index": editable_count,
                "context_signal_mode": "fixed_context",
                "prefix_text": "abc",
                "prefix_hash": "hash",
                "source_prompt_tokens": tokens,
                "source_prompt_token_digest": token_sequence_digest(tokens),
                "concept_rich_output_tokens": rich_output,
                "concept_avoiding_output_tokens": avoid_output,
                "concept_rich_output_digest": token_sequence_digest(rich_output),
                "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
                "editable_token_start": start,
                "editable_token_count": editable_count,
                "original_editable_tokens": tokens[start:],
            }
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )


def _write_eval_population(
    paths: AxBenchBpbAdvPaths,
    rows: pa.Table,
    cfg: AxBenchBpbAdvConfig,
) -> None:
    write_evaluation_population(
        paths.evaluation_population(),
        rows.to_pylist(),
        ours=cfg.ours_gcg,
    )


def _gradient_provider(
    _row: dict[str, Any],
    tokens: Sequence[int],
    _step: int,
) -> t.Tensor:
    gradient = t.zeros(len(tokens), 2, dtype=t.float32)
    gradient[-1] = t.tensor([1.0, 0.0])
    return gradient


def test_search_objective_g_minimizes_margin() -> None:
    # g = loss_contrast - loss_desired (the unweighted steering margin, MINIMIZED).
    loss_desired = 7.0
    loss_contrast = 4.0
    expected = loss_contrast - loss_desired
    assert objective_value(
        loss_desired=loss_desired,
        loss_contrast=loss_contrast,
    ) == pytest.approx(expected)


def test_search_objective_equals_primary_margin_positive() -> None:
    # Positive mapping: desired=rich, contrast=avoid; primary_margin_y = avoid - rich.
    loss_rich = 7.0
    loss_avoid = 4.0
    primary_margin_y = loss_avoid - loss_rich
    g = objective_value(
        loss_desired=loss_rich,
        loss_contrast=loss_avoid,
    )
    assert g == pytest.approx(primary_margin_y)


def test_search_objective_gradient_g_form() -> None:
    # grad g = grad_contrast - grad_desired (the unweighted steering margin grad).
    grad = objective_gradient(
        grad_loss_desired=t.tensor([5.0, 1.0]),
        grad_loss_contrast=t.tensor([2.0, 4.0]),
    )
    expected = t.tensor([2.0, 4.0]) - t.tensor([5.0, 1.0])
    assert t.equal(grad, expected)


def test_control_candidate_carries_both_completions_and_core_prefix_id() -> None:
    row = _attack_rows(editable_count=0).to_pylist()[0]
    candidate = build_control_candidate(row=row, tokenizer=_TinyTokenizer())
    assert candidate["core_prefix_id"] == "coreprefix000001"
    assert candidate["concept_rich_sample_id"] == "rich-sample"
    assert candidate["concept_avoiding_sample_id"] == "avoid-sample"
    assert candidate["concept_rich_output_tokens"] == [100]
    assert candidate["concept_avoiding_output_tokens"] == [101]
    assert "content_class" not in candidate
    assert "output_tokens" not in candidate


def test_faster_gcg_builds_final_candidate_and_not_saved_steps() -> None:
    cfg = FasterGcgConfig(
        max_steps=2,
        candidate_batch_size=1,
        top_k=1,
        distance_weight=0.0,
        temperature=1.0,
    )
    embedding_weight = t.zeros(121, 2, dtype=t.float32)
    embedding_weight[120] = t.tensor([-1.0, 0.0])
    events: list[dict[str, Any]] = []

    candidates = build_faster_gcg_candidates(
        attack_rows=_attack_rows(),
        tokenizer=_TinyTokenizer(),
        cfg=cfg,
        embedding_weight=embedding_weight,
        allowed_token_ids=[120],
        objective=lambda _row, tokens: 0.0 if int(tokens[-1]) == 120 else 1.0,
        gradient_provider=_gradient_provider,
        progress_callback=events.append,
    )

    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    rows = candidates.to_pylist()
    assert len(rows) == 1
    assert rows[0]["attacked_prompt_tokens"] == [0, 97, 98, 120]
    assert rows[0]["replacement_tokens"] == [120]
    assert rows[0]["objective_value"] == 0.0
    assert [event["event"] for event in events] == [
        "row_start",
        "step_done",
        "step_done",
        "row_done",
    ]
    assert events[0]["row_index"] == 1
    assert events[0]["total_rows"] == 1
    assert events[-1]["objective_value"] == 0.0
    assert events[-1]["accepted_steps"] == 1


def test_faster_gcg_replacement_scores_include_current_token_delta() -> None:
    embedding_weight = t.tensor(
        [
            [0.0],
            [10.0],
            [0.0],
            [2.0],
        ],
        dtype=t.float32,
    )
    gradient = t.ones(3, 1, dtype=t.float32)

    replacements = rank_replacements_from_gradients(
        embedding_gradient=gradient,
        embedding_weight=embedding_weight,
        editable_positions=[1, 2],
        allowed_token_ids=[3],
        current_tokens=[0, 1, 2],
        top_k=1,
        distance_weight=0.0,
    )

    assert replacements[1][0].score == -8.0
    assert replacements[2][0].score == 2.0


def test_faster_gcg_k0_bypasses_objective() -> None:
    cfg = FasterGcgConfig()
    events: list[dict[str, Any]] = []

    candidates = build_faster_gcg_candidates(
        attack_rows=_attack_rows(editable_count=0),
        tokenizer=_TinyTokenizer(),
        cfg=cfg,
        embedding_weight=t.zeros(2, 1),
        allowed_token_ids=[1],
        objective=lambda _row, _tokens: pytest.fail("k=0 should not optimize"),
        gradient_provider=lambda _row, _tokens, _step: pytest.fail("k=0 should not optimize"),
        progress_callback=events.append,
    )

    row = candidates.to_pylist()[0]
    assert row["attacked_prompt_tokens"] == [0, 97, 98, 99]
    assert row["replacement_tokens"] == []
    assert events == [
        {
            "event": "row_done",
            "row_index": 1,
            "total_rows": 1,
            "attack_row_id": "row",
            "target_model": "encdec",
            "method_name": "diffmean_encdec",
            "coefficient_str": "1.0",
            "adversarial_strength": 0.0,
            "context_signal_mode": "fixed_context",
            "editable_token_count": 0,
            "objective_value": None,
            "accepted_steps": 0,
            "visited_sequences": 1,
            "control": True,
        }
    ]


def test_gepa_candidate_builder_emits_progress_events() -> None:
    rows = _attack_rows().to_pylist()
    rows[0]["adversarial_method"] = "gepa"
    rows[0]["adversarial_strength"] = 8.0
    rows[0]["strength_unit"] = "metric_calls"
    table = pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)
    events: list[dict[str, Any]] = []

    candidates = build_gepa_candidates(
        attack_rows=table,
        tokenizer=_TinyTokenizer(),
        optimizer=lambda _row, seed, _budget: list(seed),
        progress_callback=events.append,
    )

    assert candidates.num_rows == 1
    assert [event["event"] for event in events] == ["row_start", "row_done"]
    assert events[0]["adversarial_strength"] == 8.0
    assert events[0]["editable_token_count"] == 1


def test_gepa_candidate_parser_rejects_invalid_artifacts() -> None:
    assert parse_gepa_token_candidate("[1, 2]", expected_len=2, vocab_size=10) == [1, 2]
    with pytest.raises(AxBenchBpbAdvError, match="length"):
        parse_gepa_token_candidate("[1]", expected_len=2, vocab_size=10)
    with pytest.raises(AxBenchBpbAdvError, match="outside vocab"):
        parse_gepa_token_candidate("[11]", expected_len=1, vocab_size=10)
    with pytest.raises(AxBenchBpbAdvError, match="JSON"):
        parse_gepa_token_candidate("not-json", expected_len=1, vocab_size=10)


def test_model_backed_candidate_stage_requires_test_tokenizer_with_embedding_weight(
    tmp_path: Path,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.attack_rows_dir().mkdir(parents=True)
    rows = _attack_rows(editable_count=1)
    pq.write_table(rows, paths.attack_rows_path())

    with pytest.raises(AxBenchBpbAdvError, match="tokenizer must be provided"):
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            embedding_weight=t.zeros(2, 1),
        )


def test_embedding_weight_candidate_path_requires_explicit_test_callbacks(
    tmp_path: Path,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(1,),
                max_steps=1,
                candidate_batch_size=1,
                top_k=1,
                distance_weight=0.0,
                temperature=1.0,
            )
        ],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.attack_rows_dir().mkdir(parents=True)
    rows = _attack_rows(editable_count=1)
    pq.write_table(rows, paths.attack_rows_path())
    _write_eval_population(paths, rows, cfg)

    with pytest.raises(AxBenchBpbAdvError, match="explicit test_objective"):
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=_TinyTokenizer(),
            embedding_weight=t.zeros(2, 1),
        )


def test_embedding_weight_candidate_path_filters_allowed_ids_to_embedding_vocab(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(1,),
                max_steps=1,
                candidate_batch_size=1,
                top_k=2,
                distance_weight=0.0,
                temperature=1.0,
            )
        ],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.attack_rows_dir().mkdir(parents=True)
    rows = _attack_rows(editable_count=1)
    pq.write_table(rows, paths.attack_rows_path())
    _write_eval_population(paths, rows, cfg)
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tokenizer: [1, 999])

    written = run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_TinyTokenizer(),
        embedding_weight=t.zeros(100, 1),
        test_objective=zero_objective,
        test_gradient_provider_batch=zero_gradient_batch,
    )

    assert written == 1


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_gradient_does_not_crash_candidate_sampling(
    bad_value: float,
) -> None:
    """Once the producer clamps non-finite gradient entries (the F2 fix), the
    Faster-GCG ranking + sampling path must complete with finite proposal scores
    instead of raising ``ValueError: Total of weights must be finite``.

    This encodes the contract that the clamp at the gradient producer
    (``t.nan_to_num(...)`` in ``teacher_forced_bpb_gradient``) renders the
    downstream candidate stage safe.
    """

    embedding_weight = t.tensor(
        [[0.0], [10.0], [0.0], [2.0]], dtype=t.float32
    )
    gradient = t.ones(3, 1)
    gradient[1, 0] = bad_value

    clamped = t.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)

    reps = rank_replacements_from_gradients(
        embedding_gradient=clamped,
        embedding_weight=embedding_weight,
        editable_positions=[1, 2],
        allowed_token_ids=[3],
        current_tokens=[0, 1, 2],
        top_k=1,
        distance_weight=0.0,
    )

    candidates = sample_candidate_sequences(
        current_tokens=[0, 1, 2],
        replacements_by_position=reps,
        batch_size=2,
        temperature=0.1,
        rng=random.Random(0),
    )

    assert candidates
    assert all(math.isfinite(candidate.proposal_score) for candidate in candidates)
