from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

import scripts.eval.axbench_bpb_adv_eval as cli
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, GeneralValBpbConfig
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
)


def _candidate_table() -> pa.Table:
    prompt_tokens = [97, 98]
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": "c0",
            "attack_row_id": "r0",
            "core_prefix_id": "core0",
            "concept_rich_sample_id": "rich0",
            "concept_avoiding_sample_id": "avoid0",
            "concept_id": 1,
            "mode": "positive",
            "method_name": "diffmean_decoder_only",
            "target_model": "decoder_only",
            "coefficient_str": "1.0",
            "adversarial_method": "faster_gcg",
            "adversarial_strength": 0.0,
            "strength_unit": "optimized_tokens",
            "strength_index": 0,
            "context_signal_mode": "fixed_context",
            "prefix_text": "ab",
            "prefix_hash": "hash0",
            "source_prompt_tokens": prompt_tokens,
            "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
            "attacked_prompt_text": "ab",
            "attacked_prompt_tokens": prompt_tokens,
            "attacked_prompt_token_digest": token_sequence_digest(prompt_tokens),
            "concept_rich_output_tokens": [99],
            "concept_avoiding_output_tokens": [100],
            "concept_rich_output_digest": token_sequence_digest([99]),
            "concept_avoiding_output_digest": token_sequence_digest([100]),
            "editable_token_start": 2,
            "editable_token_count": 0,
            "replacement_tokens": [],
            "replacement_text": "",
            "objective_value": None,
            "candidate_metadata_json": "{}",
        }
    ]
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def _roundtrip_score_table() -> pa.Table:
    prompt_tokens = [97, 98]
    rows: list[dict[str, Any]] = [
        {
            "candidate_id": "c0",
            "core_prefix_id": "core0",
            "concept_id": 1,
            "mode": "positive",
            "method_name": "diffmean_decoder_only",
            "target_model": "decoder_only",
            "coefficient_str": "1.0",
            "adversarial_method": "faster_gcg",
            "adversarial_strength": 0.0,
            "strength_unit": "optimized_tokens",
            "strength_index": 0,
            "context_signal_mode": "fixed_context",
            "source_prompt_tokens": prompt_tokens,
            "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
            "attacked_prompt_tokens": prompt_tokens,
            "attacked_prompt_token_digest": token_sequence_digest(prompt_tokens),
            "roundtrip_prompt_tokens": prompt_tokens,
            "roundtrip_prompt_token_digest": token_sequence_digest(prompt_tokens),
            "roundtrip_prompt_text": "ab",
            "control_roundtrip_prompt_tokens": prompt_tokens,
            "control_roundtrip_prompt_token_digest": token_sequence_digest(prompt_tokens),
            "control_roundtrip_prompt_text": "ab",
            "editable_token_start": 2,
            "editable_token_count": 0,
            "replacement_tokens": [],
            "loss_concept_rich": 1.0,
            "loss_concept_avoiding": 3.0,
            "sample_index": 0,
            "reference_segment_token_count": 3,
            "generation_seed": 123,
            "generated_text": " generated",
            "n_generated_tokens": 3,
            "stop_reason": "max_tokens",
            "control_generation_seed": 124,
            "control_generated_text": " control",
            "control_n_generated_tokens": 2,
            "control_stop_reason": "eos",
            "is_control": True,
            "target_feature_id": 42,
            "n_roundtrip_features": 1,
            "roundtrip_pre_act": 8.0,
            "roundtrip_post_act": 4.0,
            "roundtrip_percentile": 80.0,
            "control_n_roundtrip_features": 1,
            "control_roundtrip_pre_act": 7.0,
            "control_roundtrip_post_act": 3.5,
            "control_roundtrip_percentile": 70.0,
        }
    ]
    return pa.Table.from_pylist(rows, schema=ROUNDTRIP_SCORE_SCHEMA)


def test_stage_score_default_roundtrip_writes_roundtrip_scores(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        target_models=["decoder_only"],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(_candidate_table(), paths.candidates_path())
    calls: list[Path] = []

    def _fake_roundtrip_score_stage(
        *,
        cfg: AxBenchBpbAdvConfig,
        paths: AxBenchBpbAdvPaths,
        candidates: pa.Table,
        stats: dict[str, int] | None = None,
    ) -> pa.Table:
        del cfg, stats
        calls.append(paths.roundtrip_scores_path())
        assert candidates.schema.equals(CANDIDATES_SCHEMA)
        return _roundtrip_score_table()

    monkeypatch.setattr(
        cli,
        "run_model_backed_roundtrip_score_stage",
        _fake_roundtrip_score_stage,
    )

    cli._stage_score(cfg)

    assert calls == [paths.roundtrip_scores_path()]
    assert paths.roundtrip_scores_path().exists()
    assert not paths.scores_path().exists()
    assert pq.read_table(paths.roundtrip_scores_path()).schema.equals(
        ROUNDTRIP_SCORE_SCHEMA
    )


def test_stage_aggregate_and_plot_default_roundtrip_use_roundtrip_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        target_models=["decoder_only"],
        scatter_baseline_x="min",
        plot_methods=["diffmean_decoder_only"],
        # This test asserts general_val_path is None (no general_val join);
        # general_val_bpb now defaults to enabled=True, so keep it off here.
        general_val_bpb=GeneralValBpbConfig(enabled=False),
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.root.mkdir(parents=True, exist_ok=True)
    pq.write_table(_roundtrip_score_table(), paths.roundtrip_scores_path())
    aggregate_calls: list[tuple[Path, Path]] = []
    plot_calls: list[tuple[Path, str, tuple[str, ...]]] = []

    def _fake_aggregate(
        score_path: Path,
        out_dir: Path,
        *,
        general_val_path: Path | None = None,
    ) -> pa.Table:
        assert general_val_path is None
        aggregate_calls.append((score_path, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        macro = pa.Table.from_pylist([], schema=ROUNDTRIP_MACRO_AGG_SCHEMA)
        pq.write_table(macro, out_dir / "macro.parquet")
        return macro

    def _fake_plot(macro: pa.Table, plots_dir: Path, **kwargs: Any) -> list[Path]:
        assert macro.schema.equals(ROUNDTRIP_MACRO_AGG_SCHEMA)
        plot_calls.append(
            (
                plots_dir,
                str(kwargs["scatter_baseline_x"]),
                tuple(kwargs["plot_methods"]),
            )
        )
        return []

    monkeypatch.setattr(cli, "aggregate_roundtrip_adv_scores", _fake_aggregate)
    monkeypatch.setattr(cli, "plot_roundtrip_adv_summaries", _fake_plot)

    cli._stage_aggregate(cfg)
    cli._stage_plot(cfg)

    assert aggregate_calls[0] == (
        paths.roundtrip_scores_path(),
        paths.roundtrip_aggregates_dir(),
    )
    assert plot_calls == [
        (paths.roundtrip_plots_dir(), "min", ("diffmean_decoder_only",))
    ]
