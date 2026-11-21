"""Stage 6 ``aggregate`` end-to-end integration tests.

Exercises :func:`steering.eval.axbench_base.aggregation.aggregate`
against synthetic generations + judge_scores directory-datasets
written with the real parquet writers and schemas. The math helpers
themselves are covered in ``tests/axbench_base/unit/test_aggregation_math.py``;
these tests pin the integration contract:

* ``report.yaml``, ``leaderboard.csv``, and ``aggregation_report.yaml``
  are all written to the paths resolved by :mod:`paths`.
* ``leaderboard.csv`` carries the expected columns.
* ``report.yaml`` has a top-level ``metadata`` section.
* Methods whose invalid-rate exceeds ``cfg.judge.invalid_drop_threshold``
  are removed from the leaderboard.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.aggregation import aggregate
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    JudgeConfig,
)
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import _JUDGE_SCORES_SCHEMA
from steering.eval.axbench_base.parquet_writer import ParquetAppendWriter
from tests.axbench_base import paths as test_paths

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_cfg(
    tmp_path: Path,
    *,
    run_tag: str = "stage6_int",
    invalid_warning_threshold: float = 0.01,
    invalid_drop_threshold: float = 0.5,
) -> AxBenchBaseConfig:
    judge_cfg = JudgeConfig(
        invalid_warning_threshold=invalid_warning_threshold,
        invalid_drop_threshold=invalid_drop_threshold,
    )
    return AxBenchBaseConfig(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        ),
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=str(tmp_path / "eval_out"),
        run_tag=run_tag,
        judge=judge_cfg,
    )


def _generation_row(
    *,
    generation_key: str,
    concept_id: int = 0,
    method_name: str = "diffmean_decoder_only",
    mode: str = "positive",
    eval_seed: int = 0,
    coefficient: str = "1.0",
    generation_text: str = "text",
    target_model: str = "decoder_only",
) -> dict[str, Any]:
    return {
        "generation_key": generation_key,
        "concept_id": concept_id,
        "method_name": method_name,
        "mode": mode,
        "eval_seed": eval_seed,
        "coefficient": coefficient,
        "generation_text": generation_text,
        "target_model": target_model,
        "prefix_text": "prefix",
        "prompt_token_key": "tokens_sha256:" + "0" * 64,
        "prompt_token_count": 4,
        "seed_pool_fingerprint": "f" * 64,
        "token_count": 16,
        "stop_reason": "bos_token",
    }


def _judge_row(
    *,
    generation_key: str,
    concept_id: int = 0,
    method_name: str = "diffmean_decoder_only",
    mode: str = "positive",
    eval_seed: int = 0,
    coefficient: str = "1.0",
    concept_score: int = 2,
    fluency_score: int = 2,
    concept_invalid: bool = False,
    fluency_invalid: bool = False,
) -> dict[str, Any]:
    return {
        "generation_key": generation_key,
        "concept_id": concept_id,
        "method_name": method_name,
        "mode": mode,
        "eval_seed": eval_seed,
        "coefficient": coefficient,
        "concept_score": concept_score,
        "concept_attempts": 1,
        "concept_invalid": concept_invalid,
        "fluency_score": fluency_score,
        "fluency_attempts": 1,
        "fluency_invalid": fluency_invalid,
        "judge_model": "openai/gpt-5-nano",
    }


def _pair(
    *,
    generation_key: str,
    method_name: str,
    mode: str,
    concept_id: int,
    eval_seed: int,
    coefficient: str,
    concept_score: int,
    fluency_score: int,
    concept_invalid: bool = False,
    fluency_invalid: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    gen = _generation_row(
        generation_key=generation_key,
        concept_id=concept_id,
        method_name=method_name,
        mode=mode,
        eval_seed=eval_seed,
        coefficient=coefficient,
    )
    jud = _judge_row(
        generation_key=generation_key,
        concept_id=concept_id,
        method_name=method_name,
        mode=mode,
        eval_seed=eval_seed,
        coefficient=coefficient,
        concept_score=concept_score,
        fluency_score=fluency_score,
        concept_invalid=concept_invalid,
        fluency_invalid=fluency_invalid,
    )
    return gen, jud


def _write_generations(run_tag: str, rows: list[dict[str, Any]]) -> Path:
    gen_dir = test_paths.default_paths(run_tag).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=100,
    ) as writer:
        for row in rows:
            writer.append(row)
    return gen_dir


def _write_judges(run_tag: str, rows: list[dict[str, Any]]) -> Path:
    judge_dir = test_paths.default_paths(run_tag).judge_scores_dir()
    judge_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=judge_dir,
        schema=_JUDGE_SCORES_SCHEMA,
        batch_size=100,
    ) as writer:
        for row in rows:
            writer.append(row)
    return judge_dir


def _build_positive_fixture(
    n_concepts: int = 3,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a small positive-mode fixture with two methods.

    * ``diffmean_decoder_only``: one row per concept at coef=1.0, scored 2/2.
    * ``ours_extract_and_add_encdec``: one row per concept at coef=None,
      scored 1/1.

    Produces ``2 * n_concepts`` rows.
    """
    gens: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []
    for i in range(n_concepts):
        g, j = _pair(
            generation_key=f"dm_c{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=2,
            fluency_score=2,
        )
        gens.append(g)
        judges.append(j)
    for i in range(n_concepts):
        g, j = _pair(
            generation_key=f"uu_c{i}",
            method_name="ours_extract_and_add_encdec",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=1,
            fluency_score=1,
        )
        gens.append(g)
        judges.append(j)
    return gens, judges


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_6_aggregate_from_synthetic_scores(
    tmp_run_root: Path,
) -> None:
    """End-to-end aggregate over a synthetic fixture writes all three
    reports and returns the path to ``report.yaml``."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage6_synthetic")
    assert cfg.run_tag is not None
    gens, judges = _build_positive_fixture(n_concepts=3)
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    report_path = aggregate(cfg)

    assert report_path == paths.AxBenchBasePaths.from_config(cfg).report_yaml()
    assert report_path.exists()
    assert paths.AxBenchBasePaths.from_config(cfg).leaderboard_csv().exists()
    assert paths.AxBenchBasePaths.from_config(cfg).aggregation_report().exists()

    with open(report_path) as f:
        report = yaml.safe_load(f)
    assert "metadata" in report
    assert "diffmean_decoder_only" in report
    assert "ours_extract_and_add_encdec" in report
    # diffmean concept+fluency=2/2 → hmean=2.0 for every concept.
    assert report["diffmean_decoder_only"]["mean"] == pytest.approx(2.0)


def test_stage_6_leaderboard_csv_structure(
    tmp_run_root: Path,
) -> None:
    """Leaderboard carries the spec columns in the expected order and
    is sorted by mean desc within mode."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage6_csv")
    assert cfg.run_tag is not None

    gens: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []
    # Leader: diffmean_decoder_only at 2.0.
    for i in range(3):
        g, j = _pair(
            generation_key=f"a_{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=2,
            fluency_score=2,
        )
        gens.append(g)
        judges.append(j)
    # Runner-up: linear_probe_decoder_only at 1.0.
    for i in range(3):
        g, j = _pair(
            generation_key=f"b_{i}",
            method_name="linear_probe_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=1,
            fluency_score=1,
        )
        gens.append(g)
        judges.append(j)
    # Control baseline so positive-mode delta lookup has something.
    for i in range(3):
        g, j = _pair(
            generation_key=f"u_{i}",
            method_name="ours_extract_and_add_encdec",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=1,
            fluency_score=1,
        )
        gens.append(g)
        judges.append(j)
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    aggregate(cfg)

    csv_path = paths.AxBenchBasePaths.from_config(cfg).leaderboard_csv()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    assert fieldnames == [
        "method_name",
        "mode",
        "mean",
        "ci_lower",
        "ci_upper",
        "n_valid_rows",
    ]
    # Positive-mode rows are sorted by mean desc.
    positive_rows = [r for r in rows if r["mode"] == "positive"]
    positive_means = [float(r["mean"]) for r in positive_rows]
    assert positive_means == sorted(positive_means, reverse=True)

    # Every expected method appears.
    method_names = {r["method_name"] for r in rows}
    assert "diffmean_decoder_only" in method_names
    assert "linear_probe_decoder_only" in method_names
    assert "ours_extract_and_add_encdec" in method_names


def test_stage_6_report_yaml_has_metadata_section(
    tmp_run_root: Path,
) -> None:
    """``report.yaml`` has a top-level ``metadata`` block with
    ``run_tag``, ``timestamp``, ``feature_dir``, and the expected
    integer counts."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage6_meta")
    assert cfg.run_tag is not None
    gens, judges = _build_positive_fixture(n_concepts=2)
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    aggregate(cfg)

    with open(paths.AxBenchBasePaths.from_config(cfg).report_yaml()) as f:
        report = yaml.safe_load(f)
    assert "metadata" in report
    meta = report["metadata"]
    assert meta["run_tag"] == cfg.run_tag
    assert "timestamp" in meta
    assert meta["feature_dir"] == cfg.feature_dir
    assert meta["n_concepts_total"] == 2
    assert meta["n_generations"] == 4
    assert meta["n_judge_scores"] == 4
    assert meta["n_joined"] == 4
    assert meta["n_invalid"] == 0
    assert meta["n_methods_included"] >= 2
    assert meta["n_methods_dropped"] == 0


def test_stage_6_method_dropped_on_high_invalid_rate(
    tmp_run_root: Path,
) -> None:
    """A method whose judge invalid-rate is >50% is dropped from the
    leaderboard. The baseline stays because its rate is 0."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage6_dropped",
        invalid_drop_threshold=0.5,
    )
    assert cfg.run_tag is not None

    gens: list[dict[str, Any]] = []
    judges: list[dict[str, Any]] = []

    # diffmean_decoder_only: 3 valid + 4 invalid → invalid_rate = 4/7 ≈ 0.57.
    for i in range(3):
        g, j = _pair(
            generation_key=f"dv_{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=2,
            fluency_score=2,
        )
        gens.append(g)
        judges.append(j)
    for i in range(4):
        g, j = _pair(
            generation_key=f"di_{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=i % 3,
            eval_seed=10 + i,
            coefficient="1.5",
            concept_score=-1,
            fluency_score=2,
            concept_invalid=True,
        )
        gens.append(g)
        judges.append(j)

    # linear_probe_decoder_only: all valid, 3 concepts, pristine.
    for i in range(3):
        g, j = _pair(
            generation_key=f"lp_{i}",
            method_name="linear_probe_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=2,
            fluency_score=2,
        )
        gens.append(g)
        judges.append(j)

    # Control baseline so positive-mode delta lookup has data.
    for i in range(3):
        g, j = _pair(
            generation_key=f"uu_{i}",
            method_name="ours_extract_and_add_encdec",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=1,
            fluency_score=1,
        )
        gens.append(g)
        judges.append(j)

    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    aggregate(cfg)

    with open(paths.AxBenchBasePaths.from_config(cfg).report_yaml()) as f:
        report = yaml.safe_load(f)

    # The >50%-invalid method is dropped from the leaderboard entries.
    assert "diffmean_decoder_only" not in report
    # The clean method is retained.
    assert "linear_probe_decoder_only" in report

    csv_path = paths.AxBenchBasePaths.from_config(cfg).leaderboard_csv()
    with open(csv_path) as f:
        csv_rows = list(csv.DictReader(f))
    csv_method_names = {r["method_name"] for r in csv_rows}
    assert "diffmean_decoder_only" not in csv_method_names
    assert "linear_probe_decoder_only" in csv_method_names

    diag_path = paths.AxBenchBasePaths.from_config(cfg).aggregation_report()
    with open(diag_path) as f:
        diag = yaml.safe_load(f)
    assert diag["n_methods_dropped"] >= 1
