"""Unit tests for Stage 6 ``aggregate`` orchestration (Task 40).

Covers:

- Joining the Stage 4 generations directory-dataset with the Stage 5
  judge scores directory-dataset on ``generation_key``
- Per-row harmonic mean calculation (positive vs negative mode)
- Invalid-row filtering (``concept_invalid`` / ``fluency_invalid``)
- Per-method invalid-rate gating (warning + drop thresholds)
- Per-concept best-of-grid reduction across coefficients
- Per-method mean + bootstrap CI
- Control baseline lookup + delta computation for negative-mode methods
- report.yaml structure + metadata section
- leaderboard.csv schema + sort order
- aggregation_report.yaml diagnostic counts
- Error handling: missing run_tag, missing directories
- Determinism: same data → same CI

All tests construct synthetic generations + judge directories in
``tmp_path`` via :class:`ParquetAppendWriter`, then call the real
``aggregate()`` end-to-end. No mocking of math helpers.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from contextlib import contextmanager
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
import yaml
from loguru import logger

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.aggregation import aggregate
from steering.eval.axbench_base.config import AxBenchBaseConfig, JudgeConfig
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import _JUDGE_SCORES_SCHEMA
from steering.eval.axbench_base.parquet_writer import ParquetAppendWriter
from steering.eval.axbench_base.types import AxBenchBaseError
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@contextmanager
def _capture_loguru(level: str = "WARNING") -> Iterator[StringIO]:
    sink = StringIO()
    handler_id = logger.add(
        sink,
        format="{level} | {message}",
        level=level,
    )
    try:
        yield sink
    finally:
        logger.remove(handler_id)


def _set_base_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))


def _cfg(
    tmp_path: Path,
    *,
    run_tag: str | None = "t6",
    invalid_warning_threshold: float = 0.01,
    invalid_drop_threshold: float = 0.05,
) -> AxBenchBaseConfig:
    judge_cfg = JudgeConfig(
        invalid_warning_threshold=invalid_warning_threshold,
        invalid_drop_threshold=invalid_drop_threshold,
    )
    return make_valid_config(
        run_tag=run_tag,
        judge=judge_cfg,
    )


def _make_generation_row(
    *,
    generation_key: str,
    concept_id: int = 0,
    method_name: str = "diffmean_decoder_only",
    mode: str = "positive",
    eval_seed: int = 0,
    coefficient: str = "1.0",
    generation_text: str = "text",
    target_model: str = "decoder_only",
    prefix_text: str = "prefix",
    token_count: int = 16,
    stop_reason: str = "bos_token",
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
        "prefix_text": prefix_text,
        "prompt_token_key": "tokens_sha256:" + "0" * 64,
        "prompt_token_count": 4,
        "seed_pool_fingerprint": "f" * 64,
        "token_count": token_count,
        "stop_reason": stop_reason,
    }


def _make_judge_row(
    *,
    generation_key: str,
    concept_id: int = 0,
    method_name: str = "diffmean_decoder_only",
    mode: str = "positive",
    eval_seed: int = 0,
    coefficient: str = "1.0",
    concept_score: int = 2,
    concept_attempts: int = 1,
    concept_invalid: bool = False,
    fluency_score: int = 2,
    fluency_attempts: int = 1,
    fluency_invalid: bool = False,
    judge_model: str = "openai/gpt-5-nano",
) -> dict[str, Any]:
    return {
        "generation_key": generation_key,
        "concept_id": concept_id,
        "method_name": method_name,
        "mode": mode,
        "eval_seed": eval_seed,
        "coefficient": coefficient,
        "concept_score": concept_score,
        "concept_attempts": concept_attempts,
        "concept_invalid": concept_invalid,
        "fluency_score": fluency_score,
        "fluency_attempts": fluency_attempts,
        "fluency_invalid": fluency_invalid,
        "judge_model": judge_model,
    }


def _write_generations(run_tag: str, rows: list[dict[str, Any]]) -> Path:
    gen_dir = test_paths.default_paths(run_tag).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=10,
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
        batch_size=10,
    ) as writer:
        for row in rows:
            writer.append(row)
    return judge_dir


def _make_pair(
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
    gen = _make_generation_row(
        generation_key=generation_key,
        concept_id=concept_id,
        method_name=method_name,
        mode=mode,
        eval_seed=eval_seed,
        coefficient=coefficient,
    )
    jud = _make_judge_row(
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


# ===========================================================================
# Test 1: join generations + judges on generation_key
# ===========================================================================


def test_join_generations_and_judges(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """3 generations + 3 judges with matching keys → 3 joined rows."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    for i in range(3):
        g, j = _make_pair(
            generation_key=f"k{i}",
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
    # Add a second method so the aggregate covers multiple methods.
    for i in range(3):
        g, j = _make_pair(
            generation_key=f"u{i}",
            method_name="ours_extract_and_add_encdec",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=2,
            fluency_score=2,
        )
        gens.append(g)
        judges.append(j)
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    report_path = aggregate(cfg)
    assert report_path == paths.AxBenchBasePaths.from_config(cfg).report_yaml()
    assert report_path.exists()

    agg_report_path = paths.AxBenchBasePaths.from_config(cfg).aggregation_report()
    assert agg_report_path.exists()
    with open(agg_report_path) as f:
        diag = yaml.safe_load(f)
    assert diag["n_generations"] == 6
    assert diag["n_judge_scores"] == 6
    assert diag["n_joined"] == 6
    assert diag["n_invalid"] == 0


# ===========================================================================
# Test 2: row score in positive mode
# ===========================================================================


def test_row_score_positive_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Positive mode: concept=2, fluency=2 → hmean=2.0, method_mean=2.0."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    for i in range(4):
        g, j = _make_pair(
            generation_key=f"m{i}",
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
        g2, j2 = _make_pair(
            generation_key=f"u{i}",
            method_name="ours_extract_and_add_encdec",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=1,
            fluency_score=1,
        )
        gens.append(g2)
        judges.append(j2)
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    aggregate(cfg)
    with open(paths.AxBenchBasePaths.from_config(cfg).report_yaml()) as f:
        report = yaml.safe_load(f)
    assert report["diffmean_decoder_only"]["mean"] == pytest.approx(2.0)
    assert report["ours_extract_and_add_encdec"]["mean"] == pytest.approx(1.0)


# ===========================================================================
# Test 3: row score in negative mode (axis flip)
# ===========================================================================


def test_row_score_negative_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Negative mode: concept=0 → hmean(2-0, fluency) = hmean(2, fluency)."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    # Negative method — concept_score=0 means "no concept leak", ideal.
    for i in range(4):
        g, j = _make_pair(
            generation_key=f"n{i}",
            method_name="neg_diffmean_decoder_only",
            mode="negative",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=0,
            fluency_score=2,
        )
        gens.append(g)
        judges.append(j)
        # Control baseline with concept_score=2 (worst on neg axis) → flipped=0 → hmean=0.
        g2, j2 = _make_pair(
            generation_key=f"nu{i}",
            method_name="neg_ours_extract_and_add_encdec",
            mode="negative",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=2,
            fluency_score=2,
        )
        gens.append(g2)
        judges.append(j2)
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    aggregate(cfg)
    with open(paths.AxBenchBasePaths.from_config(cfg).report_yaml()) as f:
        report = yaml.safe_load(f)
    # Negative axis: (2 - 0)=2 → hmean(2, 2) = 2.0
    assert report["neg_diffmean_decoder_only"]["mean"] == pytest.approx(2.0)
    # Control negative: (2 - 2)=0 → hmean(0, 2) = 0.0
    assert report["neg_ours_extract_and_add_encdec"]["mean"] == pytest.approx(0.0)


# ===========================================================================
# Test 4: invalid rows dropped
# ===========================================================================


def test_invalid_rows_excluded(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``concept_invalid=True`` rows are dropped from the aggregate."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, invalid_warning_threshold=0.99, invalid_drop_threshold=0.999)
    assert cfg.run_tag is not None

    gens, judges = [], []
    # 5 rows for method A — 4 valid (score 2,2) + 1 invalid.
    for i in range(5):
        invalid = (i == 0)
        g, j = _make_pair(
            generation_key=f"a{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=-1 if invalid else 2,
            fluency_score=2,
            concept_invalid=invalid,
            fluency_invalid=False,
        )
        gens.append(g)
        judges.append(j)
    # Control baseline
    for i in range(5):
        g, j = _make_pair(
            generation_key=f"u{i}",
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
    entry = report["diffmean_decoder_only"]
    assert entry["n_valid_rows"] == 4
    assert entry["n_invalid_rows"] == 1
    assert entry["n_concepts"] == 4  # invalid row's concept was dropped
    assert entry["invalid_rate"] == pytest.approx(1 / 5)

    diag_path = paths.AxBenchBasePaths.from_config(cfg).aggregation_report()
    with open(diag_path) as f:
        diag = yaml.safe_load(f)
    assert diag["n_invalid"] == 1


# ===========================================================================
# Test 5: drop threshold excludes a method from the leaderboard
# ===========================================================================


def test_method_dropped_when_invalid_rate_exceeds_threshold(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """> invalid_drop_threshold → method never appears in leaderboard."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(
        tmp_path,
        invalid_warning_threshold=0.001,
        invalid_drop_threshold=0.05,
    )
    assert cfg.run_tag is not None

    gens, judges = [], []
    # 10 rows for method A — 5 invalid → 50% invalid rate.
    for i in range(10):
        invalid = (i < 5)
        g, j = _make_pair(
            generation_key=f"bad{i}",
            method_name="lora_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="None",
            concept_score=-1 if invalid else 2,
            fluency_score=2,
            concept_invalid=invalid,
        )
        gens.append(g)
        judges.append(j)
    # A clean method for contrast
    for i in range(5):
        g, j = _make_pair(
            generation_key=f"clean{i}",
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
    # Control baseline
    for i in range(5):
        g, j = _make_pair(
            generation_key=f"u{i}",
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
    assert "lora_decoder_only" not in report
    assert "diffmean_decoder_only" in report

    csv_path = paths.AxBenchBasePaths.from_config(cfg).leaderboard_csv()
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        names = [row["method_name"] for row in reader]
    assert "lora_decoder_only" not in names
    assert "diffmean_decoder_only" in names

    with open(paths.AxBenchBasePaths.from_config(cfg).aggregation_report()) as f:
        diag = yaml.safe_load(f)
    assert diag["n_methods_dropped"] == 1
    assert diag["n_methods_included"] >= 1


# ===========================================================================
# Test 6: warning threshold logs but keeps method
# ===========================================================================


def test_warning_logged_at_1_percent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """> invalid_warning_threshold but <= drop_threshold → warning + kept."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(
        tmp_path,
        invalid_warning_threshold=0.01,
        invalid_drop_threshold=0.99,
    )
    assert cfg.run_tag is not None

    gens, judges = [], []
    # 20 rows — 1 invalid → 5% rate, above 1% warning, below 99% drop
    for i in range(20):
        invalid = (i == 0)
        g, j = _make_pair(
            generation_key=f"w{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=i,
            eval_seed=0,
            coefficient="1.0",
            concept_score=-1 if invalid else 2,
            fluency_score=2,
            concept_invalid=invalid,
        )
        gens.append(g)
        judges.append(j)
    for i in range(20):
        g, j = _make_pair(
            generation_key=f"u{i}",
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

    with _capture_loguru(level="WARNING") as sink:
        aggregate(cfg)
    captured = sink.getvalue()
    assert "WARNING" in captured
    assert "invalid" in captured.lower()
    assert "diffmean_decoder_only" in captured

    with open(paths.AxBenchBasePaths.from_config(cfg).report_yaml()) as f:
        report = yaml.safe_load(f)
    assert "diffmean_decoder_only" in report


# ===========================================================================
# Test 7: per-concept best-of-grid
# ===========================================================================


def test_per_concept_best_of_grid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Multiple coefficients per concept → max wins."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, invalid_drop_threshold=0.99)
    assert cfg.run_tag is not None

    gens, judges = [], []
    # 1 concept, 3 coefficients — only the best (2,2) should count.
    for i, (cs, fs, coef) in enumerate([(1, 1, "0.5"), (2, 2, "1.5"), (0, 2, "3.0")]):
        g, j = _make_pair(
            generation_key=f"m{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=0,
            eval_seed=0,
            coefficient=coef,
            concept_score=cs,
            fluency_score=fs,
        )
        gens.append(g)
        judges.append(j)
    # Control baseline (1 concept, 1 row)
    g, j = _make_pair(
        generation_key="uu",
        method_name="ours_extract_and_add_encdec",
        mode="positive",
        concept_id=0,
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
    # Best of [hmean(1,1)=1, hmean(2,2)=2, hmean(0,2)=0] = 2
    assert report["diffmean_decoder_only"]["mean"] == pytest.approx(2.0)
    assert report["diffmean_decoder_only"]["n_concepts"] == 1


# ===========================================================================
# Test 8: leaderboard CSV has spec columns
# ===========================================================================


def test_leaderboard_csv_structure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Leaderboard has required columns, sorted by mean desc within mode."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    # Positive-mode leader: method A (mean=2), method B (mean=1)
    for i in range(3):
        g, j = _make_pair(
            generation_key=f"a{i}",
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
    for i in range(3):
        g, j = _make_pair(
            generation_key=f"b{i}",
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
    for i in range(3):
        g, j = _make_pair(
            generation_key=f"u{i}",
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
    assert reader.fieldnames == [
        "method_name",
        "mode",
        "mean",
        "ci_lower",
        "ci_upper",
        "n_valid_rows",
    ]
    # Sort check: diffmean (mean=2) before linear_probe (mean=1) within positive
    positive_rows = [r for r in rows if r["mode"] == "positive"]
    means = [float(r["mean"]) for r in positive_rows]
    assert means == sorted(means, reverse=True)
# ===========================================================================
# Test 9: report.yaml has metadata section
# ===========================================================================


def test_report_yaml_has_metadata_section(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    for i in range(2):
        g, j = _make_pair(
            generation_key=f"m{i}",
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
    for i in range(2):
        g, j = _make_pair(
            generation_key=f"u{i}",
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
    assert "metadata" in report
    meta = report["metadata"]
    assert meta["run_tag"] == cfg.run_tag
    assert "timestamp" in meta
    assert meta["feature_dir"] == cfg.feature_dir
    assert meta["n_concepts_total"] == 2


# ===========================================================================
# Test 10: empty dirs raise
# ===========================================================================


def test_empty_dirs_raise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Missing generations dir → clear error.

    Missing judge_scores dir → clear error.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    # No dirs created at all → generations-missing error.
    with pytest.raises(AxBenchBaseError, match="generations"):
        aggregate(cfg)

    # Create the generations dir but no judge_scores dir.
    gens = [
        _make_generation_row(
            generation_key="m0",
            concept_id=0,
        ),
    ]
    _write_generations(cfg.run_tag, gens)
    with pytest.raises(AxBenchBaseError, match="judge"):
        aggregate(cfg)


# ===========================================================================
# Test 12: run_tag not set raises
# ===========================================================================


def test_run_tag_not_set_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path, run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        aggregate(cfg)


# ===========================================================================
# Test 13: mean score matches manual calc
# ===========================================================================


def test_mean_score_matches_manual_calc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Hand-verified two-concept aggregate.

    Concept 0: best of [hmean(1,2)=4/3, hmean(2,1)=4/3] = 4/3
    Concept 1: best of [hmean(2,2)=2]              = 2
    mean = (4/3 + 2) / 2 = 5/3
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    # concept 0 — two coefficients
    for i, (cs, fs) in enumerate([(1, 2), (2, 1)]):
        g, j = _make_pair(
            generation_key=f"c0_{i}",
            method_name="diffmean_decoder_only",
            mode="positive",
            concept_id=0,
            eval_seed=0,
            coefficient=f"{i}",
            concept_score=cs,
            fluency_score=fs,
        )
        gens.append(g)
        judges.append(j)
    # concept 1 — one coefficient
    g, j = _make_pair(
        generation_key="c1_0",
        method_name="diffmean_decoder_only",
        mode="positive",
        concept_id=1,
        eval_seed=0,
        coefficient="0",
        concept_score=2,
        fluency_score=2,
    )
    gens.append(g)
    judges.append(j)
    # Control for both
    for cid in (0, 1):
        g, j = _make_pair(
            generation_key=f"u{cid}",
            method_name="ours_extract_and_add_encdec",
            mode="positive",
            concept_id=cid,
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
    assert report["diffmean_decoder_only"]["mean"] == pytest.approx(5 / 3)
    assert report["diffmean_decoder_only"]["n_concepts"] == 2


# ===========================================================================
# Test 14: bootstrap CI deterministic across runs
# ===========================================================================


def test_missing_judge_coverage_raises_before_writing_reports(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Three generations plus two judge rows fail before any report write."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _cfg(tmp_path)
    assert cfg.run_tag is not None

    gens, judges = [], []
    for i in range(3):
        g, j = _make_pair(
            generation_key=f"m{i}",
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
    judges.pop()
    _write_generations(cfg.run_tag, gens)
    _write_judges(cfg.run_tag, judges)

    report_paths = (
        paths.AxBenchBasePaths.from_config(cfg).report_yaml(),
        paths.AxBenchBasePaths.from_config(cfg).leaderboard_csv(),
        paths.AxBenchBasePaths.from_config(cfg).aggregation_report(),
    )
    before: dict[Path, bytes] = {}
    for index, report_path in enumerate(report_paths):
        payload = f"pre-existing-report-{index}\n".encode()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_bytes(payload)
        before[report_path] = payload

    with pytest.raises(AxBenchBaseError) as excinfo:
        aggregate(cfg)

    message = str(excinfo.value)
    assert "3 generation rows" in message
    assert "2 judge rows" in message
    assert "1 missing" in message
    assert "m2" in message
    assert {path: path.read_bytes() for path in report_paths} == before


def test_bootstrap_ci_deterministic_across_runs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Same data → same CI. Two separate aggregate() calls on identical
    fixtures (with output dirs cleared between) must produce bit-identical
    CI values."""
    _set_base_dir(monkeypatch, tmp_path)

    def _run(run_tag: str) -> tuple[float, float]:
        cfg = _cfg(tmp_path, run_tag=run_tag)
        gens, judges = [], []
        for i in range(10):
            g, j = _make_pair(
                generation_key=f"{run_tag}_m{i}",
                method_name="diffmean_decoder_only",
                mode="positive",
                concept_id=i,
                eval_seed=0,
                coefficient="1.0",
                concept_score=(i % 2) + 1,
                fluency_score=2,
            )
            gens.append(g)
            judges.append(j)
        for i in range(10):
            g, j = _make_pair(
                generation_key=f"{run_tag}_u{i}",
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
        _write_generations(run_tag, gens)
        _write_judges(run_tag, judges)
        aggregate(cfg)
        with open(test_paths.default_paths(run_tag).report_yaml()) as f:
            report = yaml.safe_load(f)
        entry = report["diffmean_decoder_only"]
        return float(entry["ci_lower"]), float(entry["ci_upper"])

    a = _run("t_a")
    b = _run("t_b")
    assert a == b
