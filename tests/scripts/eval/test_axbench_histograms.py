"""Numerical, file-boundary, rendering, and real-CLI histogram contracts."""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from omegaconf import OmegaConf

from scripts.eval import plot_axbench_histograms as cli
from steering.eval import axbench_histograms as hist
from steering.eval.axbench_adv.best_point import per_concept_ratings
from steering.eval.axbench_adv.types import PLOT_EFFORT_LEVELS

ROOT = Path(__file__).resolve().parents[3]
FACTORS = {"DiffMean": 2.0, "SelfConditioning": 1.0, "LoRA": 1.0}
ADV_FACTORS = {
    "DiffMean": 2.0,
    "LinearProbe": 3.0,
    "LoRA": 1.0,
    "PromptSteering": 1.0,
    "SelfConditioning": 6.0,
    "SelfConditioningMasked": 5.0,
}


def _write(directory: Path, entries: list[dict], frame: pd.DataFrame) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "steering.jsonl").write_text("\n".join(json.dumps(e) for e in entries) + "\n")
    frame.to_parquet(directory / "steering_data.parquet", index=False)


def _fixture(
    directory: Path,
    *,
    adv: bool = False,
    concepts: int = 3,
    shift: int = 0,
    upstream_grids: dict[str, tuple[float, ...]] | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    grids = (
        {m: (f,) for m, f in ADV_FACTORS.items()}
        if adv
        else upstream_grids or {
            "DiffMean": (1.0, 2.0),
            "SelfConditioning": (1.0, 2.0),
            "LoRA": (1.0,),
        }
    )
    rows = []
    for concept in range(concepts):
        for input_id in range(10):
            for factor in sorted({v for grid in grids.values() for v in grid}):
                row: dict[str, Any] = {
                    "concept_id": concept,
                    "input_id": input_id,
                    "factor": factor,
                    "input_concept": f"Concept {concept} — Unicode λ",
                    "original_prompt": f"Prompt {concept}/{input_id}",
                    "dataset_name": "AlpacaEval",
                }
                for m_index, (method, grid) in enumerate(grids.items()):
                    for a_index, field in enumerate(hist.AXES):
                        row[f"{method}_LMJudgeEvaluator_{field}"] = (
                            ((a_index + concept + input_id % 2 + m_index + int(factor) + shift) % 5)
                            / 2
                            if factor in grid
                            else None
                        )
                rows.append(row)
    frame = pd.DataFrame(rows)
    entries = []
    for concept in range(concepts):
        selected = frame.loc[frame.concept_id.eq(concept)]
        methods = {}
        for method, grid in grids.items():
            block: dict[str, Any] = {"factor": list(grid)}
            for field in hist.AXES:
                column = f"{method}_LMJudgeEvaluator_{field}"
                block[field] = [
                    float(selected.loc[selected.factor.eq(f), column].mean()) for f in grid
                ]
                block[f"raw_{field}"] = [None if pd.isna(v) else v for v in selected[column]]
            methods[method] = block
        entries.append({"concept_id": concept, "results": {"LMJudgeEvaluator": methods}})
    _write(directory, entries, frame)
    return entries, frame


def _adv_fixture(directory: Path) -> None:
    for level in PLOT_EFFORT_LEVELS:
        _fixture(directory / "points" / f"p{level}" / "judged", adv=True, shift=level)
    manifest = {
        "schema_version": 2,
        "max_concepts": 3,
        "best_factors": ADV_FACTORS,
        "files": {
            str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in directory.rglob("*")
            if p.is_file()
        },
    }
    (directory / "manifest.json").write_text(json.dumps(manifest))


def _run(source: Path, output: Path, **kwargs: Any) -> Path:
    return hist.generate_histogram_reports(
        input_dir=source, output_dir=output, pipeline="upstream", method_factors=FACTORS, **kwargs
    )


def test_saved_means_exactly_preserved_and_match_existing_extractor(tmp_path: Path) -> None:
    source = tmp_path / "input"
    entries, _ = _fixture(source)
    scores, population = hist._load_scores(source, FACTORS, {})
    assert len(population) == 30
    assert set(scores.n_prompts) == {10}
    for method, factor in FACTORS.items():
        concept, fluency = per_concept_ratings(entries, method, factor)
        for field, expected in zip(tuple(hist.AXES)[:2], (concept, fluency), strict=True):
            assert (
                scores.loc[scores.method.eq(method) & scores.axis.eq(field), "score"].tolist()
                == expected
            )
    assert scores.loc[
        scores.method.eq("DiffMean") & scores.axis.eq("relevance_concept_ratings"), "score"
    ].tolist() == [1.25, 1.75, 1.0]


@pytest.mark.parametrize("bins", [1, 2, 10, 100])
def test_histogram_exact_edges_and_adjacent_floats(bins: int) -> None:
    edges = np.linspace(0, 2, bins + 1)
    values = [0.0, 2.0, 0.125, 1.75]
    for edge in edges[1:-1]:
        values.extend([float(np.nextafter(edge, 0)), float(edge), float(np.nextafter(edge, 2))])
    expected = [0] * bins
    for value in values:
        expected[min(bisect.bisect_right(edges, value) - 1, bins - 1)] += 1
    counts, percentages = hist._histogram(values, edges)
    assert counts == expected
    assert sum(counts) == len(values)
    assert percentages == [n * (100.0 / len(values)) for n in expected]
    assert sum(percentages) == pytest.approx(100)


def test_complete_report_artist_data_stats_hashes_and_repeatability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    _fixture(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    snapshots = []
    original = Figure.savefig

    def save(figure: Figure, *args: Any, **kwargs: Any) -> None:
        snapshots.append(
            [
                (
                    a.get_title(),
                        a.get_xlim(),
                        a.get_ylim(),
                        [
                            [p.get_x(), p.get_width(), p.get_height()]
                            for p in cast(list[Rectangle], a.patches)
                        ],
                        a.get_ylabel(),
                        [text.get_text() for text in a.texts],
                )
                for a in figure.axes
            ]
        )
        original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    path = _run(source, tmp_path / "report")
    metadata = json.loads(path.read_text())
    assert len(snapshots[0]) == len(FACTORS) * len(hist.AXES)
    assert [p[0] for p in snapshots[0][:3]] == [
        "Concept relevance", "Fluency", "Instruction-following"
    ]
    assert [snapshots[0][row * len(hist.AXES)][4] for row in range(len(FACTORS))] == [
        f"{method} (factor {FACTORS[method]:g})\nConcepts (%)"
        for method in sorted(FACTORS)
    ]
    assert all(p[1:3] == ((0.0, 2.0), (0.0, 100.0)) for p in snapshots[0])
    for axis_index, field in enumerate(hist.AXES):
        summaries = [r for r in metadata["reports"]["upstream"] if r["axis"] == field]
        assert [p[2] for p in snapshots[0][axis_index][3]] == summaries[0]["percentages"]
        for summary in summaries:
            assert sum(summary["counts"]) == 3
            assert summary["n_prompts_per_concept"] == [10] * 3
    summary = metadata["reports"]["upstream"][0]
    assert summary["mean"] == 4 / 3
    assert summary["sample_std"] == pytest.approx(np.std([1.25, 1.75, 1], ddof=1))
    assert summary["sample_variance"] == pytest.approx(np.var([1.25, 1.75, 1], ddof=1))
    assert snapshots[0][0][5] == ["mean 1.33  var 0.146"]
    assert metadata["bins"] == 20
    summary_csv = pd.read_csv(path.parent / "summary.csv")
    assert "sample_variance" in summary_csv.columns
    assert metadata["layout"] == {"rows": "method", "columns": list(hist.AXES)}
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    for name, content in before.items():
        assert metadata["sources"][str(source / name)] == hashlib.sha256(content).hexdigest()
    second = _run(source, tmp_path / "report2")
    assert second.read_bytes() == path.read_bytes()
    for name in ("concept_scores.csv", "summary.csv", "score_histograms.png"):
        assert (second.parent / name).read_bytes() == (path.parent / name).read_bytes()


def test_single_concept_and_raw_suppression_semantics(tmp_path: Path) -> None:
    source = tmp_path / "input"
    entries, frame = _fixture(source, concepts=1)
    frame["dataset_name"] = "AlpacaEvalSuppress"
    _write(source, entries, frame)
    meta = json.loads(_run(source, tmp_path / "out").read_text())
    assert meta["dataset"] == "AlpacaEvalSuppress"
    assert all(r["sample_std"] is None for r in meta["reports"]["upstream"])
    assert all(r["sample_variance"] is None for r in meta["reports"]["upstream"])
    assert meta["reports"]["upstream"][0]["mean"] == 1.25


@pytest.mark.parametrize(
    "case",
    [
        "empty",
        "entry_shape",
        "duplicate_concept",
        "boolean_concept",
        "missing_concept",
        "missing_column",
        "null_identity",
        "float_id",
        "blank_prompt",
        "boolean_factor",
        "duplicate_row",
        "extra_concept",
        "mixed_dataset",
        "changed_concept",
        "changed_prompt",
        "missing_input",
        "missing_block",
        "bad_results",
        "bad_factors",
        "duplicate_factor",
        "absent_factor",
        "score_length",
        "missing_score",
        "raw_length",
        "raw_disagreement",
        "mean_disagreement",
        "boolean_score",
        "null_score",
        "nan_score",
        "inf_score",
        "range_score",
    ],
)
def test_invalid_populations_and_scores_fail_before_output(tmp_path: Path, case: str) -> None:
    source = tmp_path / "input"
    entries, frame = _fixture(source)
    block = entries[0]["results"]["LMJudgeEvaluator"]["DiffMean"]
    field = "relevance_concept_ratings"
    column = f"DiffMean_LMJudgeEvaluator_{field}"
    match case:
        case "empty":
            entries = []
        case "entry_shape":
            entries[0]["extra"] = True
        case "duplicate_concept":
            entries[1]["concept_id"] = 0
        case "boolean_concept":
            entries[0]["concept_id"] = False
        case "missing_concept":
            entries.pop()
        case "missing_column":
            frame = frame.drop(columns="input_id")
        case "null_identity":
            frame.loc[0, "original_prompt"] = None
        case "float_id":
            frame["concept_id"] = frame.concept_id.astype(float)
        case "blank_prompt":
            frame.loc[0, "original_prompt"] = " "
        case "boolean_factor":
            frame["factor"] = True
        case "duplicate_row":
            frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        case "extra_concept":
            frame.loc[0, "concept_id"] = 99
        case "mixed_dataset":
            frame.loc[0, "dataset_name"] = "other"
        case "changed_concept":
            frame.loc[0, "input_concept"] = "other"
        case "changed_prompt":
            frame.loc[0, "original_prompt"] = "other"
        case "missing_input":
            frame = frame.loc[~(frame.concept_id.eq(0) & frame.input_id.eq(9))]
        case "missing_block":
            del entries[0]["results"]["LMJudgeEvaluator"]["DiffMean"]
        case "bad_results":
            entries[0]["results"] = []
        case "bad_factors":
            block["factor"] = "wrong"
        case "duplicate_factor":
            block["factor"] = [2, 2]
        case "absent_factor":
            block["factor"] = [1, 3]
        case "score_length":
            block[field] = [1]
        case "missing_score":
            frame = frame.drop(columns=column)
        case "raw_length":
            block[f"raw_{field}"] = []
        case "raw_disagreement":
            block[f"raw_{field}"][1] = 0
        case "mean_disagreement":
            block[field][1] = 0
        case "boolean_score":
            block[field][1] = True
        case "null_score":
            frame.loc[1, column] = None
        case "nan_score":
            block[field][1] = float("nan")
        case "inf_score":
            block[field][1] = float("inf")
        case "range_score":
            block[field][1] = 2.1
    _write(source, entries, frame)
    with pytest.raises(ValueError):
        _run(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("content", ["{", '{"concept_id":0,"concept_id":1,"results":{}}'])
def test_bad_json(tmp_path: Path, content: str) -> None:
    source = tmp_path / "input"
    _fixture(source)
    (source / "steering.jsonl").write_text(content)
    with pytest.raises(ValueError):
        _run(source, tmp_path / "out")


@pytest.mark.parametrize(
    "factors", [{}, {"unknown": 1}, {"LoRA": 2}, {"DiffMean": True}, {"DiffMean": float("inf")}]
)
def test_invalid_selection(tmp_path: Path, factors: dict) -> None:
    with pytest.raises(ValueError):
        hist.generate_histogram_reports(
            input_dir=tmp_path / "in",
            output_dir=tmp_path / "out",
            pipeline="upstream",
            method_factors=factors,
        )


@pytest.mark.parametrize("bins", [0, 101, True, 2.5])
def test_invalid_bins(tmp_path: Path, bins: Any) -> None:
    with pytest.raises(ValueError, match="bins"):
        _run(tmp_path / "input", tmp_path / "output", bins=bins)


@pytest.mark.parametrize("pipeline,factors", [("invalid", {}), ("adv", None)])
def test_invalid_pipeline_and_map(tmp_path: Path, pipeline: str, factors: Any) -> None:
    with pytest.raises(ValueError):
        hist.generate_histogram_reports(
            input_dir=tmp_path / "in",
            output_dir=tmp_path / "out",
            pipeline=pipeline,
            method_factors=factors,
        )


@pytest.mark.parametrize("case", ["valid", "extra", "null_path", "missing", "blank_pipeline"])
def test_cli_configuration_boundary(tmp_path: Path, case: str) -> None:
    source = tmp_path / "input"
    _fixture(source)
    values: dict[str, Any] = {
        "input_dir": str(source),
        "output_dir": str(tmp_path / "out"),
        "pipeline": "upstream",
        "method_factors": FACTORS,
        "bins": 10,
    }
    if case == "extra":
        values["extra"] = 1
    elif case == "null_path":
        values["input_dir"] = None
    elif case == "missing":
        del values["bins"]
    elif case == "blank_pipeline":
        values["pipeline"] = " "
    cfg = OmegaConf.create(values)
    if case == "valid":
        cli.main.__wrapped__(cfg)
        assert (tmp_path / "out/metadata.json").is_file()
    else:
        with pytest.raises(ValueError):
            cli.main.__wrapped__(cfg)
        assert not (tmp_path / "out").exists()


def test_source_reordering_with_aligned_raw_json_is_semantically_identical(tmp_path: Path) -> None:
    source = tmp_path / "input"
    entries, frame = _fixture(source)
    expected, _ = hist._load_scores(source, FACTORS, {})
    frame = frame.iloc[::-1].reset_index(drop=True)
    for entry in entries:
        for block in entry["results"]["LMJudgeEvaluator"].values():
            for field in hist.AXES:
                block[f"raw_{field}"] = block[f"raw_{field}"][::-1]
    _write(source, entries, frame)
    observed, _ = hist._load_scores(source, FACTORS, {})
    pd.testing.assert_frame_equal(observed, expected)


def test_missing_judged_file_does_not_write_and_palette_covers_registry(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _run(tmp_path / "absent", tmp_path / "out")
    assert not (tmp_path / "out").exists()
    assert set(hist.COLORS) == set(hist.METHOD_FACTORS)


def test_optional_nl_report_preserves_default_palette_and_has_fixed_style(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "input"
    grids = {
        "DiffMean": (1.0, 2.0),
        "SelfConditioning": (1.0, 2.0),
        "LoRA": (1.0,),
        "NLSteering": (1.0, 2.0),
    }
    _fixture(source, upstream_grids=grids)
    observed: dict[str, tuple[object, object]] = {}
    original = Figure.savefig

    def capture(figure: Figure, *args: Any, **kwargs: Any) -> None:
        for axis in figure.axes:
            for patch in axis.patches:
                method = str(patch.get_label()).split(" (factor", 1)[0]
                observed[method] = (patch.get_edgecolor(), patch.get_alpha())
        original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    path = hist.generate_histogram_reports(
        input_dir=source,
        output_dir=tmp_path / "out",
        pipeline="upstream",
        method_factors={method: factors[0] for method, factors in grids.items()},
    )

    metadata = json.loads(path.read_text())
    assert metadata["method_factors"]["NLSteering"] == 1.0
    assert len(metadata["reports"]["upstream"]) == len(grids) * len(hist.AXES)
    assert observed["NLSteering"][1] == pytest.approx(0.4)
    assert all(observed["NLSteering"][0] != to_rgba(color) for color in hist.COLORS.values())
    assert set(hist.COLORS) == set(hist.METHOD_FACTORS)


def test_output_collisions_and_failure_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "input"
    _fixture(source)
    alias = tmp_path / "alias"
    alias.symlink_to(source, target_is_directory=True)
    for output in (source, source / "plots", tmp_path, alias):
        with pytest.raises(ValueError, match="path-disjoint"):
            _run(source, output)
    output = tmp_path / "out"
    output.mkdir()
    (output / "keep").write_text("untouched")
    with pytest.raises(ValueError, match="new or empty"):
        _run(source, output)
    assert (output / "keep").read_text() == "untouched"
    cleared = []
    original = Figure.clear

    def clear(figure: Figure, *args: Any, **kwargs: Any) -> None:
        cleared.append(True)
        original(figure, *args, **kwargs)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(Figure, "clear", clear)
    monkeypatch.setattr(Figure, "savefig", fail)
    with pytest.raises(OSError, match="disk full"):
        _run(source, tmp_path / "failed")
    assert cleared
    assert not (tmp_path / "failed" / "metadata.json").exists()


def test_adv_all_levels_use_manifest_factors_and_regenerated_control(tmp_path: Path) -> None:
    source = tmp_path / "adv"
    _adv_fixture(source)
    path = hist.generate_histogram_reports(
        input_dir=source, output_dir=tmp_path / "out", pipeline="adv", method_factors={}
    )
    meta = json.loads(path.read_text())
    assert meta["method_factors"] == ADV_FACTORS
    assert set(meta["reports"]) == {f"p{v}" for v in PLOT_EFFORT_LEVELS}
    assert meta["reports"]["p0"] != meta["reports"]["p1"]
    for level in PLOT_EFFORT_LEVELS:
        assert (path.parent / f"p{level}" / "score_histograms.png").is_file()
        scores = pd.read_csv(path.parent / f"p{level}" / "concept_scores.csv")
        assert len(scores) == 6 * 3 * 3


@pytest.mark.parametrize("concept_ids", [
    None, [], [True, 1, 2], [0.0, 1, 2], [0, 0, 2], [2, 1, 0],
    [-1, 0, 1], [0, 1, 3], "0,1,2", [1, 2],
])
def test_adv_rejects_invalid_or_incomplete_declared_population(
    tmp_path: Path, concept_ids: object,
) -> None:
    source = tmp_path / "adv"
    _adv_fixture(source)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["concept_ids"] = concept_ids
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="concept_ids|expected ascending population"):
        hist.generate_histogram_reports(
            input_dir=source, output_dir=tmp_path / "out", pipeline="adv",
            method_factors={},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("level", [0, 8])
def test_adv_rejects_same_size_wrong_concept_ids_at_any_level(tmp_path: Path, level: int) -> None:
    source = tmp_path / "adv"
    _adv_fixture(source)
    directory = source / "points" / f"p{level}" / "judged"
    entries = [json.loads(line) for line in (directory / "steering.jsonl").read_text().splitlines()]
    frame = pd.read_parquet(directory / "steering_data.parquet")
    for entry in entries:
        entry["concept_id"] += 1
    frame["concept_id"] += 1
    _write(directory, entries, frame)
    path = source / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["max_concepts"] = 4
    manifest["concept_ids"] = [0, 1, 2]
    for name in ("steering.jsonl", "steering_data.parquet"):
        file = directory / name
        manifest["files"][str(file.relative_to(source))] = hashlib.sha256(file.read_bytes()).hexdigest()
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="expected ascending population"):
        hist.generate_histogram_reports(
            input_dir=source, output_dir=tmp_path / "out", pipeline="adv",
            method_factors={},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(
    "case",
    [
        "hash",
        "count",
        "invalid_count",
        "schema",
        "missing_method",
        "no_hashes",
        "level_identity",
        "missing_level",
        "override",
    ],
)
def test_adv_rejects_invalid_run_before_any_output(tmp_path: Path, case: str) -> None:
    source = tmp_path / "adv"
    _adv_fixture(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    match case:
        case "hash":
            manifest["files"]["points/p0/judged/steering.jsonl"] = "0" * 64
        case "count":
            manifest["max_concepts"] = 4
        case "invalid_count":
            manifest["max_concepts"] = 0
        case "schema":
            manifest["schema_version"] = 99
        case "missing_method":
            del manifest["best_factors"]["LoRA"]
        case "no_hashes":
            manifest["files"] = []
        case "level_identity":
            directory = source / "points/p8/judged"
            entries = [
                json.loads(s) for s in (directory / "steering.jsonl").read_text().splitlines()
            ]
            frame = pd.read_parquet(directory / "steering_data.parquet")
            frame["original_prompt"] = "another prompt"
            _write(directory, entries, frame)
        case "missing_level":
            (source / "points/p8/judged/steering.jsonl").unlink()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises((ValueError, FileNotFoundError)):
        hist.generate_histogram_reports(
            input_dir=source,
            output_dir=tmp_path / "out",
            pipeline="adv",
            method_factors=FACTORS if case == "override" else {},
        )
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("pipeline", ["upstream", "adv"])
def test_real_hydra_cli_offline(tmp_path: Path, pipeline: str) -> None:
    source = tmp_path / "source with spaces"
    if pipeline == "adv":
        _adv_fixture(source)
    else:
        _fixture(source)
    command: list[str] = [
        sys.executable,
        "-m",
        "scripts.eval.plot_axbench_histograms",
        f"pipeline={pipeline}",
        f"input_dir={source}",
        f"output_dir={tmp_path / 'out'}",
    ]
    if pipeline == "upstream":
        command.append("+method_factors={DiffMean:2,SelfConditioning:1,LoRA:1}")
    environment: dict[str, str] = {**os.environ, "PYTHONPATH": str(ROOT), "MPLBACKEND": "Agg"}
    environment.pop("OPENROUTER_API_KEY", None)
    result = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "out/metadata.json").is_file()
    assert not (tmp_path / "plot_axbench_histograms.log").exists()
    rejected = subprocess.run(
        command,
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert rejected.returncode != 0
    assert "new or empty" in rejected.stderr


def test_optional_nl_preserves_all_existing_histogram_colors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Literal pre-NL order: an inserted registry entry must not shift old styles.
    legacy = (
        "DiffMean", "LinearProbe", "LoRA", "PromptSteering",
        "SelfConditioning", "SelfConditioningMasked", "EncDecDiffMean",
        "EncDecLinearProbe", "EncDecPromptSteering", "EncDecLoRA",
    )
    expected = {method: (*to_rgba(hist.COLORS[method])[:3], 0.4) for method in legacy}
    observed: list[dict[str, str]] = []

    def capture(figure: Figure, *_args: Any, **_kwargs: Any) -> None:
        styles = {}
        for axis in figure.axes:
            for patch in axis.patches:
                label, color = patch.get_label(), patch.get_edgecolor()
                assert isinstance(label, str)
                method = label.split(" (factor")[0]
                styles[method] = color
                if method == "NLSteering":
                    assert patch.get_edgecolor() not in {to_rgba(color) for color in hist.COLORS.values()}
                else:
                    assert patch.get_edgecolor()[:3] == to_rgba(hist.COLORS[method])[:3]
        observed.append(styles)

    monkeypatch.setattr(Figure, "savefig", capture)
    for methods in (legacy, (*legacy, "NLSteering")):
        scores = pd.DataFrame([
            {"method": method, "axis": axis, "factor": 1.0, "score": 1.0, "n_prompts": 10, "concept_id": 0, "concept": "topic"}
            for method in methods for axis in hist.AXES
        ])
        hist._render(scores, np.linspace(0, 2, 11), "styles", tmp_path / "unused.png")
    assert observed[0] == expected
    assert {method: observed[1][method] for method in legacy} == expected


@pytest.mark.parametrize("case", [
    "default_reordered", "default_missing_member", "nl_missing_methods",
    "nl_missing_member", "nl_reordered",
])
def test_adv_manifest_methods_must_match_selected_factors(
    tmp_path: Path, case: str,
) -> None:
    source = tmp_path / "source"
    _adv_fixture(source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    methods = list(ADV_FACTORS)
    if case.startswith("nl_"):
        manifest["best_factors"]["NLSteering"] = 3.0
        methods.append("NLSteering")
    if case.endswith("reordered"):
        manifest["methods"] = methods[::-1]
    elif case.endswith("missing_member"):
        manifest["methods"] = methods[:-1]
    # nl_missing_methods retains the historical manifest with no methods field.
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Adv manifest method order differs from its factors"):
        hist.generate_histogram_reports(
            input_dir=source, output_dir=tmp_path / "out", pipeline="adv",
            method_factors={}, bins=10,
        )
    assert not (tmp_path / "out").exists()
