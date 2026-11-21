"""Focused semantic tests for the AxBench baseline-plus-project (six-method) merge.

Covers :mod:`scripts.eval.axbench_upstream.merge_baseline_project_methods`,
the six-method sibling of the four-method `merge_baseline_methods` and the
ten-method `merge_methods` -- used for a base model (e.g. Pythia, SmolLM3)
compared against the project's own SelfConditioning/SelfConditioningMasked
encoder-decoder methods, with no encoder-decoder baseline analogs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.eval.axbench_upstream import merge_baseline_project_methods as merger
from scripts.eval.axbench_upstream.merge_baseline_project_methods import (
    BASELINE_PROJECT_METHODS,
    BASELINE_PROJECT_NL_METHODS,
    merge_baseline_project_frames,
    merge_baseline_project_parquets,
)
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, INPUT_COLUMNS

_METHODS = {
    "main": ("DiffMean", "LinearProbe"),
    "aux": ("PromptSteering", "LoRA"),
    "project": ("SelfConditioning", "SelfConditioningMasked"),
    "project_nl": ("SelfConditioning", "SelfConditioningMasked", "NLSteering"),
}
_PERPLEXITY_METHODS = frozenset({"PromptSteering", "LoRA"})


def _metric_column(method: str) -> str:
    suffix = "perplexity" if method in _PERPLEXITY_METHODS else "strength"
    return f"{method}_{suffix}"


def _frame(source: str, *, max_concepts: int = 2) -> pd.DataFrame:
    factors = (1.0,) if source == "aux" else FULL_FACTORS
    rows = []
    for concept_id in range(max_concepts):
        for input_id in range(10):
            for factor in factors:
                row: dict[str, object] = {
                    "dataset_name": "AlpacaEval",
                    "concept_id": concept_id,
                    "input_concept": f"concept-{concept_id}",
                    "input_id": input_id,
                    "factor": factor,
                    "original_prompt": f"original prompt {concept_id}/{input_id}",
                    "steered_input": f"steered input {concept_id}/{input_id}",
                    "input": f"plain input {concept_id}/{input_id}",
                    "suppress_original": f"original suppression {concept_id}/{input_id}",
                    "suppress_rewrite": f"rewritten suppression {concept_id}/{input_id}",
                    "steered_prompt": f"steered prompt {concept_id}/{input_id}",
                    "defense": "none",
                    "sae_link": f"https://example.test/sae/{concept_id}",
                    "sae_id": 100 + concept_id,
                }
                for method in _METHODS[source]:
                    row[f"{method}_steered_generation"] = (
                        f"{method} generated a healthy response for "
                        f"{concept_id}/{input_id}/{factor}"
                    )
                    row[_metric_column(method)] = (
                        2.0 + concept_id + input_id / 100.0
                        if method in _PERPLEXITY_METHODS
                        else factor * (concept_id + 1.0)
                    )
                rows.append(row)

    frame = pd.DataFrame(rows)
    frame["concept_id"] = frame["concept_id"].astype("int64")
    frame["input_id"] = frame["input_id"].astype("int64")
    frame["sae_id"] = frame["sae_id"].astype("int64")
    frame["factor"] = frame["factor"].astype("float64")
    for method in _METHODS[source]:
        frame[_metric_column(method)] = frame[_metric_column(method)].astype("float64")
    return frame.loc[:, [*INPUT_COLUMNS, *frame.columns.difference(INPUT_COLUMNS, sort=False)]]


def _valid_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = _frame("main")
    aux = _frame("aux")
    aux["steered_input"] = aux["steered_input"].map(lambda value: f"prompt-selected: {value}")
    project = _frame("project")
    return main, aux, project


def test_baseline_project_methods_is_the_six_methods_in_order() -> None:
    assert BASELINE_PROJECT_METHODS == (
        "DiffMean",
        "LinearProbe",
        "PromptSteering",
        "LoRA",
        "SelfConditioning",
        "SelfConditioningMasked",
    )
    assert BASELINE_PROJECT_NL_METHODS == (*BASELINE_PROJECT_METHODS, "NLSteering")


def test_merge_baseline_project_frames_produces_all_six_method_columns() -> None:
    main, aux, project = _valid_frames()
    merged = merge_baseline_project_frames(main, aux, project, max_concepts=2)

    expected_columns = {
        *INPUT_COLUMNS,
        "DiffMean_steered_generation",
        "DiffMean_strength",
        "LinearProbe_steered_generation",
        "LinearProbe_strength",
        "PromptSteering_steered_generation",
        "PromptSteering_perplexity",
        "LoRA_steered_generation",
        "LoRA_perplexity",
        "SelfConditioning_steered_generation",
        "SelfConditioning_strength",
        "SelfConditioningMasked_steered_generation",
        "SelfConditioningMasked_strength",
    }
    assert set(merged.columns) == expected_columns
    assert len(merged) == len(main)

    # aux methods are declared only at factor == 1.0; every other row is null.
    off_grid = merged.loc[merged["factor"].ne(1.0)]
    assert off_grid["PromptSteering_steered_generation"].isna().all()
    assert off_grid["LoRA_perplexity"].isna().all()
    on_grid = merged.loc[merged["factor"].eq(1.0)]
    assert on_grid["PromptSteering_steered_generation"].notna().all()
    assert on_grid["LoRA_perplexity"].notna().all()

    # project methods, like main, declare the full factor grid.
    assert merged["SelfConditioning_steered_generation"].notna().all()
    assert merged["SelfConditioningMasked_strength"].notna().all()


def test_explicit_nl_merge_preserves_old_cells_and_adds_the_complete_nl_grid() -> None:
    main, aux, project = _valid_frames()
    project_nl = _frame("project_nl")
    default = merge_baseline_project_frames(main, aux, project, max_concepts=2)
    selected = merge_baseline_project_frames(
        main,
        aux,
        project_nl,
        max_concepts=2,
        methods=BASELINE_PROJECT_NL_METHODS,
    )

    pd.testing.assert_frame_equal(selected.loc[:, default.columns], default)
    assert tuple(selected.columns[-2:]) == (
        "NLSteering_steered_generation",
        "NLSteering_strength",
    )
    assert selected["NLSteering_steered_generation"].notna().all()
    assert selected["NLSteering_strength"].tolist() == project_nl["NLSteering_strength"].tolist()


def test_method_selection_never_infers_nl_from_columns() -> None:
    main, aux, _ = _valid_frames()
    project_nl = _frame("project_nl")
    with pytest.raises(ValueError, match="unknown method result columns.*NLSteering"):
        merge_baseline_project_frames(main, aux, project_nl, max_concepts=2)

    project = _frame("project")
    with pytest.raises(ValueError, match="missing=.*NLSteering"):
        merge_baseline_project_frames(
            main,
            aux,
            project,
            max_concepts=2,
            methods=BASELINE_PROJECT_NL_METHODS,
        )


def test_public_merge_rejects_a_reordered_explicit_method_tuple() -> None:
    main, aux, project = _valid_frames()
    reordered = (*BASELINE_PROJECT_METHODS[:-1], "NLSteering", BASELINE_PROJECT_METHODS[-1])

    with pytest.raises(ValueError, match="must be exactly the six-method default"):
        merge_baseline_project_frames(
            main,
            aux,
            project,
            max_concepts=2,
            methods=reordered,
        )


@pytest.mark.parametrize(
    ("optional_flag", "expected_methods"),
    [
        ([], BASELINE_PROJECT_METHODS),
        (["--include-nl-steering"], BASELINE_PROJECT_NL_METHODS),
    ],
)
def test_merge_cli_dispatches_default_and_explicit_nl_method_sets(
    monkeypatch: pytest.MonkeyPatch,
    optional_flag: list[str],
    expected_methods: tuple[str, ...],
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_merge(*args: object, **kwargs: object) -> pd.DataFrame:
        calls.append((args, kwargs))
        return pd.DataFrame([{}])

    monkeypatch.setattr(merger, "merge_baseline_project_parquets", fake_merge)

    merger.main([
        "--main",
        "main.parquet",
        "--aux",
        "aux.parquet",
        "--project",
        "project.parquet",
        "--output",
        "merged.parquet",
        "--max-concepts",
        "2",
        *optional_flag,
    ])

    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == tuple(
        Path(name)
        for name in ("main.parquet", "aux.parquet", "project.parquet", "merged.parquet")
    )
    assert kwargs == {"max_concepts": 2, "methods": expected_methods, "force": False}


def test_merge_baseline_project_frames_rejects_mismatched_project_inputs() -> None:
    main, aux, project = _valid_frames()
    # Mutate every factor row for one (concept_id, input_id) so the project
    # frame stays internally self-consistent (an invariant column may not
    # vary across factors) while still differing from main.
    mask = project["concept_id"].eq(0) & project["input_id"].eq(0)
    project.loc[mask, "original_prompt"] = "a completely different prompt"
    with pytest.raises(ValueError, match="project inputs do not match the main frame"):
        merge_baseline_project_frames(main, aux, project, max_concepts=2)


def test_merge_baseline_project_frames_rejects_mismatched_aux_inputs() -> None:
    main, aux, project = _valid_frames()
    aux.loc[aux.index[0], "original_prompt"] = "a completely different prompt"
    with pytest.raises(ValueError, match="aux inputs do not match the main frame"):
        merge_baseline_project_frames(main, aux, project, max_concepts=2)


def test_merge_baseline_project_frames_rejects_an_invalid_project_frame() -> None:
    main, aux, project = _valid_frames()
    project = project.drop(columns=["SelfConditioning_strength"])
    with pytest.raises(ValueError, match="columns do not match"):
        merge_baseline_project_frames(main, aux, project, max_concepts=2)


def test_merge_baseline_project_frames_uses_the_baseline_blank_fraction_bound() -> None:
    """Same base-model-calibrated bound as the four-method merge, since the

    baseline model side of a six-method run is still a base model (Pythia,
    SmolLM3), not a chat model."""
    main, aux, project = _valid_frames()
    main = _frame("main", max_concepts=5)
    aux = _frame("aux", max_concepts=5)
    aux["steered_input"] = aux["steered_input"].map(lambda value: f"prompt-selected: {value}")
    project = _frame("project", max_concepts=5)
    # 5 concepts x 10 inputs x 12 factors = 600 declared DiffMean cells;
    # 9 blanks is 1.5%, over the 1% chat-model default but under the 2% bound.
    main.loc[main.index[:9], "DiffMean_steered_generation"] = " "

    merged = merge_baseline_project_frames(main, aux, project, max_concepts=5)
    assert merged["DiffMean_steered_generation"].str.strip().str.len().eq(0).sum() == 9


def test_merge_baseline_project_parquets_round_trip(tmp_path: Path) -> None:
    main, aux, project = _valid_frames()
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    project_path = tmp_path / "project.parquet"
    output_path = tmp_path / "merged.parquet"
    main.to_parquet(main_path)
    aux.to_parquet(aux_path)
    project.to_parquet(project_path)

    merged = merge_baseline_project_parquets(
        main_path, aux_path, project_path, output_path, max_concepts=2
    )
    assert output_path.is_file()
    reread = pd.read_parquet(output_path)
    assert len(reread) == len(merged)


def test_nl_parquet_roundtrip_preserves_exact_output_schema_and_values(tmp_path: Path) -> None:
    main, aux, _ = _valid_frames()
    project = _frame("project_nl")
    paths = [tmp_path / name for name in ("main.parquet", "aux.parquet", "project.parquet")]
    for frame, path in zip((main, aux, project), paths, strict=True):
        frame.to_parquet(path, index=False)
    output = tmp_path / "merged.parquet"

    merged = merge_baseline_project_parquets(
        paths[0],
        paths[1],
        paths[2],
        output,
        max_concepts=2,
        methods=BASELINE_PROJECT_NL_METHODS,
    )

    pd.testing.assert_frame_equal(pd.read_parquet(output), merged)


def test_merge_baseline_project_parquets_refuses_existing_output_without_force(
    tmp_path: Path,
) -> None:
    main, aux, project = _valid_frames()
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    project_path = tmp_path / "project.parquet"
    output_path = tmp_path / "merged.parquet"
    main.to_parquet(main_path)
    aux.to_parquet(aux_path)
    project.to_parquet(project_path)
    output_path.write_bytes(b"existing content")

    with pytest.raises(FileExistsError, match="merged output already exists"):
        merge_baseline_project_parquets(
            main_path, aux_path, project_path, output_path, max_concepts=2
        )

    merge_baseline_project_parquets(
        main_path, aux_path, project_path, output_path, max_concepts=2, force=True
    )
    assert pd.read_parquet(output_path).shape[0] == len(main)
