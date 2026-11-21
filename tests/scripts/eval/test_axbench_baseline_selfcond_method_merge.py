"""Focused semantic tests for the AxBench baseline-plus-SelfConditioning
(five-method, no SelfConditioningMasked) merge.

Covers :mod:`scripts.eval.axbench_upstream.merge_baseline_selfcond_methods`,
the five-method sibling of the six-method `merge_baseline_project_methods`
-- used to judge a base model (e.g. Pythia, SmolLM3) against the project's
own SelfConditioning method ALONE, deliberately excluding
SelfConditioningMasked until its rank-based rewrite has its own project data
regenerated and separately validated.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.eval.axbench_upstream.merge_baseline_methods import (
    BASELINE_MAX_BLANK_GENERATION_FRACTION,
)
from scripts.eval.axbench_upstream.merge_baseline_selfcond_methods import (
    BASELINE_SELFCOND_METHODS,
    merge_baseline_selfcond_frames,
    merge_baseline_selfcond_parquets,
)
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, INPUT_COLUMNS

_METHODS = {
    "main": ("DiffMean", "LinearProbe"),
    "aux": ("PromptSteering", "LoRA"),
    "project": ("SelfConditioning", "SelfConditioningMasked"),
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
    # The real project parquet always carries BOTH project methods'
    # columns -- generation computes SelfConditioning and
    # SelfConditioningMasked together. Keep both here so tests exercise the
    # real shape, not an already-narrowed fixture.
    project = _frame("project")
    return main, aux, project


def test_baseline_selfcond_methods_is_five_methods_no_masked() -> None:
    assert BASELINE_SELFCOND_METHODS == (
        "DiffMean",
        "LinearProbe",
        "PromptSteering",
        "LoRA",
        "SelfConditioning",
    )
    assert "SelfConditioningMasked" not in BASELINE_SELFCOND_METHODS


def test_merge_drops_selfconditioningmasked_from_a_realistic_project_frame() -> None:
    """The project parquet always has both methods' columns; the merge must
    narrow to SelfConditioning alone rather than erroring or leaking masked
    columns into the output."""
    main, aux, project = _valid_frames()
    assert "SelfConditioningMasked_steered_generation" in project.columns

    merged = merge_baseline_selfcond_frames(main, aux, project, max_concepts=2)

    assert "SelfConditioningMasked_steered_generation" not in merged.columns
    assert "SelfConditioningMasked_strength" not in merged.columns
    assert "SelfConditioning_steered_generation" in merged.columns
    assert merged["SelfConditioning_steered_generation"].notna().all()


def test_merge_baseline_selfcond_frames_produces_all_five_method_columns() -> None:
    main, aux, project = _valid_frames()
    merged = merge_baseline_selfcond_frames(main, aux, project, max_concepts=2)

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

    # SelfConditioning, like main, declares the full factor grid.
    assert merged["SelfConditioning_steered_generation"].notna().all()


def test_merge_baseline_selfcond_frames_rejects_mismatched_project_inputs() -> None:
    main, aux, project = _valid_frames()
    # Mutate every factor row for one (concept_id, input_id) so the project
    # frame stays internally self-consistent (an invariant column may not
    # vary across factors) while still differing from main.
    mask = project["concept_id"].eq(0) & project["input_id"].eq(0)
    project.loc[mask, "original_prompt"] = "a completely different prompt"
    with pytest.raises(ValueError, match="project inputs do not match the main frame"):
        merge_baseline_selfcond_frames(main, aux, project, max_concepts=2)


def test_merge_baseline_selfcond_frames_rejects_mismatched_aux_inputs() -> None:
    main, aux, project = _valid_frames()
    aux.loc[aux.index[0], "original_prompt"] = "a completely different prompt"
    with pytest.raises(ValueError, match="aux inputs do not match the main frame"):
        merge_baseline_selfcond_frames(main, aux, project, max_concepts=2)


def test_merge_baseline_selfcond_frames_rejects_an_invalid_project_frame() -> None:
    main, aux, project = _valid_frames()
    project = project.drop(columns=["SelfConditioning_strength"])
    with pytest.raises(ValueError, match="columns do not match"):
        merge_baseline_selfcond_frames(main, aux, project, max_concepts=2)


def test_merge_baseline_selfcond_frames_uses_the_baseline_blank_fraction_bound() -> None:
    """Same base-model-calibrated bound as the four/six-method merges, since
    the baseline model side is still a base model (Pythia, SmolLM3), not a
    chat model."""
    main = _frame("main", max_concepts=5)
    aux = _frame("aux", max_concepts=5)
    aux["steered_input"] = aux["steered_input"].map(lambda value: f"prompt-selected: {value}")
    project = _frame("project", max_concepts=5)
    # 5 concepts x 10 inputs x 12 factors = 600 declared DiffMean cells;
    # 9 blanks is 1.5%, over the 1% chat-model default but under the 2% bound.
    main.loc[main.index[:9], "DiffMean_steered_generation"] = " "

    merged = merge_baseline_selfcond_frames(main, aux, project, max_concepts=5)
    assert merged["DiffMean_steered_generation"].str.strip().str.len().eq(0).sum() == 9
    assert BASELINE_MAX_BLANK_GENERATION_FRACTION == 0.02


def test_merge_baseline_selfcond_parquets_round_trip(tmp_path: Path) -> None:
    main, aux, project = _valid_frames()
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    project_path = tmp_path / "project.parquet"
    output_path = tmp_path / "merged.parquet"
    main.to_parquet(main_path)
    aux.to_parquet(aux_path)
    project.to_parquet(project_path)

    merged = merge_baseline_selfcond_parquets(
        main_path, aux_path, project_path, output_path, max_concepts=2
    )
    assert output_path.is_file()
    reread = pd.read_parquet(output_path)
    assert len(reread) == len(merged)
    assert "SelfConditioningMasked_steered_generation" not in reread.columns


def test_merge_baseline_selfcond_parquets_refuses_existing_output_without_force(
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
        merge_baseline_selfcond_parquets(
            main_path, aux_path, project_path, output_path, max_concepts=2
        )

    merge_baseline_selfcond_parquets(
        main_path, aux_path, project_path, output_path, max_concepts=2, force=True
    )
    assert pd.read_parquet(output_path).shape[0] == len(main)
