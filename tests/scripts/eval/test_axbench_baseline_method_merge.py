"""Focused semantic tests for the AxBench baseline-only (main+aux) merge.

Covers :mod:`scripts.eval.axbench_upstream.merge_baseline_methods`, the
narrower sibling of the ten-method merge used for a base model (e.g. Pythia)
that has no project/encoder-decoder method comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.eval.axbench_upstream.merge_baseline_methods import (
    BASELINE_MAX_BLANK_GENERATION_FRACTION,
    BASELINE_METHODS,
    merge_baseline_frames,
    merge_baseline_parquets,
)
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    MAX_BLANK_GENERATION_FRACTION,
)

_METHODS = {
    "main": ("DiffMean", "LinearProbe"),
    "aux": ("PromptSteering", "LoRA"),
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


def _valid_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    main = _frame("main")
    aux = _frame("aux")
    aux["steered_input"] = aux["steered_input"].map(lambda value: f"prompt-selected: {value}")
    return main, aux


def test_baseline_methods_is_exactly_the_four_vendored_methods() -> None:
    assert BASELINE_METHODS == ("DiffMean", "LinearProbe", "PromptSteering", "LoRA")


def test_baseline_blank_fraction_bound_is_wider_than_the_chat_model_default() -> None:
    assert BASELINE_MAX_BLANK_GENERATION_FRACTION > MAX_BLANK_GENERATION_FRACTION


def test_merge_baseline_frames_tolerates_a_base_model_blank_rate_the_default_would_reject() -> None:
    """A blank rate above the chat-model (Qwen) 1% bound but within the
    base-model 2% bound -- the exact shape observed on real Pythia-2.8B
    data -- merges successfully under ``merge_baseline_frames``'s own bound,
    without the caller passing anything explicitly."""
    main, aux = _valid_frames()
    # 5 concepts x 10 inputs x 12 factors = 600 declared DiffMean cells;
    # 9 blanks is 1.5%, over the 1% chat-model default but under 2%.
    main = _frame("main", max_concepts=5)
    aux = _frame("aux", max_concepts=5)
    aux["steered_input"] = aux["steered_input"].map(lambda value: f"prompt-selected: {value}")
    main.loc[main.index[:9], "DiffMean_steered_generation"] = " "

    merged = merge_baseline_frames(main, aux, max_concepts=5)
    assert merged["DiffMean_steered_generation"].str.strip().str.len().eq(0).sum() == 9


def test_merge_baseline_frames_produces_all_four_method_columns() -> None:
    main, aux = _valid_frames()
    merged = merge_baseline_frames(main, aux, max_concepts=2)

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


def test_merge_baseline_frames_rejects_mismatched_aux_inputs() -> None:
    main, aux = _valid_frames()
    aux.loc[aux.index[0], "original_prompt"] = "a completely different prompt"
    with pytest.raises(ValueError, match="aux inputs do not match the main frame"):
        merge_baseline_frames(main, aux, max_concepts=2)


def test_merge_baseline_frames_rejects_an_invalid_main_frame() -> None:
    main, aux = _valid_frames()
    main = main.drop(columns=["DiffMean_strength"])
    with pytest.raises(ValueError, match="columns do not match"):
        merge_baseline_frames(main, aux, max_concepts=2)


def test_merge_baseline_parquets_round_trip(tmp_path: Path) -> None:
    main, aux = _valid_frames()
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    output_path = tmp_path / "merged.parquet"
    main.to_parquet(main_path)
    aux.to_parquet(aux_path)

    merged = merge_baseline_parquets(main_path, aux_path, output_path, max_concepts=2)
    assert output_path.is_file()
    reread = pd.read_parquet(output_path)
    assert len(reread) == len(merged)


def test_merge_baseline_parquets_refuses_existing_output_without_force(tmp_path: Path) -> None:
    main, aux = _valid_frames()
    main_path = tmp_path / "main.parquet"
    aux_path = tmp_path / "aux.parquet"
    output_path = tmp_path / "merged.parquet"
    main.to_parquet(main_path)
    aux.to_parquet(aux_path)
    output_path.write_bytes(b"existing content")

    with pytest.raises(FileExistsError, match="merged output already exists"):
        merge_baseline_parquets(main_path, aux_path, output_path, max_concepts=2)

    merge_baseline_parquets(main_path, aux_path, output_path, max_concepts=2, force=True)
    assert pd.read_parquet(output_path).shape[0] == len(main)
