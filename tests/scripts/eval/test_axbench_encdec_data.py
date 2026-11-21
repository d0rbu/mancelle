from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from steering.eval.axbench_upstream.data import (
    select_training_pairs,
    select_training_texts,
)


def _write_prepared_dir(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    prepared_dir = tmp_path / "prepared"
    generate_dir = prepared_dir / "generate"
    generate_dir.mkdir(parents=True)
    metadata = (
        {
            "concept_id": 0,
            "concept": "alpha",
            "concept_genres_map": {"alpha": ["genre-a"]},
        },
        {
            "concept_id": 1,
            "concept": "beta",
            "concept_genres_map": {"beta": ["genre-b"]},
        },
    )
    (generate_dir / "metadata.jsonl").write_text(
        "".join(f"{json.dumps(row)}\n" for row in metadata),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        [
            {
                "input": "neg-a-0-in",
                "output": "neg-a-0-out",
                "output_concept": "EEEEE",
                "category": "negative",
                "concept_genre": "genre-a",
            },
            {
                "input": "alpha-0-in",
                "output": "alpha-0-out",
                "output_concept": "alpha",
                "category": "positive",
                "concept_genre": "genre-a",
            },
            {
                "input": "beta-0-in",
                "output": "beta-0-out",
                "output_concept": "beta",
                "category": "positive",
                "concept_genre": "genre-b",
            },
            {
                "input": "neg-a-1-in",
                "output": "neg-a-1-out",
                "output_concept": "EEEEE",
                "category": "negative",
                "concept_genre": "genre-a",
            },
            {
                "input": "alpha-1-in",
                "output": "alpha-1-out",
                "output_concept": "alpha",
                "category": "positive",
                "concept_genre": "genre-a",
            },
            {
                "input": "neg-b-0-in",
                "output": "neg-b-0-out",
                "output_concept": "EEEEE",
                "category": "negative",
                "concept_genre": "genre-b",
            },
        ]
    )
    frame.to_parquet(generate_dir / "train_data.parquet", index=False)
    return prepared_dir, frame


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def test_pair_selection_matches_concatenated_texts_in_source_order(tmp_path: Path) -> None:
    prepared_dir, _ = _write_prepared_dir(tmp_path)

    pairs = select_training_pairs(prepared_dir, max_concepts=2)
    texts = select_training_texts(prepared_dir, max_concepts=2)

    assert tuple(pairs) == (0, 1)
    for concept_id in pairs:
        selected_pairs = pairs[concept_id]
        selected_texts = texts[concept_id]
        assert (
            selected_pairs.concept_id,
            selected_pairs.concept,
            selected_pairs.genre,
        ) == (
            selected_texts.concept_id,
            selected_texts.concept,
            selected_texts.genre,
        )
        assert tuple(inp + out for inp, out in selected_pairs.positive) == (selected_texts.positive)
        assert tuple(inp + out for inp, out in selected_pairs.negative) == (selected_texts.negative)

    assert pairs[0].positive == (
        ("alpha-0-in", "alpha-0-out"),
        ("alpha-1-in", "alpha-1-out"),
    )
    assert pairs[0].negative == (
        ("neg-a-0-in", "neg-a-0-out"),
        ("neg-a-1-in", "neg-a-1-out"),
    )
    assert pairs[1].positive == (("beta-0-in", "beta-0-out"),)
    assert pairs[1].negative == (("neg-b-0-in", "neg-b-0-out"),)


def test_missing_training_file_uses_existing_error(tmp_path: Path) -> None:
    prepared_dir, _ = _write_prepared_dir(tmp_path)
    train_path = prepared_dir / "generate" / "train_data.parquet"
    train_path.unlink()

    with pytest.raises(
        FileNotFoundError,
        match=_exact(f"AxBench training data does not exist: {train_path}"),
    ):
        select_training_pairs(prepared_dir, max_concepts=2)


def test_missing_training_columns_use_existing_error(tmp_path: Path) -> None:
    prepared_dir, frame = _write_prepared_dir(tmp_path)
    train_path = prepared_dir / "generate" / "train_data.parquet"
    frame.drop(columns=["output", "category"]).to_parquet(train_path, index=False)

    with pytest.raises(
        ValueError,
        match=_exact(
            f"AxBench training data is missing columns ['category', 'output']: {train_path}"
        ),
    ):
        select_training_pairs(prepared_dir, max_concepts=2)


def test_empty_positive_class_uses_existing_error(tmp_path: Path) -> None:
    prepared_dir, frame = _write_prepared_dir(tmp_path)
    train_path = prepared_dir / "generate" / "train_data.parquet"
    frame.loc[frame["output_concept"].eq("alpha"), "output_concept"] = "other"
    frame.to_parquet(train_path, index=False)

    with pytest.raises(
        ValueError,
        match=_exact("no positive AxBench training rows for concept 0: 'alpha'"),
    ):
        select_training_pairs(prepared_dir, max_concepts=2)


def test_empty_negative_class_uses_existing_error(tmp_path: Path) -> None:
    prepared_dir, frame = _write_prepared_dir(tmp_path)
    train_path = prepared_dir / "generate" / "train_data.parquet"
    frame.loc[
        frame["output_concept"].eq("EEEEE") & frame["concept_genre"].eq("genre-a"),
        "concept_genre",
    ] = "other"
    frame.to_parquet(train_path, index=False)

    with pytest.raises(
        ValueError,
        match=_exact("no same-genre EEEEE negative rows for concept 0: 'genre-a'"),
    ):
        select_training_pairs(prepared_dir, max_concepts=2)


def test_bad_metadata_concept_id_uses_existing_error(tmp_path: Path) -> None:
    prepared_dir, _ = _write_prepared_dir(tmp_path)
    metadata_path = prepared_dir / "generate" / "metadata.jsonl"
    metadata_path.write_text(
        json.dumps(
            {
                "concept_id": 7,
                "concept": "alpha",
                "concept_genres_map": {"alpha": ["genre-a"]},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match=_exact("metadata concept_id must match JSONL position 0, got 7"),
    ):
        select_training_pairs(prepared_dir, max_concepts=1)
