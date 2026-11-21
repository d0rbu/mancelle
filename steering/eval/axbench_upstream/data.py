"""Select the AxBench concept-training texts used by project methods."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

EMPTY_CONCEPT = "EEEEE"
_REQUIRED_TRAIN_COLUMNS = (
    "input",
    "output",
    "output_concept",
    "category",
    "concept_genre",
)


@dataclass(frozen=True)
class ConceptTrainingTexts:
    """The exact positive/negative text contrast for one metadata row."""

    concept_id: int
    concept: str
    genre: str
    positive: tuple[str, ...]
    negative: tuple[str, ...]


@dataclass(frozen=True)
class ConceptTrainingPairs:
    """The exact positive/negative input-output pairs for one metadata row."""

    concept_id: int
    concept: str
    genre: str
    positive: tuple[tuple[str, str], ...]
    negative: tuple[tuple[str, str], ...]


def load_metadata(metadata_path: Path) -> list[dict[str, Any]]:
    """Load metadata in JSONL order, which is AxBench's concept-id order."""

    metadata_path = Path(metadata_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(f"AxBench metadata does not exist: {metadata_path}")

    rows: list[dict[str, Any]] = []
    with metadata_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"invalid JSON in {metadata_path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object in {metadata_path} at line {line_number}")
            row_id = row.get("concept_id")
            if row_id is not None and (type(row_id) is not int or row_id != len(rows)):
                raise ValueError(
                    f"metadata concept_id must match JSONL position {len(rows)}, got {row_id!r}"
                )
            rows.append(row)
    if not rows:
        raise ValueError(f"AxBench metadata is empty: {metadata_path}")
    return rows


def _concept_and_genre(metadata: list[dict[str, Any]], concept_id: int) -> tuple[str, str]:
    if concept_id < 0 or concept_id >= len(metadata):
        raise ValueError(
            f"concept_id {concept_id} is outside metadata positions 0..{len(metadata) - 1}"
        )
    entry = metadata[concept_id]
    concept = entry.get("concept")
    if not isinstance(concept, str) or not concept:
        raise ValueError(f"metadata position {concept_id} has no nonempty concept")
    genre_map = entry.get("concept_genres_map")
    if not isinstance(genre_map, dict):
        raise ValueError(f"metadata position {concept_id} has no concept_genres_map")
    genres = genre_map.get(concept)
    if (
        not isinstance(genres, list)
        or not genres
        or not isinstance(genres[0], str)
        or not genres[0]
    ):
        raise ValueError(f"metadata position {concept_id} has no primary genre for {concept!r}")
    return concept, genres[0]


def _input_output_pairs(
    frame: pd.DataFrame,
    *,
    label: str,
) -> tuple[tuple[str, str], ...]:
    values: list[tuple[str, str]] = []
    for row_number, (input_text, output_text) in enumerate(
        zip(frame["input"].tolist(), frame["output"].tolist(), strict=True)
    ):
        if not isinstance(input_text, str) or not isinstance(output_text, str):
            raise ValueError(f"{label} training row {row_number} must have string input and output")
        values.append((input_text, output_text))
    return tuple(values)


def _combined_texts(frame: pd.DataFrame, *, label: str) -> tuple[str, ...]:
    return tuple(
        input_text + output_text
        for input_text, output_text in _input_output_pairs(frame, label=label)
    )


def _select_training_rows(
    prepared_dir: Path,
    *,
    max_concepts: int,
) -> Iterator[tuple[int, str, str, pd.DataFrame, pd.DataFrame]]:
    if type(max_concepts) is not int or max_concepts <= 0:
        raise ValueError("max_concepts must be a positive integer")

    generate_dir = Path(prepared_dir) / "generate"
    metadata = load_metadata(generate_dir / "metadata.jsonl")
    if max_concepts > len(metadata):
        raise ValueError(f"max_concepts={max_concepts} exceeds {len(metadata)} metadata rows")

    train_path = generate_dir / "train_data.parquet"
    if not train_path.is_file():
        raise FileNotFoundError(f"AxBench training data does not exist: {train_path}")
    frame = pd.read_parquet(train_path)
    missing = sorted(set(_REQUIRED_TRAIN_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"AxBench training data is missing columns {missing}: {train_path}")

    negative_pool = frame.loc[
        frame["output_concept"].eq(EMPTY_CONCEPT) & frame["category"].eq("negative")
    ]
    for concept_id in range(max_concepts):
        concept, genre = _concept_and_genre(metadata, concept_id)
        positive_frame = frame.loc[
            frame["output_concept"].eq(concept) & frame["category"].eq("positive")
        ]
        negative_frame = negative_pool.loc[negative_pool["concept_genre"].eq(genre)]
        if positive_frame.empty:
            raise ValueError(
                f"no positive AxBench training rows for concept {concept_id}: {concept!r}"
            )
        if negative_frame.empty:
            raise ValueError(
                f"no same-genre EEEEE negative rows for concept {concept_id}: {genre!r}"
            )
        yield concept_id, concept, genre, positive_frame, negative_frame


def select_training_texts(
    prepared_dir: Path,
    *,
    max_concepts: int,
) -> dict[int, ConceptTrainingTexts]:
    """Mirror AxBench's positive and same-genre EEEEE-negative selection.

    The metadata row position is the concept id. The training Parquet is read
    once, and pandas boolean selection preserves its row order. Texts use the
    upstream non-chat binarization convention: ``input + output``.
    """

    selected: dict[int, ConceptTrainingTexts] = {}
    for concept_id, concept, genre, positive_frame, negative_frame in _select_training_rows(
        prepared_dir,
        max_concepts=max_concepts,
    ):
        selected[concept_id] = ConceptTrainingTexts(
            concept_id=concept_id,
            concept=concept,
            genre=genre,
            positive=_combined_texts(positive_frame, label=f"concept {concept_id} positive"),
            negative=_combined_texts(negative_frame, label=f"concept {concept_id} negative"),
        )
    return selected


def select_training_pairs(
    prepared_dir: Path,
    *,
    max_concepts: int,
) -> dict[int, ConceptTrainingPairs]:
    """Select AxBench positive and same-genre negative rows as input-output pairs."""

    selected: dict[int, ConceptTrainingPairs] = {}
    for concept_id, concept, genre, positive_frame, negative_frame in _select_training_rows(
        prepared_dir,
        max_concepts=max_concepts,
    ):
        selected[concept_id] = ConceptTrainingPairs(
            concept_id=concept_id,
            concept=concept,
            genre=genre,
            positive=_input_output_pairs(
                positive_frame,
                label=f"concept {concept_id} positive",
            ),
            negative=_input_output_pairs(
                negative_frame,
                label=f"concept {concept_id} negative",
            ),
        )
    return selected


__all__ = [
    "EMPTY_CONCEPT",
    "ConceptTrainingPairs",
    "ConceptTrainingTexts",
    "load_metadata",
    "select_training_pairs",
    "select_training_texts",
]
