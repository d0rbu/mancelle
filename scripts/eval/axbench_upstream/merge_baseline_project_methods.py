"""Merge four vendored baselines with the selected project-method product.

Unlike :mod:`merge_methods` (all ten methods, including the encoder-decoder
baseline analogs) and :mod:`merge_baseline_methods` (the four vendored
baselines alone), this merges ``main`` + ``aux`` (DiffMean, LinearProbe,
PromptSteering, LoRA) with ``project`` (SelfConditioning,
SelfConditioningMasked) into a six-method frame -- for a base model (e.g.
Pythia, SmolLM3) compared against the project's own encoder-decoder
conditioning methods, with no encoder-decoder baseline analogs in the mix.
The explicit NL option adds ``NLSteering`` without changing that default.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
from loguru import logger

from nanochat.atomic_io import atomic_write_parquet
from scripts.eval.axbench_upstream.merge_baseline_methods import (
    BASELINE_MAX_BLANK_GENERATION_FRACTION,
    BASELINE_METHODS,
)
from scripts.eval.axbench_upstream.merge_methods import (
    INPUT_COLUMNS,
    KEY_COLUMNS,
    NL_PROJECT_METHODS,
    FrameSource,
    _require_matching_inputs,
    _sorted,
    method_columns,
    methods_for_source,
    validate_steering_frame,
)

BASELINE_PROJECT_METHODS = (*BASELINE_METHODS, *methods_for_source(FrameSource.PROJECT))
BASELINE_PROJECT_NL_METHODS = (*BASELINE_METHODS, *NL_PROJECT_METHODS)


def require_baseline_project_methods(methods: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(methods)
    if selected not in {BASELINE_PROJECT_METHODS, BASELINE_PROJECT_NL_METHODS}:
        raise ValueError(
            "baseline+project methods must be exactly the six-method default or "
            f"the seven-method NL tuple, got {selected!r}"
        )
    return selected


def merge_baseline_project_frames(
    main: pd.DataFrame,
    aux: pd.DataFrame,
    project: pd.DataFrame,
    *,
    max_concepts: int,
    methods: tuple[str, ...] = BASELINE_PROJECT_METHODS,
) -> pd.DataFrame:
    """Validate and merge the main, aux, and project frames."""

    methods = require_baseline_project_methods(methods)
    project_methods = methods[len(BASELINE_METHODS) :]

    validate_steering_frame(
        main,
        source=FrameSource.MAIN,
        max_concepts=max_concepts,
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )
    validate_steering_frame(
        aux,
        source=FrameSource.AUX,
        max_concepts=max_concepts,
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )
    validate_steering_frame(
        project,
        source=FrameSource.PROJECT,
        max_concepts=max_concepts,
        methods=project_methods,
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )

    _require_matching_inputs(main, project, candidate_name="project", columns=INPUT_COLUMNS)
    main_factor_one = main.loc[main["factor"].eq(1.0)]
    stable_aux_columns = tuple(column for column in INPUT_COLUMNS if column != "steered_input")
    _require_matching_inputs(
        main_factor_one,
        aux,
        candidate_name="aux",
        columns=stable_aux_columns,
    )

    merged = _sorted(main)
    for source_frame, source_methods in (
        (aux, methods_for_source(FrameSource.AUX)),
        (project, project_methods),
    ):
        additions = source_frame.loc[:, [*KEY_COLUMNS, *method_columns(source_methods)]]
        merged = merged.merge(
            additions,
            on=list(KEY_COLUMNS),
            how="left",
            sort=False,
            validate="one_to_one",
        )
    merged = merged.loc[:, [*INPUT_COLUMNS, *method_columns(methods)]]
    validate_steering_frame(
        merged,
        source=FrameSource.MERGED,
        max_concepts=max_concepts,
        methods=methods,
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )
    return merged


def merge_baseline_project_parquets(
    main_path: Path,
    aux_path: Path,
    project_path: Path,
    output_path: Path,
    *,
    max_concepts: int,
    methods: tuple[str, ...] = BASELINE_PROJECT_METHODS,
    force: bool = False,
) -> pd.DataFrame:
    """Read, validate, merge, then write three steering Parquets."""

    if type(force) is not bool:
        raise TypeError("force must be a boolean")
    resolved_inputs = {
        Path(main_path).resolve(),
        Path(aux_path).resolve(),
        Path(project_path).resolve(),
    }
    resolved_output = Path(output_path).resolve()
    if resolved_output in resolved_inputs:
        raise ValueError("merged output path must differ from every input path")
    if resolved_output.exists() and not force:
        raise FileExistsError(
            f"merged output already exists: {resolved_output}; pass force=True to replace it"
        )

    merged = merge_baseline_project_frames(
        pd.read_parquet(main_path),
        pd.read_parquet(aux_path),
        pd.read_parquet(project_path),
        max_concepts=max_concepts,
        methods=methods,
    )
    atomic_write_parquet(pa.Table.from_pandas(merged, preserve_index=False), resolved_output)
    return merged


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main", type=Path, required=True, help="main baseline steering Parquet")
    parser.add_argument("--aux", type=Path, required=True, help="aux baseline steering Parquet")
    parser.add_argument("--project", type=Path, required=True, help="project-method Parquet")
    parser.add_argument("--output", type=Path, required=True, help="merged output Parquet")
    parser.add_argument("--max-concepts", type=_positive_int, required=True)
    parser.add_argument(
        "--include-nl-steering",
        action="store_true",
        help="merge the explicit seven-method product including NLSteering",
    )
    parser.add_argument("--force", action="store_true", help="replace an existing merged output")
    args = parser.parse_args(argv)

    merged = merge_baseline_project_parquets(
        args.main,
        args.aux,
        args.project,
        args.output,
        max_concepts=args.max_concepts,
        methods=(
            BASELINE_PROJECT_NL_METHODS
            if args.include_nl_steering
            else BASELINE_PROJECT_METHODS
        ),
        force=args.force,
    )
    logger.info("Wrote {} validated rows to {}", len(merged), args.output)


if __name__ == "__main__":
    main()
