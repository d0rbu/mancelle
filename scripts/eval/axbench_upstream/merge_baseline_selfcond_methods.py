"""Validate and merge the four vendored baselines with SelfConditioning ALONE.

Unlike :mod:`merge_baseline_project_methods` (the four vendored baselines
plus BOTH project methods, SelfConditioning and SelfConditioningMasked),
this deliberately excludes SelfConditioningMasked -- its rank-based
candidate-pool rewrite (methods.py's ``..._sign_rank_v3`` policy) has not
yet had its own project-methods data regenerated and separately validated,
so it is temporarily dropped from the judged comparison rather than judged
on stale or unvalidated data. The project Parquet still carries
SelfConditioningMasked columns (generation always produces both methods
together); this module simply narrows the declared method set to
``("SelfConditioning",)`` when validating/merging so those columns are
ignored, not judged. Re-add SelfConditioningMasked by going back to
:mod:`merge_baseline_project_methods` once its rank-era data is confirmed.
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
    FrameSource,
    _require_matching_inputs,
    _sorted,
    method_columns,
    methods_for_source,
    validate_steering_frame,
)

PROJECT_METHODS = ("SelfConditioning",)
BASELINE_SELFCOND_METHODS = (*BASELINE_METHODS, *PROJECT_METHODS)


def merge_baseline_selfcond_frames(
    main: pd.DataFrame,
    aux: pd.DataFrame,
    project: pd.DataFrame,
    *,
    max_concepts: int,
) -> pd.DataFrame:
    """Validate and merge the main, aux, and project frames -- SelfConditioning only."""

    # The project parquet always carries both project methods' columns
    # (generation computes SelfConditioning and SelfConditioningMasked
    # together); narrow to SelfConditioning alone before validating against
    # the narrower declared method set -- validate_steering_frame's
    # `methods` override rejects a frame carrying UNDECLARED extra columns,
    # it does not itself drop them. Select only columns actually present so
    # a genuinely missing expected column surfaces through the validator's
    # own "columns do not match" error, not a raw KeyError here.
    wanted = [*INPUT_COLUMNS, *method_columns(PROJECT_METHODS)]
    project = project.loc[:, [column for column in wanted if column in project.columns]]

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
        methods=PROJECT_METHODS,
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
    for source_frame, methods in (
        (aux, methods_for_source(FrameSource.AUX)),
        (project, PROJECT_METHODS),
    ):
        additions = source_frame.loc[:, [*KEY_COLUMNS, *method_columns(methods)]]
        merged = merged.merge(
            additions,
            on=list(KEY_COLUMNS),
            how="left",
            sort=False,
            validate="one_to_one",
        )
    merged = merged.loc[:, [*INPUT_COLUMNS, *method_columns(BASELINE_SELFCOND_METHODS)]]
    validate_steering_frame(
        merged,
        source=FrameSource.MERGED,
        max_concepts=max_concepts,
        methods=BASELINE_SELFCOND_METHODS,
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )
    return merged


def merge_baseline_selfcond_parquets(
    main_path: Path,
    aux_path: Path,
    project_path: Path,
    output_path: Path,
    *,
    max_concepts: int,
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

    merged = merge_baseline_selfcond_frames(
        pd.read_parquet(main_path),
        pd.read_parquet(aux_path),
        pd.read_parquet(project_path),
        max_concepts=max_concepts,
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
    parser.add_argument("--force", action="store_true", help="replace an existing merged output")
    args = parser.parse_args(argv)

    merged = merge_baseline_selfcond_parquets(
        args.main,
        args.aux,
        args.project,
        args.output,
        max_concepts=args.max_concepts,
        force=args.force,
    )
    logger.info("Wrote {} validated rows to {}", len(merged), args.output)


if __name__ == "__main__":
    main()
