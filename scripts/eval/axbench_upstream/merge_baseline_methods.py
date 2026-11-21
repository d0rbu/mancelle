"""Validate and merge the four vendored AxBench baseline method columns.

Unlike :mod:`merge_methods`, this merges only ``main`` (DiffMean,
LinearProbe) and ``aux`` (PromptSteering, LoRA) -- there is no
project/encoder-decoder comparison to align against, e.g. for a base model
(Pythia) that the project's SAE-conditioning and encoder-decoder methods do
not apply to.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
from loguru import logger

from nanochat.atomic_io import atomic_write_parquet
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

BASELINE_METHODS = (*methods_for_source(FrameSource.MAIN), *methods_for_source(FrameSource.AUX))

#: Blank-generation tripwire bound for a baseline-only (base, non-chat model)
#: comparison, overriding :data:`merge_methods.MAX_BLANK_GENERATION_FRACTION`
#: (0.01), which was calibrated on chat-tuned-model (Qwen/l22) runs.
#:
#: Calibrated on Pythia-2.8B, c50 (50 concepts, 6000 declared cells per
#: method): LinearProbe blanked 90/6000 (1.5%), including 7/500 at
#: ``factor == 0.0`` -- NO steering applied. DiffMean blanked 5/500 at
#: ``factor == 0.0`` too, at a comparable rate, and the blank rows differ
#: between the two methods at the same factor. This is honest sampler
#: output: Pythia is an untuned base model that has never seen an
#: instruction-formatted prompt, so it sometimes emits an immediate stop
#: token on this dataset's AlpacaEval-style prompts regardless of steering --
#: a rate a chat-tuned model essentially never exhibits, which is why the
#: Qwen-calibrated bound is a quarter of what a base model needs.
#:
#: 0.02 sits comfortably above the observed 1.5% with room for run-to-run
#: sampling variance, while still catching an actual wiring flood (the
#: failure this tripwire exists for): the collapse populations the Qwen bound
#: was calibrated against ran 2.75%+ blank, well above this bound too.
BASELINE_MAX_BLANK_GENERATION_FRACTION = 0.02


def merge_baseline_frames(
    main: pd.DataFrame,
    aux: pd.DataFrame,
    *,
    max_concepts: int,
) -> pd.DataFrame:
    """Validate and merge the main and aux vendored baseline frames."""

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

    main_factor_one = main.loc[main["factor"].eq(1.0)]
    stable_aux_columns = tuple(column for column in INPUT_COLUMNS if column != "steered_input")
    _require_matching_inputs(
        main_factor_one,
        aux,
        candidate_name="aux",
        columns=stable_aux_columns,
    )

    merged = _sorted(main)
    additions = aux.loc[:, [*KEY_COLUMNS, *method_columns(methods_for_source(FrameSource.AUX))]]
    merged = merged.merge(
        additions,
        on=list(KEY_COLUMNS),
        how="left",
        sort=False,
        validate="one_to_one",
    )
    merged = merged.loc[:, [*INPUT_COLUMNS, *method_columns(BASELINE_METHODS)]]
    validate_steering_frame(
        merged,
        source=FrameSource.MERGED,
        max_concepts=max_concepts,
        methods=BASELINE_METHODS,
        max_blank_fraction=BASELINE_MAX_BLANK_GENERATION_FRACTION,
    )
    return merged


def merge_baseline_parquets(
    main_path: Path,
    aux_path: Path,
    output_path: Path,
    *,
    max_concepts: int,
    force: bool = False,
) -> pd.DataFrame:
    """Read, validate, merge, then write two baseline steering Parquets."""

    if type(force) is not bool:
        raise TypeError("force must be a boolean")
    resolved_inputs = {Path(main_path).resolve(), Path(aux_path).resolve()}
    resolved_output = Path(output_path).resolve()
    if resolved_output in resolved_inputs:
        raise ValueError("merged output path must differ from every input path")
    if resolved_output.exists() and not force:
        raise FileExistsError(
            f"merged output already exists: {resolved_output}; pass force=True to replace it"
        )

    merged = merge_baseline_frames(
        pd.read_parquet(main_path),
        pd.read_parquet(aux_path),
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
    parser.add_argument("--output", type=Path, required=True, help="merged output Parquet")
    parser.add_argument("--max-concepts", type=_positive_int, required=True)
    parser.add_argument("--force", action="store_true", help="replace an existing merged output")
    args = parser.parse_args(argv)

    merged = merge_baseline_parquets(
        args.main,
        args.aux,
        args.output,
        max_concepts=args.max_concepts,
        force=args.force,
    )
    logger.info("Wrote {} validated rows to {}", len(merged), args.output)


if __name__ == "__main__":
    main()
