"""CLI for building paired AxBench-BPB source siblings for BPB-Adv."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from steering.eval.axbench_bpb_adv.source_cleaning import (
    SourceCleaningResult,
    clean_axbench_source_run,
    default_clean_output_dir,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a filter-derived clean AxBench-BPB source sibling whose "
            "synth parquets pass BPB-Adv fail-loud source validation."
        )
    )
    parser.add_argument(
        "--source-run-dir",
        type=Path,
        required=True,
        help="Stage 0-3 AxBench-BPB source run to clean.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Cleaned source output directory. Defaults to <source>_paired.",
    )
    parser.add_argument(
        "--no-filter-unaccepted",
        action="store_true",
        help=(
            "Keep qc_accepted=false rows when matching validator behavior. "
            "Use only for eval configs with filter_unaccepted_samples=false."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> SourceCleaningResult:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = default_clean_output_dir(args.source_run_dir)
    result = clean_axbench_source_run(
        args.source_run_dir,
        output_dir=output_dir,
        filter_unaccepted_samples=not args.no_filter_unaccepted,
    )
    print(f"output_dir={result.output_dir}")
    print(f"manifest_path={result.manifest_path}")
    print(f"contract_hash={result.validation.contract_hash}")
    print(f"n_dropped={result.validation.n_dropped}")
    print(f"positive_dropped_groups={len(result.positive.dropped_incomplete_groups)}")
    print(f"negative_dropped_groups={len(result.negative.dropped_incomplete_groups)}")
    return result


if __name__ == "__main__":
    main()
