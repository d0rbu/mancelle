"""Standalone replotter for AxBench-BPB-Adv steering-strength family plots."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from nanochat.atomic_io import atomic_write_parquet, atomic_write_text
from steering.eval.axbench_bpb_adv.aggregation import (
    PRIMARY_MARGIN_YLABEL,
    ROUNDTRIP_PRIMARY_YLABEL,
    _single_group_corpus,
    plot_steering_strength_families,
)
from steering.eval.axbench_bpb_adv.config import SteeringMetric
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replot AxBench-BPB-Adv steering-strength family plots."
    )
    parser.add_argument(
        "--output-root",
        default="out/eval/axbench_bpb_adv",
        help="Root containing run-tag subdirectories.",
    )
    parser.add_argument(
        "--run-tag",
        action="append",
        required=True,
        help="Run tag to plot. Repeat to concatenate several runs.",
    )
    parser.add_argument(
        "--metric",
        choices=[metric.value for metric in SteeringMetric],
        default=SteeringMetric.ROUNDTRIP.value,
        help="Aggregate family to read.",
    )
    parser.add_argument(
        "--plot-method",
        action="append",
        default=None,
        help="Method-name whitelist. Repeat to keep several methods.",
    )
    parser.add_argument(
        "--plots-dir",
        default=None,
        help=(
            "Output directory for plots. Required when concatenating more than "
            "one run tag; defaults to the run's steering_strength_plots dir for "
            "a single run."
        ),
    )
    return parser


def _metric_artifact(
    paths: AxBenchBpbAdvPaths,
    metric: SteeringMetric,
) -> tuple[Path, pa.Schema, str, str]:
    if metric is SteeringMetric.ROUNDTRIP:
        return (
            paths.roundtrip_aggregates_dir() / "macro.parquet",
            ROUNDTRIP_MACRO_AGG_SCHEMA,
            "primary_roundtrip_y",
            ROUNDTRIP_PRIMARY_YLABEL,
        )
    return (
        paths.aggregates_dir() / "macro.parquet",
        MACRO_AGG_SCHEMA,
        "primary_margin_y",
        PRIMARY_MARGIN_YLABEL,
    )


def _read_contract_hash(contract_path: Path, *, run_tag: str) -> str:
    """The run's validated ``contract_hash``. Every replotted run must carry one.

    ``validate_source`` writes ``source_contract.json`` for every real run, so a
    missing contract means an incomplete/hand-copied artifact — always rejected
    (fail-loud, issue #635), single-run or combined alike. A PRESENT-but-malformed
    contract (bad JSON / no ``contract_hash`` key) is likewise a loud, named error,
    never a bare ``KeyError``/``JSONDecodeError``. The returned hash is only
    COMPARED across runs when combining (>1 run), but it is required and recorded
    in provenance for every run so a replot always states its validated source.
    """
    if not contract_path.exists():
        raise AxBenchBpbAdvError(
            f"run {run_tag!r} has no source_contract.json at {contract_path}; "
            "every replotted run must carry its validated source contract "
            "(re-run stage=validate_source for this run)"
        )
    try:
        return str(json.loads(contract_path.read_text())["contract_hash"])
    except (json.JSONDecodeError, KeyError, OSError) as exc:
        raise AxBenchBpbAdvError(
            f"run {run_tag!r} has a malformed source_contract.json at "
            f"{contract_path}: {exc}"
        ) from exc


def _read_macro(path: Path, *, schema: pa.Schema) -> pa.Table:
    if not path.exists():
        raise AxBenchBpbAdvError(f"macro aggregate does not exist: {path}")
    actual_schema = pq.read_schema(path)
    if not actual_schema.equals(schema):
        raise AxBenchBpbAdvError(
            f"macro aggregate schema mismatch for {path}: expected {schema}, "
            f"got {actual_schema}"
        )
    return pq.read_table(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    metric = SteeringMetric.coerce(args.metric)
    run_tags = [str(tag) for tag in args.run_tag]
    if len(run_tags) > 1 and args.plots_dir is None:
        raise AxBenchBpbAdvError("--plots-dir is required with multiple --run-tag values")

    tables: list[pa.Table] = []
    run_infos: list[dict[str, object]] = []
    output_dir: Path | None = None
    metric_column = ""
    metric_label = ""
    expected_schema: pa.Schema | None = None
    for run_tag in run_tags:
        paths = AxBenchBpbAdvPaths(output_root=args.output_root, run_tag=run_tag)
        macro_path, schema, metric_column, metric_label = _metric_artifact(paths, metric)
        if expected_schema is None:
            expected_schema = schema
        elif not schema.equals(expected_schema):
            raise AxBenchBpbAdvError("internal metric schema mismatch across run tags")
        tables.append(_read_macro(macro_path, schema=schema))
        if output_dir is None:
            output_dir = paths.steering_strength_plots_dir()

        contract_hash = _read_contract_hash(paths.source_contract(), run_tag=run_tag)
        run_infos.append(
            {
                "run_tag": run_tag,
                "macro_path": str(macro_path),
                "contract_hash": contract_hash,
                "n_macro_rows": tables[-1].num_rows,
            }
        )

    # Cross-run equality is only meaningful when COMBINING (D3); a single run has
    # nothing to compare against (and, per _read_contract_hash, may carry no
    # contract at all).
    if len(run_tags) > 1:
        distinct_hashes = {str(info["contract_hash"]) for info in run_infos}
        if len(distinct_hashes) > 1:
            detail = "; ".join(f"{info['run_tag']}={info['contract_hash']}" for info in run_infos)
            raise AxBenchBpbAdvError(
                "combined frontier requires every run to share one validated source "
                f"(equal contract_hash, spec D3); got {detail}"
            )

    if args.plots_dir is not None:
        output_dir = Path(args.plots_dir)
    if output_dir is None:
        raise AxBenchBpbAdvError("no output directory resolved")
    macro = pa.concat_tables(tables, promote_options="none")
    written = plot_steering_strength_families(
        macro,
        output_dir,
        metric_column=metric_column,
        metric_label=metric_label,
        plot_methods=args.plot_method,
    )

    # Provenance records the DRAWN figure's corpus, so mirror the plot's
    # plot_methods whitelist, then reduce via the SAME canonical corpus reducer
    # the macro rollup uses (_single_group_corpus). plot_steering_strength_families
    # has already validated the drawn rows carry one corpus, so this is a shared
    # fail-loud re-check (catches the two whitelist filters drifting) rather than a
    # third hand-rolled copy of the corpus-consistency rule.
    provenance_rows = macro.to_pylist()
    if args.plot_method is not None:
        keep = set(args.plot_method)
        provenance_rows = [row for row in provenance_rows if str(row["method_name"]) in keep]
    provenance_corpus = _single_group_corpus(
        provenance_rows, group=("combined-frontier provenance",)
    )
    if len(run_tags) > 1:
        atomic_write_parquet(macro, output_dir / "combined_macro.parquet")
    atomic_write_text(
        yaml.safe_dump(
            {
                "runs": run_infos,
                "metric": metric.value,
                "plot_methods": args.plot_method,
                "general_val_corpus": provenance_corpus,
                "written": [str(p) for p in written],
            },
            sort_keys=True,
        ),
        output_dir / "provenance.yaml",
    )

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
