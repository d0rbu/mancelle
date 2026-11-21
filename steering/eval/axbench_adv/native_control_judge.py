"""Replay the authenticated standalone three-axis judge for the shared control."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import (
    _CALL_ACCOUNTING_FILE,
    _EVALUATION_FIELDS,
    FinalJudgeResult,
    validate_final_judge_outputs,
)
from steering.eval.axbench_adv.native_mailbox import digest_json, write_json
from steering.eval.axbench_adv.native_methods import (
    _METHOD_SLUG,
    METHODS,
    ZERO_CONTROL_ARCHIVED,
    _identity_zero_control_source,
    _validate_union,
    _validate_union_method,
)
from steering.eval.axbench_upstream.native_baseline_judge import SUMMARY
from steering.eval.axbench_upstream.native_contract import sha256


def control_lineage(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Portable exact lineage; generation identity is never relabelled as recovery."""

    if _identity_zero_control_source(identity) != ZERO_CONTROL_ARCHIVED:
        raise AxBenchAdvError("runtime zero control cannot claim archived replay lineage")
    return {
        "format": "native-control-judge-replay-v1",
        "run_identity_sha256": digest_json(identity),
        "union_provenance_sha256": identity["source_identity"]["union_provenance_sha256"],
        "union_files": identity["source_identity"]["union_files"],
    }


def replay_native_control(
    *,
    final_root: Path,
    identity: Mapping[str, Any],
    input_table: Any,
    emissions: Mapping[tuple[str, int, int], Any],
    output_dir: Path,
) -> FinalJudgeResult:
    """Project existing paid results, then apply the unchanged strict final validator.

    No judge request is made and no cache is changed. The standard source files
    were authenticated during generation; require those exact bytes again.
    """

    lineage = control_lineage(identity)
    if sha256(final_root / "union-provenance.json") != lineage["union_provenance_sha256"]:
        raise AxBenchAdvError("native control union provenance differs from generation")
    union, files = _validate_union(final_root)
    if files != lineage["union_files"]:
        raise AxBenchAdvError("native control union files differ from generation")
    if output_dir.exists():
        raise AxBenchAdvError("native control replay output already exists")
    frame = input_table.to_pandas()
    evaluated = frame.copy()
    concepts = tuple(identity["concept_ids"])
    entries = {
        concept: {"concept_id": concept, "results": {"LMJudgeEvaluator": {}}}
        for concept in concepts
    }
    for method in METHODS:
        generation, judged = _validate_union_method(final_root, union, method=method)
        factor = identity["best_factors"][method]
        selected = generation.loc[
            generation.concept_id.isin(concepts) & generation.factor.eq(factor)
        ].set_index(["concept_id", "input_id", "factor"])
        declared = frame.loc[frame.factor.eq(factor)]
        keys = list(
            declared[["concept_id", "input_id", "factor"]].itertuples(index=False, name=None)
        )
        selected = selected.loc[keys].reset_index()
        columns = [
            "dataset_name",
            "concept_id",
            "input_concept",
            "input_id",
            "factor",
            "original_prompt",
            f"{method}_steered_generation",
        ]
        if not selected[columns].equals(declared[columns].reset_index(drop=True)):
            raise AxBenchAdvError(f"native control source prompt/generation differs: {method}")
        source_frame = pq.read_table(
            final_root / _METHOD_SLUG[method] / "steering_data.parquet"
        ).to_pandas()
        source_frame = source_frame.set_index(["concept_id", "input_id", "factor"])
        for suffix in _EVALUATION_FIELDS:
            column = f"{method}_{suffix}"
            evaluated[column] = [
                source_frame.at[(row.concept_id, row.input_id, factor), column]
                if row.factor == factor
                else None
                for row in frame.itertuples(index=False)
            ]
        for concept in concepts:
            source_block = judged[concept]["results"]["LMJudgeEvaluator"][method]
            factor_index = source_block["factor"].index(factor)
            block = {
                "factor": [factor],
                **{name: [source_block[name][factor_index]] for name in SUMMARY},
            }
            for suffix, field in _EVALUATION_FIELDS.items():
                # Keep sparse off-grid slots null, including numeric columns.
                block[field] = [
                    source_frame.at[(concept, row.input_id, factor), f"{method}_{suffix}"]
                    if row.factor == factor
                    else None
                    for row in frame.loc[frame.concept_id.eq(concept)].itertuples(index=False)
                ]
            entries[concept]["results"]["LMJudgeEvaluator"][method] = block
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="native-control-replay-", dir=output_dir.parent
    ) as temp:
        staging = Path(temp) / "judged"
        staging.mkdir()
        evaluated.to_parquet(staging / "steering_data.parquet", index=False)
        (staging / "steering.jsonl").write_text(
            "".join(json.dumps(entry, allow_nan=False) + "\n" for entry in entries.values())
        )
        validate_final_judge_outputs(
            input_table=input_table,
            output_dir=staging,
            best_factors=identity["best_factors"],
            emissions=emissions,
            max_concepts=100,
            concept_ids=concepts,
        )
        write_json(staging / _CALL_ACCOUNTING_FILE, {"total_calls": 0, "total_cache_hits": 0})
        write_json(
            staging / "manifest.json",
            {
                "schema_version": 1,
                "best_factors": identity["best_factors"],
                "input_logical_sha256": logical_table_digest(input_table),
                "call_accounting": {"actual_evaluator_calls": 0, "paid_judge_cache_misses": 0},
                "control_replay": lineage,
                "files": {
                    name: {"sha256": sha256(staging / name)}
                    for name in (_CALL_ACCOUNTING_FILE, "steering.jsonl")
                }
                | {
                    "steering_data.parquet": {
                        "logical_sha256": logical_table_digest(
                            pq.read_table(staging / "steering_data.parquet")
                        ),
                    }
                },
            },
        )
        os.replace(staging, output_dir)
    return FinalJudgeResult(output_dir / "steering.jsonl", 0, 0)
