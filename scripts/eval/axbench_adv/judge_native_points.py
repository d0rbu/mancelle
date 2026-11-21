"""Judge staged native Adv point frames with the standard three AxBench metrics."""
from __future__ import annotations

from pathlib import Path

import hydra
import pyarrow.parquet as pq
from omegaconf import DictConfig

from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.execution_scope import validate_concept_ids
from steering.eval.axbench_adv.final_judge import (
    build_point_frame,
    run_final_judge,
    validate_final_judge_outputs,
)
from steering.eval.axbench_adv.native_mailbox import checked_raw, read_json, write_json
from steering.eval.axbench_adv.types import NATIVE_METHOD, PROPOSAL_COUNTS
from steering.eval.axbench_upstream.native_contract import sha256


@hydra.main(version_base=None, config_path="../../../configs/eval/axbench_adv", config_name="native_final_judge")
def main(cfg: DictConfig) -> None:
    root = Path(str(cfg.points_dir))
    receipts = []
    levels = (0, *tuple(cfg.proposal_counts))
    expected_concepts = validate_concept_ids(
        range(cfg.max_concepts) if cfg.concept_ids is None else cfg.concept_ids, max_concepts=cfg.max_concepts)
    if len(set(levels)) != len(levels) or any(type(level) is not int or level not in (0, *PROPOSAL_COUNTS) for level in levels):
        raise ValueError("native final levels must be zero plus unique declared GEPA proposal counts")
    prepared = []
    for level in levels:
        point = root / f"p{level}"
        if not (point / "inference/steering_data.parquet").is_file() or not (point / "emissions.json").is_file():
            raise ValueError(f"required native point is incomplete: {point}")
        table = pq.read_table(point / "inference/steering_data.parquet")
        emissions = {}
        for entry in read_json(point / "emissions.json")["emissions"]:
            if entry["method"] != NATIVE_METHOD:
                raise ValueError("native final judge cannot evaluate other methods")
            key = (entry["method"], entry["concept_id"], entry["input_id"])
            if key in emissions:
                raise ValueError("duplicate native point emission")
            emissions[key] = checked_raw(entry["raw"], entry["raw"]["generation"])
        expected = build_point_frame(source_frame=table.to_pandas(), best_factors={NATIVE_METHOD: 1.0},
                                     emissions=emissions, max_concepts=cfg.max_concepts, concept_ids=expected_concepts)
        if not table.equals(expected):
            raise ValueError("native point frame population or generations differ from emissions")
        prepared.append((point, table, emissions))
    for point, table, emissions in prepared:
        input_path = point / "inference/steering_data.parquet"
        output = Path(str(cfg.output_dir)) / point.name
        source_identity = {"input_sha256": sha256(input_path),
                           "emissions_sha256": sha256(point / "emissions.json")}
        receipt_path = Path(str(cfg.output_dir)) / f"{point.name}-receipt.json"
        if output.exists():
            manifest = read_json(output / "manifest.json")
            files = {name: {"sha256": sha256(output / name)}
                     for name in ("steering.jsonl", "judge_call_accounting.json")}
            files["steering_data.parquet"] = {"logical_sha256": logical_table_digest(pq.read_table(output / "steering_data.parquet"))}
            if manifest["files"] != files or manifest["input_logical_sha256"] != logical_table_digest(table):
                raise ValueError("native completed judge manifest does not match input or outputs")
            if receipt_path.exists() and read_json(receipt_path)["source"] != source_identity:
                raise ValueError("native point source differs from completed judge receipt")
            validate_final_judge_outputs(input_table=table, output_dir=output, best_factors={NATIVE_METHOD: 1.0},
                                         emissions=emissions, max_concepts=cfg.max_concepts, concept_ids=expected_concepts)
            accounting = manifest["call_accounting"]
        else:
            result = run_final_judge(
                input_table=table, input_path=input_path, output_dir=output, cache_path=Path(str(cfg.judge_cache)),
                best_factors={NATIVE_METHOD: 1.0}, emissions=emissions,
                max_concepts=cfg.max_concepts, concept_ids=expected_concepts,
            )
            accounting = {"actual_evaluator_calls": result.actual_evaluator_calls,
                          "paid_judge_cache_misses": result.paid_judge_cache_misses}
        receipt = {"point": point.name, "steering_jsonl": str(output / "steering.jsonl"),
                   **accounting, "source": source_identity}
        write_json(receipt_path, receipt)
        receipts.append(receipt)
    write_json(Path(str(cfg.output_dir)) / "receipt.json", {"method": NATIVE_METHOD, "points": receipts})


if __name__ == "__main__":
    main()
