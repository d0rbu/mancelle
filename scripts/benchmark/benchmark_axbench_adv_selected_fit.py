"""Compare full and shard-only project DiffMean fits on one loaded CUDA stack."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.source_contract import validate_source_contract
from steering.eval.axbench_upstream.methods import (
    fit_concept_dense_diffmeans,
    load_extraction_stack,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concept-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if not t.cuda.is_available():
        parser.error("requires CUDA")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    raw = OmegaConf.load(args.config)
    if not isinstance(raw, DictConfig):
        raise ValueError("config must be a mapping")
    config = build_config(raw)
    validated = validate_source_contract(
        paths=config.source_paths,
        expectations=config.source_expectations,
        expected_hashes=config.source_hashes,
        output_dir=output,
        reserved_paths=(config.judge_cache, config.reflection_cache),
        nl_attachment_paths=config.nl_attachment_paths,
        nl_attachment_expectations=config.nl_attachment_expectations,
    )
    training = dict(validated.training_texts)
    selected = tuple(args.concept_ids)
    if len(set(selected)) != len(selected) or not set(selected).issubset(training):
        raise ValueError("concept IDs must be unique source concept IDs")
    shard_training = {cid: examples for cid, examples in training.items() if cid in selected}
    model, sae = load_extraction_stack(validated.method_setup, config.device)
    report: dict[str, object] = {
        "device": t.cuda.get_device_name(config.device),
        "source_concepts": len(training),
        "selected_concept_ids": list(selected),
        "extraction_batch_size": validated.extraction_batch_size,
        "reuse_duplicate_texts": config.project_reuse_diffmean_texts,
        "rows": [],
    }
    baseline: dict[int, t.Tensor] | None = None
    rows = report["rows"]
    assert isinstance(rows, list)
    for scope in ("full", "selected", "selected", "full"):
        gc.collect()
        t.cuda.empty_cache()
        t.cuda.synchronize()
        t.cuda.reset_peak_memory_stats(config.device)
        start = time.perf_counter()
        fitted = fit_concept_dense_diffmeans(
            validated.method_setup,
            model,
            sae,
            training if scope == "full" else shard_training,
            extraction_batch_size=validated.extraction_batch_size,
            reuse_duplicate_texts=config.project_reuse_diffmean_texts,
        )
        t.cuda.synchronize()
        wall = time.perf_counter() - start
        if baseline is None:
            baseline = {cid: fitted[cid].clone() for cid in selected}
        exact = all(t.equal(fitted[cid], baseline[cid]) for cid in selected)
        row = {
            "scope": scope,
            "wall_seconds": wall,
            "exact_selected_directions": exact,
            "maximum_absolute_difference": max(
                float((fitted[cid] - baseline[cid]).abs().max()) for cid in selected
            ),
            "peak_cuda_allocated_bytes": t.cuda.max_memory_allocated(config.device),
        }
        rows.append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not exact:
            raise AssertionError("selected DiffMean direction differs from full fit")


if __name__ == "__main__":
    main()
