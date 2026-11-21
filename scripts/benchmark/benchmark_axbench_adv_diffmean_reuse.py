"""Pair batch-one project DiffMean fits with exact repeated-text reuse on CUDA."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections import Counter
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
    if validated.extraction_batch_size != 1:
        raise ValueError("repeated-text benchmark requires source extraction_batch_size=1")
    training = dict(validated.training_texts)
    counts = Counter(
        text for examples in training.values()
        for text in (*examples.positive, *examples.negative)
    )
    model, sae = load_extraction_stack(validated.method_setup, config.device)
    report: dict[str, object] = {
        "device": t.cuda.get_device_name(config.device),
        "concepts": len(training),
        "text_rows": sum(counts.values()),
        "unique_texts": len(counts),
        "repeated_texts": sum(count > 1 for count in counts.values()),
        "rows": [],
    }
    baseline: dict[int, t.Tensor] | None = None
    rows = report["rows"]
    assert isinstance(rows, list)
    for reuse in (False, True, True, False):
        gc.collect()
        t.cuda.empty_cache()
        t.cuda.synchronize()
        t.cuda.reset_peak_memory_stats(config.device)
        start = time.perf_counter()
        fitted = fit_concept_dense_diffmeans(
            validated.method_setup, model, sae, training,
            extraction_batch_size=1, reuse_duplicate_texts=reuse,
        )
        t.cuda.synchronize()
        wall = time.perf_counter() - start
        exact = baseline is None or all(
            t.equal(value, baseline[concept_id])
            for concept_id, value in fitted.items()
        )
        max_diff = 0.0 if baseline is None else max(
            float((value - baseline[concept_id]).abs().max())
            for concept_id, value in fitted.items()
        )
        if baseline is None:
            baseline = {concept_id: value.clone() for concept_id, value in fitted.items()}
        row = {
            "reuse_duplicate_texts": reuse,
            "wall_seconds": wall,
            "exact_baseline": exact,
            "maximum_absolute_difference": max_diff,
            "peak_cuda_allocated_bytes": t.cuda.max_memory_allocated(config.device),
        }
        rows.append(row)
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)
        if not exact:
            raise AssertionError("reused DiffMean directions differ from batch-one baseline")


if __name__ == "__main__":
    main()
