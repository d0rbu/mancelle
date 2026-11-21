"""Collect judged Campaign 2 cells and plot one line per variant/strength."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from scripts.eval.campaign2 import collect_records
from scripts.eval.plot_harmfulness_progression import save_harmfulness_progression_plots
from scripts.utils.campaign2 import Stage, install_json_once, load_strict_json_object
from steering.eval.campaign2 import Variant, strength_leaf
from steering.eval.chat_eval_result_identity import semantic_digest
from steering.eval.harmfulness_artifact import validate_harmfulness_payload
from steering.eval.harmfulness_progression import (
    HarmfulnessRecord,
    _parse_harmfulness_metrics,
)

VARIANT_LABELS = {
    Variant.ENCDEC: "Encoder-decoder",
    Variant.ENCDEC_SAFETY: "Encoder-decoder safety",
    Variant.PYTHIA: "Pythia",
    Variant.PYTHIA_DIFFMEAN: "Pythia DiffMean",
    Variant.PYTHIA_PROMPT: "Pythia safety prompt",
}

def collect_cells(cfg: DictConfig) -> list[dict[str, Any]]:
    variants = [Variant(value) for value in cfg.variants]
    if not variants or len(variants) != len(set(variants)):
        raise ValueError("progression requires distinct nonempty variants")
    strengths = []
    if Variant.ENCDEC_SAFETY in variants:
        values = list(cfg.safety_strengths)
        leaves = [strength_leaf(value) for value in values]
        if not leaves or len(leaves) != len(set(leaves)):
            raise ValueError("progression requires distinct nonempty safety_strengths")
        strengths = [float(value) for value in values]
    canonical_prompts = None
    cells = []
    for stage in Stage:
        for variant in variants:
            sweep_identity = None
            for strength in strengths if variant is Variant.ENCDEC_SAFETY else [None]:
                root = Path(cfg.input_root) / stage.value / variant.value
                if strength is not None:
                    root = root / strength_leaf(strength)
                identity = load_strict_json_object(root / "identity.json")
                config = identity["config"]
                if config["stage"] != stage.value or config["variant"] != variant.value:
                    raise ValueError("cell identity disagrees with progression directory")
                if strength is not None and strength_leaf(config["safety"]["strength"]) != strength_leaf(strength):
                    raise ValueError("cell strength identity disagrees with progression directory")
                if strength is not None:
                    comparison = json.loads(json.dumps(identity))
                    comparison["config"].pop("output_dir", None)
                    comparison["config"]["safety"].pop("strength")
                    completion = comparison["safety_binding"]["completion"]
                    for block in (completion, completion["safety_steering"]):
                        if strength_leaf(block.pop("strength")) != strength_leaf(strength):
                            raise ValueError("safety completion strength disagrees with cell identity")
                    if sweep_identity is None:
                        sweep_identity = comparison
                    elif comparison != sweep_identity:
                        raise ValueError("strength sweep cells have different target/checkpoint/evaluation identities")
                prompts = json.loads((root / "prompts.json").read_text())
                if canonical_prompts is None:
                    canonical_prompts = prompts
                elif prompts != canonical_prompts:
                    raise ValueError("progression cells have different ordered prompt pools")
                records = collect_records(root, prompts, identity if strength is not None else None)
                if strength is not None:
                    receipt = load_strict_json_object(root / "responses.json")
                    if receipt != {"cell_sha256": semantic_digest(identity), "records_sha256": semantic_digest(records)}:
                        raise ValueError("safety response artifact disagrees with cell identity or records")
                results = load_strict_json_object(root / "results.json")
                validate_harmfulness_payload(results, model=config["judge_model"], pairs=records)
                cell = {"stage": stage.value, "variant": variant.value, "identity": identity, "results": results}
                if strength is not None:
                    cell["strength"] = strength
                cells.append(cell)
    return cells

def plot_records(cells: list[dict[str, Any]]) -> list[HarmfulnessRecord]:
    records = []
    for cell in cells:
        variant = Variant(cell["variant"])
        label = VARIANT_LABELS[variant]
        if variant is Variant.ENCDEC_SAFETY:
            label += f" (strength={format(cell['strength'], '.17g')})"
        source_path = Path("campaign2") / cell["stage"] / cell["variant"] / "results.json"
        records.append(HarmfulnessRecord(
            model_tag="Campaign 2",
            source=cell["stage"],
            signal=label,
            full_conditioning=variant in (Variant.ENCDEC, Variant.ENCDEC_SAFETY),
            metrics=_parse_harmfulness_metrics(cell["results"], source_path),
        ))
    return records

@hydra.main(version_base=None, config_path="../../configs/eval", config_name="campaign2_progression")
def main(cfg: DictConfig) -> None:
    cells = collect_cells(cfg)
    install_json_once(
        Path(cfg.output_file), {"format": "campaign2_progression_v1", "cells": cells}
    )
    if cfg.get("plots_dir") is not None:
        save_harmfulness_progression_plots(
            plot_records(cells),
            Path(cfg.plots_dir),
            source_to_stage={stage.value: index for index, stage in enumerate(Stage)},
            stage_labels=tuple(stage.value.title() for stage in Stage),
            filename_prefix="campaign2",
            title_template="Campaign 2: {ylabel}",
        )

if __name__ == "__main__":
    main()
