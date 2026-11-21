"""Ingress, exact positive grids and histogram support for soft suffixes."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_adv.pressures import COUNTER_PRESSURES, LEGACY_PRESSURES, Pressure
from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv import pipeline
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample, RawEvaluation
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_histograms import generate_histogram_reports
from tests.axbench_adv.unit.test_gcg_pipeline_integration import _write_histogram_point
from tests.axbench_adv.unit.test_run_config import _raw
from tests.axbench_adv.unit.test_soft_prompt_driver import training_pairs


def test_soft_hydra_defaults_and_closed_ingress_preserve_rng() -> None:
    yaml = OmegaConf.load(
        Path(__file__).resolve().parents[3] / "configs/eval/axbench_adv/axbench_adv.yaml"
    )
    raw = _raw()
    raw.update(pressures=["soft_prompt"], soft_prompt=OmegaConf.to_container(yaml.soft_prompt))
    python_state = random.getstate()
    config = build_config(OmegaConf.create(raw))
    assert random.getstate() == python_state
    assert config.soft_prompt == AxBenchAdvSoftPromptConfig()
    assert config.pressures == [Pressure.SOFT_PROMPT]
    assert Pressure.SOFT_PROMPT not in LEGACY_PRESSURES
    assert Pressure.SOFT_PROMPT not in COUNTER_PRESSURES
    raw["soft_prompt"] = {**asdict(config.soft_prompt), "candidate_batch_size": 16}
    with pytest.raises(AxBenchAdvError, match="exactly"):
        build_config(OmegaConf.create(raw))


@pytest.mark.parametrize("grid", [{}, {0: [], 1: []}, {2: []}])
def test_cell_rejects_missing_extra_or_control_search_results(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, grid: dict
) -> None:
    class Runtime:
        def evaluate(self, suffix, input_ids):
            assert suffix == ""
            return [
                RawEvaluation(
                    generation="control",
                    concept_rating=1.0,
                    fluency_rating=1.0,
                    judge_rationale="{}",
                    paid_judge_cache_misses=0,
                )
                for _ in input_ids
            ]

    monkeypatch.setattr(pipeline, "execute_soft_prompt_cell", lambda **_kwargs: grid)
    config = SimpleNamespace(
        pressures=(Pressure.SOFT_PROMPT,),
        master_seed=42,
        soft_prompt=AxBenchAdvSoftPromptConfig(optimized_token_counts=(0, 1)),
    )
    with pytest.raises(AxBenchAdvError, match="token-count grid"):
        pipeline._execute_cell(
            config=config,  # ty: ignore[invalid-argument-type]
            runtime=Runtime(),  # ty: ignore[invalid-argument-type]
            reflection_lm=None,
            optimizer=None,  # ty: ignore[invalid-argument-type]
            output_dir=tmp_path,
            method="DiffMean",
            concept_id=0,
            concept="topic 0",
            factor=1.0,
            optimization_examples=[
                OptimizationExample(input_id=i, user_message=f"eval {i}") for i in range(10)
            ],
            emissions={0: {}},
            totals={"control_generation_calls": 0, "control_paid_judge_cache_misses": 0},
            counter_emissions={},
            gcg_emissions={},
            soft_prompt_emissions={1: {}},
            directions={},
            training_pairs=training_pairs(),
        )


def test_soft_histograms_read_the_manifest_k_grid_and_shared_control(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_histogram_point(root / "points/p0/judged", shift=0.0)
    _write_histogram_point(root / "soft_prompt/points/k2/judged", shift=0.1)
    manifest = {
        "schema_version": 2,
        "methods": ["DiffMean"],
        "max_concepts": 2,
        "concept_ids": [0, 1],
        "best_factors": {"DiffMean": 1.0},
        "pressures": ["soft_prompt"],
        "soft_prompt": {"optimized_token_counts": [0, 2]},
        "files": {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    path = generate_histogram_reports(
        input_dir=root,
        output_dir=tmp_path / "histograms",
        pipeline="adv",
        pressure="soft_prompt",
        method_factors={},
    )
    result = json.loads(path.read_text())
    assert result["pressure"] == "soft_prompt"
    assert set(result["reports"]) == {"k0", "k2"}
    assert (path.parent / "k0/score_histograms.png").is_file()
    assert (path.parent / "k2/score_histograms.png").is_file()
