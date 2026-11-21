"""Synthetic reverse-order harmfulness YAMLs -> PNG script output."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from scripts.eval.plot_harmfulness_reversed_progression import _main as plot_main

SAFETY_DELTA = (
    "safety_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/"
    "max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/"
    "seed_42/r_mean_pct_incl_zeros/m0.5"
)
CAPABILITY_DELTA = (
    "capabilities_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/"
    "max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/"
    "seed_42/r_mean_pct_incl_zeros/m0.5"
)


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _result(source: str, feature_configs: str, value: float) -> dict:
    return {
        "harmfulness_eval_schema_version": 4,
        "config": {
            "model_tag": "d12_baseline",
            "num_harmful": 700,
            "source": source,
            "feature_configs": feature_configs,
        },
        "harmful_prompts": {
            "asr": value,
            "safe_response_rate": 1.0 - value,
        },
        "benign_prompts": {"compliance_rate": 0.9 - value / 2},
        "overall": {
            "mean_score": 1.0 + value,
            "safe_response_rate": 1.0 - value,
        },
    }


def test_reversed_plot_main_writes_expected_pngs(tmp_path: Path) -> None:
    Image = pytest.importorskip("PIL.Image")

    base_dir = tmp_path / "nanochat_cache"
    out_dir = tmp_path / "fig"

    records = [
        ("base", SAFETY_DELTA, 0.20),
        ("base", CAPABILITY_DELTA, 0.30),
        ("base/capabilities_cond", SAFETY_DELTA, 0.25),
        ("base/capabilities_cond", CAPABILITY_DELTA, 0.35),
        ("base/capabilities_cond/safety_cond", SAFETY_DELTA, 0.15),
        ("base/capabilities_cond/safety_cond", CAPABILITY_DELTA, 0.40),
    ]
    for idx, (source, feature_configs, value) in enumerate(records):
        _write_yaml(
            base_dir / "harmfulness_results" / f"{idx}" / "result.yaml",
            _result(source, feature_configs, value),
        )

    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(base_dir),
            "output_dir": str(out_dir),
            "num_harmful": 700,
        }
    )
    plot_main(cfg)

    expected = [
        "reverse_conditioned_asr.png",
        "reverse_conditioned_harmful_safe_response_rate.png",
        "reverse_conditioned_benign_compliance_rate.png",
        "reverse_conditioned_mean_score.png",
    ]
    for name in expected:
        path = out_dir / name
        assert path.exists(), f"{path} missing"
        assert path.stat().st_size > 0

    with Image.open(out_dir / "reverse_conditioned_asr.png") as img:
        w, h = img.size
        assert w > 100 and h > 100
