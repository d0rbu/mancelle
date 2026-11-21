"""End-to-end: synthetic harmfulness YAML tree + synthetic progression
JSONs → run the script CLI's _main(cfg) in-process → assert all expected PNGs
are created with non-zero size and reload-able dimensions.

Note: PIL is not a declared project dependency; pytest.importorskip is used so
this test is skipped gracefully in environments that lack Pillow. The import is
deferred inside the test function intentionally (one of the rare justified cases
of a late import), guarded by pytest.importorskip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from scripts.eval.plot_harmfulness_progression import _main as plot_main

SAFETY_DELTA = (
    "safety_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/"
    "neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/"
    "r_mean_pct_incl_zeros/m0.5"
)


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _ok(model_tag: str, source: str, feature_configs: str) -> dict:
    return {
        "harmfulness_eval_schema_version": 4,
        "config": {
            "model_tag": model_tag,
            "num_harmful": 700,
            "source": source,
            "feature_configs": feature_configs,
        },
        "harmful_prompts": {"asr": 0.1, "safe_response_rate": 0.9},
        "benign_prompts": {"compliance_rate": 0.8},
        "overall": {"mean_score": 1.5, "safe_response_rate": 0.7},
    }


def test_full_pipeline_to_png(tmp_path: Path) -> None:
    # PIL is not a declared project dep; skip gracefully if unavailable.
    Image = pytest.importorskip("PIL.Image")

    base_dir = tmp_path / "nanochat_cache"
    out_dir = tmp_path / "out"

    # Two records in the conditioned SFT chain.
    _write_yaml(
        base_dir / "harmfulness_results" / "a" / "x.yaml",
        _ok(
            model_tag="d12_baseline",
            source="base/safety_cond",
            feature_configs=SAFETY_DELTA,
        ),
    )
    _write_yaml(
        base_dir / "harmfulness_results" / "b" / "y.yaml",
        _ok(
            model_tag="d12_baseline",
            source="base/safety_cond/capabilities_cond",
            feature_configs=SAFETY_DELTA,
        ),
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
        "conditioned_asr.png",
        "conditioned_harmful_safe_response_rate.png",
        "conditioned_benign_compliance_rate.png",
        "conditioned_mean_score.png",
    ]
    for name in expected:
        p = out_dir / name
        assert p.exists(), f"{p} missing"
        assert p.stat().st_size > 0

    # Sanity: load one PNG and check sane dimensions.
    with Image.open(out_dir / "conditioned_asr.png") as img:
        w, h = img.size
        assert w > 100 and h > 100, f"unexpected PNG size {w}x{h}"
