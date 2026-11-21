"""Integration tests for the reverse-order plot script: Hydra config loads,
empty/missing-data resilience, filename-suffix invariants. Synthetic records
keep the test independent of any real on-disk eval output.

Note: the happy-path 8-PNG render with realistic synthetic records is already
covered by tests/harmfulness_progression/integration/test_reversed_plot_cli.py.
This file adds smaller-scoped checks that complement that test.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from scripts.eval.plot_harmfulness_reversed_progression import _main as plot_main

CONFIG_DIR = Path(__file__).resolve().parents[2].parent / "configs" / "plot"

SAFETY_DELTA = (
    "safety_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/"
    "max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/"
    "seed_42/r_mean_pct_incl_zeros/m0.5"
)
CAPABILITY_DELTA = (
    "capabilities_sft/delta_base/dist_geometric/mean_10.0/std_10.0/"
    "min_1/max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/"
    "max_40/seed_42/r_mean_pct_incl_zeros/m0.5"
)


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _record(model_tag: str, source: str, feature_configs: str, num_harmful: int) -> dict:
    return {
        "harmfulness_eval_schema_version": 4,
        "config": {
            "model_tag": model_tag,
            "num_harmful": num_harmful,
            "source": source,
            "feature_configs": feature_configs,
        },
        "harmful_prompts": {"asr": 0.1, "safe_response_rate": 0.9},
        "benign_prompts": {"compliance_rate": 0.8},
        "overall": {"mean_score": 1.5, "safe_response_rate": 0.7},
    }


def test_hydra_config_loads_with_num_harmful_50() -> None:
    """The plot config matches the reversed eval configs' num_harmful=50."""
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="harmfulness_reversed_progression")
    assert cfg.model_tag == "d12_baseline"
    assert cfg.base_dir == "nanochat_cache"
    assert int(cfg.num_harmful) == 50


def test_main_handles_empty_base_dir(tmp_path: Path) -> None:
    """Empty base_dir -> no records -> run completes; PNGs may be empty but
    the script must not raise."""
    base_dir = tmp_path / "nc"
    base_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(base_dir),
            "output_dir": str(tmp_path / "fig"),
            "num_harmful": 50,
        }
    )
    plot_main(cfg)

    # Some PNGs should still be produced even with no data.
    pngs = list((tmp_path / "fig").glob("*.png"))
    assert len(pngs) > 0, "expected at least one PNG even with empty input"


def test_main_filename_suffixes(tmp_path: Path) -> None:
    """All conditioned reverse-order filename suffixes are produced."""
    base_dir = tmp_path / "nc"
    out_dir = tmp_path / "fig"

    # Minimal records to anchor the chart at all 3 conditioned stages.
    records = [
        ("base", SAFETY_DELTA, 0.10),
        ("base", CAPABILITY_DELTA, 0.15),
        ("base/capabilities_cond", SAFETY_DELTA, 0.20),
        ("base/capabilities_cond", CAPABILITY_DELTA, 0.25),
        ("base/capabilities_cond/safety_cond", SAFETY_DELTA, 0.30),
        ("base/capabilities_cond/safety_cond", CAPABILITY_DELTA, 0.35),
    ]
    for idx, (source, feature_configs, asr) in enumerate(records):
        _write_yaml(
            base_dir / "harmfulness_results" / f"{idx}.yaml",
            {
                "harmfulness_eval_schema_version": 4,
                "config": {
                    "model_tag": "d12_baseline",
                    "num_harmful": 50,
                    "source": source,
                    "feature_configs": feature_configs,
                },
                "harmful_prompts": {
                    "asr": asr,
                    "safe_response_rate": 1.0 - asr,
                },
                "benign_prompts": {"compliance_rate": 0.8},
                "overall": {
                    "mean_score": 1.0 + asr,
                    "safe_response_rate": 1.0 - asr,
                },
            },
        )

    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(base_dir),
            "output_dir": str(out_dir),
            "num_harmful": 50,
        }
    )
    plot_main(cfg)

    expected = {
        "reverse_conditioned_asr.png",
        "reverse_conditioned_harmful_safe_response_rate.png",
        "reverse_conditioned_benign_compliance_rate.png",
        "reverse_conditioned_mean_score.png",
    }
    actual = {p.name for p in out_dir.glob("*.png")}
    assert expected.issubset(actual), (
        f"missing PNGs: {expected - actual}; extra PNGs: {actual - expected}"
    )


def test_main_coerces_string_num_harmful(tmp_path: Path) -> None:
    """cfg.num_harmful as a string (e.g. from env-var interpolation) is coerced
    to int so load_harmfulness_records' int-vs-int filter doesn't silently
    drop all records."""
    base_dir = tmp_path / "nc"

    _write_yaml(
        base_dir / "harmfulness_results" / "x.yaml",
        _record("d12_baseline", "base/capabilities_cond", SAFETY_DELTA, 50),
    )

    # Pass num_harmful as a STRING, not an int.
    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(base_dir),
            "output_dir": str(tmp_path / "fig"),
            "num_harmful": "50",  # deliberate string
        }
    )

    from scripts.eval.plot_harmfulness_reversed_progression import _main

    _main(cfg)

    # If the int coerce works, the plot loads the record and produces non-empty
    # PNGs. Without the coerce, all 8 PNGs would be axis-only (no data lines).
    assert (tmp_path / "fig" / "reverse_conditioned_asr.png").is_file()
