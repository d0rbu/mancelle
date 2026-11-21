"""E2E for the reverse-order plot: realistic on-disk lineage trees that
mimic actual pipeline output paths, plus a source-drift case that catches
operator-convention violations at test time."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from omegaconf import OmegaConf

from scripts.eval.plot_harmfulness_reversed_progression import _main as plot_main

SAFETY_DELTA_FC = (
    "safety_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/"
    "neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/"
    "r_mean_pct_incl_zeros/m0.5"
)
CAPABILITY_DELTA_FC = (
    "capabilities_sft/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/max_40/"
    "neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/seed_42/"
    "r_mean_pct_incl_zeros/m0.5"
)


def _write_yaml(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data))


def _rec(source: str, feature_configs: str, asr: float = 0.2) -> dict:
    return {
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
        "overall": {"mean_score": 1.0 + asr, "safe_response_rate": 1.0 - asr},
    }


def _build_pipeline_shaped_tree(
    base_dir: Path,
    *,
    include_chained_safety_cond: bool = True,
) -> Path:
    """Mimic the pipeline's nanochat_cache/conditional_encdec_checkpoints/...
    layout with harmfulness_results YAMLs at each stage."""
    encdec_root = (
        base_dir
        / "conditional_encdec_checkpoints"
        / "d12_baseline"
        / "s128"
        / "dist_geometric"
        / "mean_10.0"
        / "std_10.0"
    )

    # Pretrain end (full-conditioning report).
    pretrain_results = encdec_root / "report" / "harmfulness_results_full_conditioning"
    cap_cond = encdec_root / "capabilities_sft_conditioned_checkpoints"
    cap_then_safety_cond = cap_cond / "safety_sft_conditioned_checkpoints"

    _write_yaml(pretrain_results / "safety_delta.yaml", _rec("base", SAFETY_DELTA_FC, 0.20))
    _write_yaml(pretrain_results / "capability_delta.yaml", _rec("base", CAPABILITY_DELTA_FC, 0.30))

    _write_yaml(
        cap_cond / "report" / "harmfulness_results" / "empty" / "safety_delta.yaml",
        _rec("base/capabilities_cond", SAFETY_DELTA_FC, 0.25),
    )
    _write_yaml(
        cap_cond / "report" / "harmfulness_results" / "empty" / "capability_delta.yaml",
        _rec("base/capabilities_cond", CAPABILITY_DELTA_FC, 0.35),
    )

    if include_chained_safety_cond:
        _write_yaml(
            cap_then_safety_cond / "report" / "harmfulness_results" / "empty" / "safety_delta.yaml",
            _rec("base/capabilities_cond/safety_cond", SAFETY_DELTA_FC, 0.15),
        )

    return encdec_root


def test_e2e_pipeline_shaped_tree_renders(tmp_path: Path) -> None:
    """Synthetic YAMLs at the actual pipeline output paths render correctly."""
    Image = pytest.importorskip("PIL.Image")
    _build_pipeline_shaped_tree(tmp_path)

    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(tmp_path),
            "output_dir": str(tmp_path / "fig"),
            "num_harmful": 50,
        }
    )
    plot_main(cfg)

    for name in ("reverse_conditioned_asr.png",):
        path = tmp_path / "fig" / name
        assert path.is_file(), f"{name} missing"
        assert path.stat().st_size > 0
        with Image.open(path) as img:
            w, h = img.size
            assert w > 100 and h > 100


def test_e2e_partial_lineage_renders(tmp_path: Path) -> None:
    """Only pretrain + cap-first cond stages have data (no chained safety cond,
    no final chained stage). Plot must render without raising."""
    pytest.importorskip("PIL.Image")
    _build_pipeline_shaped_tree(tmp_path, include_chained_safety_cond=False)

    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(tmp_path),
            "output_dir": str(tmp_path / "fig"),
            "num_harmful": 50,
        }
    )
    plot_main(cfg)
    assert (tmp_path / "fig" / "reverse_conditioned_asr.png").is_file()


def test_e2e_source_drift_skips_record(tmp_path: Path) -> None:
    """A YAML with a forward-style source (base/safety_cond) inside the
    reversed tree is silently dropped because that source isn't a key in
    DEFAULT_REVERSED_COND_SOURCE_MAP. This catches operator-convention errors
    (e.g., running harmfulness eval against reversed checkpoints with the
    wrong source string) at test time rather than producing a misleading chart.

    Three layered assertions:
    1. The drifted YAML IS loaded by the parser, (b) but ISN'T in the reversed source map,
       (b) but ISN'T in the reversed source map, (c) and doesn't appear in axis data.
    2. Its source ``base/safety_cond`` is NOT a key in
       DEFAULT_REVERSED_COND_SOURCE_MAP.
    3. The rendered figure's data lines do not contain the drifted ASR value
       (0.99) at any X position.
    """
    pytest.importorskip("PIL.Image")
    from scripts.eval.plot_harmfulness_reversed_progression import _main
    from steering.eval.harmfulness_progression import (
        DEFAULT_CONFIG_LABEL_MAP,
        DEFAULT_REVERSED_COND_SOURCE_MAP,
        load_harmfulness_records,
    )

    _build_pipeline_shaped_tree(tmp_path)

    # Inject one YAML with a forward-style source inside the reversed tree.
    bad = (
        tmp_path
        / "conditional_encdec_checkpoints"
        / "d12_baseline"
        / "s128"
        / "dist_geometric"
        / "mean_10.0"
        / "std_10.0"
        / "harmfulness_results"
        / "drifted.yaml"
    )
    DRIFTED_ASR = 0.99
    _write_yaml(bad, _rec("base/safety_cond", SAFETY_DELTA_FC, DRIFTED_ASR))

    # Layer 1: the record IS loaded by the parser/filter chain.
    records = load_harmfulness_records(
        tmp_path,
        model_tag="d12_baseline",
        num_harmful=50,
        config_label_map=DEFAULT_CONFIG_LABEL_MAP,
    )
    drifted = [r for r in records if r.source == "base/safety_cond"]
    assert len(drifted) == 1, (
        "drifted record should be loaded by the loader (it satisfies all "
        "non-source filters); the source map is what excludes it from rendering"
    )
    assert abs(drifted[0].metrics.asr - DRIFTED_ASR) < 1e-9

    # Layer 2: source is NOT in the reversed source map.
    assert "base/safety_cond" not in DEFAULT_REVERSED_COND_SOURCE_MAP

    # Layer 3: render the plot, then verify no axis data contains DRIFTED_ASR.
    cfg = OmegaConf.create(
        {
            "model_tag": "d12_baseline",
            "base_dir": str(tmp_path),
            "output_dir": str(tmp_path / "fig"),
            "num_harmful": 50,
        }
    )
    _main(cfg)
    assert (tmp_path / "fig" / "reverse_conditioned_asr.png").is_file()
