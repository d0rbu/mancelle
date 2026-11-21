"""Unit tests for the committed faster-GCG timing-probe config.

``configs/eval/axbench_bpb_adv/axbench_bpb_adv_timing_probe.yaml`` is the cheap,
measured spot-check config (Gap E / spec 2E). These tests load it through the
real hydra + ``cfg_to_dataclass`` path so that the committed knobs cannot drift
from the documented cost model.
"""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from steering.eval.axbench_bpb_adv.config import (
    AdversarialMethod,
    FasterGcgConfig,
    get_faster_gcg_config,
    validate_config,
)
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass

_CONFIG_DIR = (
    Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_bpb_adv"
)


def _load_timing_probe():
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="axbench_bpb_adv_timing_probe",
            overrides=[
                "source_run_dir=/tmp/source",
                "feature_dir=/tmp/features",
                "feature_statistics_path=/tmp/stats.safetensors",
            ],
        )
    return cfg_to_dataclass(cfg)


def test_timing_probe_config_loads_and_validates() -> None:
    dataclass_cfg = _load_timing_probe()
    dataclass_cfg.run_tag = "probe"
    validate_config(dataclass_cfg)


def test_timing_probe_uses_cheap_faster_gcg_knobs() -> None:
    dataclass_cfg = _load_timing_probe()
    faster_gcg = get_faster_gcg_config(dataclass_cfg)
    assert isinstance(faster_gcg, FasterGcgConfig)
    assert faster_gcg.optimized_token_counts == (0, 1, 2)
    assert faster_gcg.max_steps == 2
    assert faster_gcg.candidate_batch_size == 4
    assert faster_gcg.top_k == 8


def test_timing_probe_is_a_subsampled_single_method_decoder_only_probe() -> None:
    dataclass_cfg = _load_timing_probe()
    # Single optimized base method. The encdec target is present only so the
    # representative fair Feature-DiffMean target row can be scored at k=0.
    assert dataclass_cfg.base_methods == ["diffmean_decoder_only"]
    assert dataclass_cfg.representative_target_methods == ["feature_diffmean_encdec"]
    assert dataclass_cfg.target_models == ["decoder_only", "encdec"]
    # faster_gcg only -- no additive_direction control sweep in the probe.
    methods = {AdversarialMethod.coerce(m.method) for m in dataclass_cfg.adversarial_methods}
    assert methods == {AdversarialMethod.FASTER_GCG}
