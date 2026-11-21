"""general_val combo intervention max_act scaling (issue #588, Task 8)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv.general_val import (
    _build_combo_intervention,
    _InterventionCombo,
)

_MINI = (
    Path(__file__).resolve().parents[2]
    / "axbench_bpb"
    / "fixtures"
    / "mini_axbench_base_run"
)
_FID = 7
_MAX_ACT = 5.0


def _concept() -> Concept:
    return Concept(
        concept_id=1,
        feature_index=_FID,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="concept",
        corpus_count=10,
        description_source="test",
    )


def _combo() -> _InterventionCombo:
    return _InterventionCombo(
        target_model="decoder_only",
        method_name="diffmean_decoder_only",
        mode="positive",
        concept_id=1,
        coefficient_str="4.0",
        adversarial_method="faster_gcg",
        adversarial_strength=0.0,
        strength_index=0,
    )


def _build(
    source_paths: SourcePaths,
    *,
    apply_max_act_scale: bool,
    diffmean_layer: int = 8,
):
    return _build_combo_intervention(
        combo=_combo(),
        method=get_method_by_name("diffmean_decoder_only"),
        concept=_concept(),
        source_paths=source_paths,
        diffmean_layer=diffmean_layer,
        target_model_meta=SimpleNamespace(),
        inactive_sampler=None,
        tokenizer=SimpleNamespace(),
        counter_direction_family="diffmean",
        apply_max_act_scale=apply_max_act_scale,
    )


def test_general_val_combo_scales_baseline(tmp_path):
    source_paths = SourcePaths(tmp_path)
    path = source_paths.diffmean_direction("decoder_only", "positive", 1)
    path.parent.mkdir(parents=True)
    save_file(
        {"direction": t.tensor([1.0, 0.0])},
        str(path),
        metadata={
            "concept_id": "1",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": "8",
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(_MAX_ACT),
        },
    )
    assert _build(source_paths, apply_max_act_scale=True).direction_coefficient == (
        4.0 * _MAX_ACT
    )


def test_general_val_combo_none_provider_is_identity():
    assert _build(
        SourcePaths(source_run_dir=_MINI),
        apply_max_act_scale=False,
        diffmean_layer=1,
    ).direction_coefficient == 4.0
