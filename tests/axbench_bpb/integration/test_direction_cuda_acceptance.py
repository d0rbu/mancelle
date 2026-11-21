"""CUDA acceptance for validated direction installation and legacy parity."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch as t
import torch.nn as nn
from safetensors.torch import save_file

from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_bpb.intervention import (
    _build_direction_intervention,
    apply_intervention,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.types import Intervention


class _IdentityBlock(nn.Module):
    def forward(self, value: t.Tensor) -> t.Tensor:
        return value


class _TwoBlockModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(t.zeros(()))
        self.transformer = nn.ModuleDict(
            {"h": nn.ModuleList([_IdentityBlock(), _IdentityBlock()])}
        )
        self.layer_zero_output: t.Tensor | None = None

    def forward(self, value: t.Tensor) -> t.Tensor:
        blocks = cast(nn.ModuleList, self.transformer["h"])
        value = blocks[0](value)
        self.layer_zero_output = value.detach().clone()
        return blocks[1](value)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_validated_cuda_direction_uses_trained_layer_with_exact_legacy_output(
    tmp_path: Path,
) -> None:
    source_paths = SourcePaths(source_run_dir=tmp_path / "source")
    artifact_path = source_paths.diffmean_direction(
        "decoder_only",
        "positive",
        3,
    )
    artifact_path.parent.mkdir(parents=True)
    direction = t.tensor([0.5, -1.0, 2.0, -3.0], dtype=t.float32)
    save_file(
        {"direction": direction},
        str(artifact_path),
        metadata={
            "concept_id": "3",
            "layer": "1",
            "mode": "positive",
            "target_model": "decoder_only",
        },
    )

    validated = _build_direction_intervention(
        method=get_method_by_name("diffmean_decoder_only"),
        concept=type("Concept", (), {"concept_id": 3})(),
        coefficient=1.75,
        source_paths=source_paths,
        diffmean_layer=1,
        apply_max_act_scale=False,
    )
    legacy = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=1,
        coefficient=1.75,
    )

    device = t.device("cuda")
    model = _TwoBlockModel().to(device)
    inputs = t.randn(2, 3, 4, device=device)
    baseline = model(inputs)
    baseline_layer_zero = model.layer_zero_output
    assert baseline_layer_zero is not None

    with apply_intervention(model, validated):
        validated_output = model(inputs)
        validated_layer_zero = model.layer_zero_output
    with apply_intervention(model, legacy):
        legacy_output = model(inputs)

    assert validated_layer_zero is not None
    assert t.equal(validated_layer_zero, baseline_layer_zero)
    assert t.equal(validated_output, legacy_output)
    expected = baseline + (1.75 * direction.to(device))[None, None, :]
    assert t.equal(validated_output, expected)
