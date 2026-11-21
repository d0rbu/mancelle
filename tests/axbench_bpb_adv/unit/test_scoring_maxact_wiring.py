"""Direction-artifact preflight wiring for BPB-Adv (#1228)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
)
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv.scoring import (
    ModelBackedDualCompletionScorer,
    _preflight_direction_scales_for_rows,
)


def _concept() -> Concept:
    return Concept(1, 7, 8, "sae", "concept", 10, "test")


def _write(paths: SourcePaths, family: str, value: float) -> None:
    path = (
        paths.diffmean_direction("decoder_only", "positive", 1)
        if family == "diffmean"
        else paths.linear_probe_direction("decoder_only", "positive", 1)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.tensor([1.0, 0.0])},
        str(path),
        metadata={
            "concept_id": "1",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": "8",
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(value),
        },
    )


def test_preflight_uses_base_and_counter_artifact_values(tmp_path: Path) -> None:
    paths = SourcePaths(tmp_path)
    _write(paths, "linear_probe", 3.0)
    _write(paths, "diffmean", 7.0)
    result = _preflight_direction_scales_for_rows(
        candidate_rows=[
            {
                "method_name": "linear_probe_decoder_only",
                "concept_id": 1,
                "target_model": "decoder_only",
                "adversarial_method": "additive_direction",
                "adversarial_strength": 0.4,
            }
        ],
        concepts_by_id={1: _concept()},
        source_paths=paths,
        expected_layer=8,
        counter_direction_family="diffmean",
    )
    assert result.values == {
        ("linear_probe", "decoder_only", "positive", 1): 3.0,
        ("diffmean", "decoder_only", "positive", 1): 7.0,
    }


def test_preflight_skips_rows_without_direction_consumers(tmp_path: Path) -> None:
    result = _preflight_direction_scales_for_rows(
        candidate_rows=[
            {
                "method_name": "control_decoder_only",
                "concept_id": 1,
                "target_model": "decoder_only",
                "adversarial_method": "additive_direction",
                "adversarial_strength": 0.0,
            }
        ],
        concepts_by_id={1: _concept()},
        source_paths=SourcePaths(tmp_path),
        expected_layer=8,
        counter_direction_family="diffmean",
    )
    assert result.values == {}


def test_model_backed_scorer_applies_artifact_scale_directly(
    tmp_path: Path,
) -> None:
    artifact_value = 13.0
    paths = SourcePaths(tmp_path)
    _write(paths, "diffmean", artifact_value)
    scorer = ModelBackedDualCompletionScorer(
        target_bundles={
            "decoder_only": cast(
                Any,
                SimpleNamespace(model=SimpleNamespace(feature_vocab=None)),
            )
        },
        target_model_metas={"decoder_only": SimpleNamespace()},
        concepts_by_id={1: _concept()},
        source_paths=paths,
        tokenizer=SimpleNamespace(),
        device=t.device("cpu"),
        diffmean_layer=8,
        inactive_sampler=None,
        apply_max_act_scale=True,
    )
    intervention = scorer._base_steered_intervention(
        method=get_method_by_name("diffmean_decoder_only"),
        concept=_concept(),
        coefficient=2.0,
        candidate_row={
            "method_name": "diffmean_decoder_only",
            "concept_id": 1,
            "core_prefix_id": "prefix",
            "coefficient_str": "2.0",
            "target_model": "decoder_only",
        },
    )
    assert intervention.direction_coefficient == 2.0 * artifact_value


def test_model_backed_scorer_rejects_positional_scale_mode() -> None:
    with pytest.raises(TypeError, match="positional argument"):
        cast(Any, ModelBackedDualCompletionScorer)(
            {},
            {},
            {},
            SimpleNamespace(),
            SimpleNamespace(),
            t.device("cpu"),
            8,
            None,
            True,
        )
