"""Stage-4 direction-scale request derivation and fail-fast preflight."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base import generation as generation_module
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.direction_scale import (
    MAX_ACTIVATION_METADATA_KEY,
    encode_max_activation,
    preflight_direction_scales,
)
from steering.eval.axbench_base.generation import (
    _derive_stage4_direction_scale_requests,
    generate_all,
)
from steering.eval.axbench_base.method_registry import ALL_METHODS, get_method_by_name
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SteeringMode,
    TargetBundle,
)
from tests.axbench_base.unit.test_config import make_valid_config


def _concept(concept_id: int) -> Concept:
    return Concept(concept_id, concept_id + 10, 8, "sae", "concept", 10, "test")


def _paths(tmp_path: Path) -> Any:
    def _direction(family: str, target: str, mode: Any, concept_id: int) -> Path:
        mode_value = mode.value if isinstance(mode, SteeringMode) else str(mode)
        return tmp_path / family / target / mode_value / f"concept_{concept_id}.safetensors"

    return SimpleNamespace(
        diffmean_direction=lambda target, mode, concept_id: _direction(
            "diffmean", target, mode, concept_id
        ),
        linear_probe_direction=lambda target, mode, concept_id: _direction(
            "linear_probe", target, mode, concept_id
        ),
    )


def _plan(method_name: str) -> Any:
    method = get_method_by_name(method_name)
    return SimpleNamespace(method=method, target_name=method.target_model)


def _write_request_artifact(
    request: Any,
    *,
    value: float,
    direction: t.Tensor | None = None,
) -> None:
    assert request.path is not None
    request.path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            "direction": (
                t.tensor([1.0, 0.0]) if direction is None else direction
            )
        },
        str(request.path),
        metadata={
            "concept_id": str(request.concept_id),
            "mode": request.mode,
            "target_model": request.target_model,
            "layer": "3",
            MAX_ACTIVATION_METADATA_KEY: encode_max_activation(value),
        },
    )


def test_generate_all_requires_apply_max_act_scale_keyword() -> None:
    with pytest.raises(TypeError, match="apply_max_act_scale"):
        cast(Any, generate_all)(
            object(),
            concepts=[],
            targets=[],
            eval_seeds_positive=[],
            eval_seeds_negative=[],
        )


def test_bad_scalar_concept_raises_during_stage4_preflight(tmp_path: Path) -> None:
    output_paths = _paths(tmp_path)
    requests = _derive_stage4_direction_scale_requests(
        [_plan("diffmean_decoder_only")],
        eligible_by_mode={
            SteeringMode.POSITIVE: [_concept(1)],
            SteeringMode.NEGATIVE: [],
        },
        output_paths=output_paths,
    )
    _write_request_artifact(requests[0], value=0.0)
    with pytest.raises(
        AxBenchBaseError,
        match="zero-direction max_activation mismatch",
    ):
        preflight_direction_scales(
            requests,
            expected_layer=3,
            stage_label="Stage 4",
        )


def test_ours_only_stage4_plan_skips_direction_validation(tmp_path: Path) -> None:
    requests = _derive_stage4_direction_scale_requests(
        [_plan("ours_extract_and_add_encdec")],
        eligible_by_mode={
            SteeringMode.POSITIVE: [_concept(1)],
            SteeringMode.NEGATIVE: [],
        },
        output_paths=_paths(tmp_path),
    )
    assert requests == []
    assert preflight_direction_scales(
        requests,
        expected_layer=3,
        stage_label="Stage 4",
    ).values == {}


def test_valid_negative_scalar_plan_passes_stage4_preflight(tmp_path: Path) -> None:
    requests = _derive_stage4_direction_scale_requests(
        [_plan("neg_linear_probe_decoder_only")],
        eligible_by_mode={
            SteeringMode.POSITIVE: [],
            SteeringMode.NEGATIVE: [_concept(2)],
        },
        output_paths=_paths(tmp_path),
    )
    _write_request_artifact(requests[0], value=4.0)
    result = preflight_direction_scales(
        requests,
        expected_layer=3,
        stage_label="Stage 4",
    )
    assert result.values == {
        ("linear_probe", "decoder_only", "negative", 2): 4.0
    }


def test_stage4_skips_eligible_mode_without_scalar_plan(tmp_path: Path) -> None:
    requests = _derive_stage4_direction_scale_requests(
        [_plan("diffmean_decoder_only")],
        eligible_by_mode={
            SteeringMode.POSITIVE: [_concept(1)],
            SteeringMode.NEGATIVE: [_concept(2)],
        },
        output_paths=_paths(tmp_path),
    )
    assert [(request.mode, request.concept_id) for request in requests] == [
        ("positive", 1)
    ]
    _write_request_artifact(requests[0], value=3.0)
    preflight_direction_scales(
        requests,
        expected_layer=3,
        stage_label="Stage 4",
    )


def test_stage4_direction_request_set_equals_scalar_plan_cross_product(
    tmp_path: Path,
) -> None:
    requests = _derive_stage4_direction_scale_requests(
        [
            _plan("diffmean_decoder_only"),
            _plan("neg_linear_probe_encdec"),
            _plan("ours_extract_and_add_encdec"),
        ],
        eligible_by_mode={
            SteeringMode.POSITIVE: [_concept(1), _concept(3)],
            SteeringMode.NEGATIVE: [_concept(2)],
        },
        output_paths=_paths(tmp_path),
    )
    actual = {
        (
            request.family,
            request.target_model,
            request.mode,
            request.concept_id,
        )
        for request in requests
    }
    assert actual == {
        ("diffmean", "decoder_only", "positive", 1),
        ("diffmean", "decoder_only", "positive", 3),
        ("linear_probe", "encdec", "negative", 2),
    }


def test_direction_layer_mismatch_fails_before_forward_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = make_valid_config(
        output_root=str(tmp_path / "out"),
        run_tag="layer-mismatch",
        enabled_methods=["diffmean_decoder_only"],
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    direction_path = output_paths.diffmean_direction(
        "decoder_only",
        SteeringMode.POSITIVE,
        0,
    )
    direction_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.ones(8, dtype=t.float32)},
        str(direction_path),
        metadata={
            "concept_id": "0",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": "10",
        },
    )
    method = next(
        candidate
        for candidate in ALL_METHODS
        if candidate.name == "diffmean_decoder_only"
    )
    plan = SimpleNamespace(
        method=method,
        target_name="decoder_only",
        cluster_source_corpus_sha256=None,
    )
    target = cast(
        TargetBundle,
        SimpleNamespace(name="decoder_only", cluster_contract=None),
    )
    monkeypatch.setattr(
        generation_module,
        "load_seed_pool_fingerprint",
        lambda _paths: "f" * 64,
    )
    monkeypatch.setattr(
        generation_module,
        "validate_cluster_generation_resume",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        generation_module,
        "_load_existing_generation_metadata",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        generation_module,
        "_build_stage4_method_plans",
        lambda *_args, **_kwargs: [plan],
    )
    monkeypatch.setattr(
        generation_module,
        "run_intervention",
        lambda *_args, **_kwargs: pytest.fail("forward ran before direction preflight"),
    )

    class ForbiddenWriter:
        def __init__(self, **_kwargs: Any) -> None:
            pytest.fail("output writer was created before direction preflight")

    monkeypatch.setattr(generation_module, "ParquetAppendWriter", ForbiddenWriter)

    with pytest.raises(AxBenchBaseError, match="layer mismatch") as exc_info:
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[target],
            eval_seeds_positive=[],
            eval_seeds_negative=[],
            apply_max_act_scale=False,
        )

    message = str(exc_info.value)
    assert "requested/applied layer=6" in message
    assert "artifact training layer=10" in message
    assert str(direction_path) in message
    assert not output_paths.generations_dir().exists()
