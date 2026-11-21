"""Selected-factor receipts, calibration ownership, and closed config boundaries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from scripts.eval.axbench_adv.merge_slices import (
    _gcg_common_calibration,
    _validate_gcg_calibration,
    _validate_search_receipts,
)
from scripts.eval.axbench_adv.pressures import Pressure
from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_adv.counter import (
    load_project_gcg_activation_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective, parse_gcg_config
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS, ALL_METHOD_FACTORS
from steering.eval.axbench_adv.self_conditioning import _BoundSelfConditioningCell
from steering.eval.axbench_adv.types import ROOT_METHODS
from tests.axbench_adv.unit.test_direction_fitting import _publication
from tests.axbench_adv.unit.test_gcg_token_pool import _TextPolicyTokenizer


def _calibration(*, project: bool, factor: float, concept_id: int = 0) -> dict[str, Any]:
    maximum = 2.0 + concept_id * 1.3
    common = {
        "concept_id": concept_id,
        "layer": 0,
        "max_activation": maximum,
        "direction_norm": 1.25,
        "factor": factor,
        "resolved_scale": maximum * factor,
    }
    if project:
        return {
            **common,
            "objective": "activation_l2",
            "family": "diffmean",
            "path": f"/source/diffmean/c{concept_id}.safetensors",
            "sha256": str(concept_id + 1) * 64,
            "fit_manifest_sha256": "b" * 64,
            "site": "post_cross_attention",
            "mode": "positive",
            "target_model": "encdec",
            "coefficient_semantics": (
                "saved_diffmean * own_positive_max_activation * selected_method_factor"
            ),
        }
    return {
        **common,
        "site": "model.layers[0].output_after_actual_intervention",
        "direction_path": "/source/train/DiffMean_weight.pt",
        "direction_sha256": "a" * 64,
        "latent_path": "/source/inference/latent_data.parquet",
        "latent_sha256": "b" * 64,
        "raw_max_activation": maximum,
        "fallback_used": False,
        "offset_dtype_semantics": (
            "float32 [1] max_act * float32 [1] factor * saved bfloat16 direction -> float32"
        ),
    }


def _validate(calibration: Mapping[str, Any], *, factor: object, concept_id: int = 0) -> None:
    _validate_gcg_calibration(
        calibration,
        objective="activation_l2",
        concept_id=concept_id,
        model_identity={"factor": factor},
        relative=Path("search.json"),
    )


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("factor", ADV_FULL_FACTORS)
@pytest.mark.parametrize("concept_id", [0, 1])
def test_calibration_accepts_selected_factors_including_zero(
    project: bool, factor: float, concept_id: int
) -> None:
    calibration = _calibration(project=project, factor=factor, concept_id=concept_id)
    _validate(calibration, factor=factor, concept_id=concept_id)
    assert _gcg_common_calibration(calibration) == _gcg_common_calibration(
        _calibration(project=project, factor=6.0, concept_id=1 - concept_id)
    )


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize("field", ["factor", "resolved_scale", "max_activation", "direction_norm"])
@pytest.mark.parametrize(
    "invalid", [True, False, -0.1, float("nan"), float("inf"), -float("inf"), "1", None]
)
def test_calibration_rejects_invalid_numbers_even_with_matching_model_factor(
    project: bool, field: str, invalid: object
) -> None:
    calibration = _calibration(project=project, factor=3.0)
    calibration[field] = invalid
    with pytest.raises(AxBenchAdvError, match="calibration"):
        _validate(calibration, factor=calibration["factor"])


@pytest.mark.parametrize("project", [False, True])
@pytest.mark.parametrize(
    ("recorded_factor", "recorded_scale"),
    [(1.0, 3.3), (0.4, 3.3 * 0.4), (0.6, 2.0), (0.6, float(t.tensor(3.3 * 0.6)))],
    ids=["stale_one", "wrong_factor", "wrong_product", "rounded_product"],
)
def test_calibration_binds_exact_factor_and_python_float_product(
    project: bool, recorded_factor: float, recorded_scale: float
) -> None:
    calibration = _calibration(project=project, factor=0.6, concept_id=1)
    calibration.update(factor=recorded_factor, resolved_scale=recorded_scale)
    with pytest.raises(AxBenchAdvError, match="calibration"):
        _validate(calibration, factor=0.6, concept_id=1)


def test_direction_policy_is_derived_and_not_a_user_setting() -> None:
    config = AxBenchAdvGcgConfig(objective=GcgObjective.ACTIVATION_L2)
    assert "direction_factor_policy" not in asdict(config)
    with pytest.raises(AxBenchAdvError, match="exactly"):
        parse_gcg_config({**asdict(config), "direction_factor_policy": "selected_method_factor"})


@pytest.mark.parametrize("location", ["manifest", "receipt"])
@pytest.mark.parametrize("policy", ["missing", None, "fixed_one", False, True])
def test_even_zero_budget_and_factor_one_require_new_l2_policy(
    tmp_path: Path, location: str, policy: object
) -> None:
    config = AxBenchAdvGcgConfig(objective=GcgObjective.ACTIVATION_L2)
    config = replace(config, search=replace(config.search, optimized_token_counts=(0,)))
    effective = json.loads(json.dumps(config.effective_identity()))
    receipt_effective = dict(effective)
    identity = "a" * 64
    manifest = {
        "gcg": effective,
        "max_concepts": 1,
        "master_seed": 0,
        "gcg_training_source": {
            "pair_sha256": {"0": identity},
            **{
                key: {"path": f"/source/{key}", "sha256": "b" * 64, "byte_length": 1}
                for key in ("train_data", "metadata")
            },
        },
    }
    receipt = {
        "schema_version": 1,
        "objective": "activation_l2",
        "effective_config": receipt_effective,
        "training_pair_identity": identity,
        "method": "DiffMean",
        "concept_id": 0,
        "concept": "topic",
        "factor": 1.0,
        "token_count": 0,
        "suffix_token_ids": [],
        "suffix": "",
        "diagnostics": {"control": True, "accepted_steps": 0, "visited_sequences": 0},
    }
    relative = Path("gcg/DiffMean/c0/k0/search.json")
    path = tmp_path / relative
    path.parent.mkdir(parents=True)

    def validate() -> None:
        path.write_text(json.dumps(receipt))
        _validate_search_receipts(
            pressure=Pressure.GCG,
            root=tmp_path,
            manifest=manifest,
            files={str(relative): hashlib.sha256(path.read_bytes()).hexdigest()},
            methods=("DiffMean",),
            concept_ids=(0,),
            factors={"DiffMean": 1.0},
            token_counts=(0,),
        )

    validate()
    target = effective if location == "manifest" else receipt_effective
    if policy == "missing":
        target.pop("direction_factor_policy")
    else:
        target["direction_factor_policy"] = policy
    with pytest.raises(AxBenchAdvError, match="fresh L2 run"):
        validate()


def test_project_preparation_owns_factor_receipts_without_mutating_shared_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, expectations = _publication(tmp_path)
    fit = preflight_project_direction_fit(root, expectations)
    loaded = load_project_gcg_activation_direction(fit, concept_id=0)
    original = dict(loaded.provenance)
    assert not {"factor", "resolved_scale", "coefficient_semantics"} & original.keys()
    directions = {
        0: loaded,
        1: replace(
            loaded,
            max_activation=3.3,
            provenance={**original, "concept_id": 1, "max_activation": 3.3},
        ),
    }
    runtime = SimpleNamespace(
        _target_model=object(),
        _target_tokenizer=_TextPolicyTokenizer(("!", "x")),
        _gcg_activation_directions=directions,
    )
    monkeypatch.setattr(
        self_conditioning, "_project_embedding_weight", lambda _model: t.zeros(2, 3)
    )
    row = SimpleNamespace(
        prompt_prefix=(1,), clean_prompt=(1,), response=(2,), source_slot=0, target_slot=1
    )
    monkeypatch.setattr(
        _BoundSelfConditioningCell,
        "_prepare_gcg_row",
        lambda _cell, **kw: (row, (0,) * len(kw["seed_text"].split())),
    )
    sample = {"sample_id": 1, "prompt": "prompt", "response": "response"}
    receipts = []
    for method in ROOT_METHODS:
        for factor in ALL_METHOD_FACTORS[method]:
            for concept_id, direction in directions.items():
                cell = object.__new__(_BoundSelfConditioningCell)
                cell._runtime = cast(Any, runtime)
                cell._method, cell._concept_id, cell._factor = method, concept_id, factor
                for objective, k, alpha in (
                    ("activation_l2", 1, 0.0),
                    ("activation_l2", 3, 937.0),
                    ("response_margin", 1, 0.01),
                    ("activation_l2", 1, 0.01),
                ):
                    prepared = cell.prepare_gcg(
                        objective=objective,
                        alpha=alpha,
                        rows=[{"pair_id": 0, "positive": sample, "negative": sample}],
                        suffix_token_count=k,
                        device_batch_size=1,
                        separator="\n\n",
                    )
                    calibration = prepared.direction_calibration
                    if objective == "response_margin":
                        assert calibration is cell._gcg_direction is None
                        continue
                    assert calibration is not None
                    assert calibration is not direction.provenance
                    assert calibration["factor"] == prepared.identity["factor"] == factor
                    assert calibration["resolved_scale"] == direction.max_activation * factor
                    _validate(calibration, factor=factor, concept_id=concept_id)
                    receipts.append((calibration, dict(calibration)))
    assert all(receipt == snapshot for receipt, snapshot in receipts)
    assert loaded.provenance == original
    assert dict(load_project_gcg_activation_direction(fit, concept_id=0).provenance) == original
