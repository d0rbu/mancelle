"""Independent regression tests for adversarial GCG orchestration contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.eval.axbench_adv.merge_slices import (
    _gcg_training_source_compatibility,
    _validate_search_receipts,
)
from scripts.eval.axbench_adv.pressures import Pressure, parse_pressures
from steering.eval.axbench_adv import pipeline as pipeline_module
from steering.eval.axbench_adv.best_point import BestPoint
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, GcgObjective
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.types import ROOT_METHODS
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from tests.axbench_adv.unit.test_gcg_selected_factor import _calibration


class _ReachedProjectRuntime(Exception):
    """Stop the orchestration after all direction-allocation decisions."""


_ALL_PRESSURE_SUBSETS = tuple(
    subset for size in range(1, len(Pressure) + 1) for subset in combinations(tuple(Pressure), size)
)


@pytest.mark.parametrize("subset", _ALL_PRESSURE_SUBSETS)
def test_every_nonempty_pressure_subset_is_canonical_and_environment_equivalent(
    subset: tuple[Pressure, ...],
) -> None:
    """All 15 opt-in combinations resolve to enum order through both ingress paths."""

    assert parse_pressures(tuple(reversed(subset)), environ={}) == subset
    assert parse_pressures(environ={"AXBENCH_ADV_PRESSURES": ",".join(reversed(subset))}) == subset


@pytest.mark.parametrize("pressures", [(Pressure.GCG,), (Pressure.SOFT_PROMPT,), (Pressure.GCG, Pressure.SOFT_PROMPT)])
def test_zero_only_activation_l2_allocates_no_project_direction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    pressures: tuple[Pressure, ...],
) -> None:
    """A k=0-only control must not require or load an activation calibration."""

    gcg = AxBenchAdvGcgConfig(
        objective=GcgObjective.ACTIVATION_L2,
        search=replace(FasterGcgConfig(), optimized_token_counts=(0,)),
    )
    config = SimpleNamespace(
        pressures=pressures,
        gcg=gcg,
        soft_prompt=AxBenchAdvSoftPromptConfig(objective=GcgObjective.ACTIVATION_L2, optimized_token_counts=(0,)),
        factor_overrides={},
        project_counter_direction_root=None,
        project_counter_layer=None,
        project_counter_fit_seed=42,
        counter_resume_snapshot=None,
        counter_resume_sha256=None,
        judge_cache=tmp_path / "judge.sqlite",
        reflection_cache=tmp_path / "reflection.sqlite",
        master_seed=42,
        device="cpu",
        source_paths=SimpleNamespace(prepared_dir=tmp_path),
        source_expectations=SimpleNamespace(max_concepts=1, output_length=1, layer=0),
        worker_timeout_seconds=1.0,
        project_judge_inflight=1,
        project_judge_workers=1,
        project_judge_pair_concurrency=1,
        project_reuse_judge_workers=False,
        project_reuse_single_token_mask=False,
        project_bf16_decoder_linears=False,
        project_reuse_diffmean_texts=False,
        project_weight_free_extraction=False,
        project_native_extraction=False,
        project_fit_selected_concepts_only=False,
        cache_only=True,
        api_base_url=None,
    )
    validated = SimpleNamespace(
        output_dir=tmp_path / "out",
        project_methods=(ROOT_METHODS[0],),
        method_setup=object(),
        training_texts={},
        main_frame=object(),
        extraction_batch_size=1,
        nl_target_scores=None,
    )

    monkeypatch.setattr(
        pipeline_module,
        "_best_points",
        lambda *_args, **_kwargs: {ROOT_METHODS[0]: BestPoint(ROOT_METHODS[0], 1.0, 1.0, 1.0, 1.0)},
    )
    monkeypatch.setattr(pipeline_module, "write_best_points", lambda *_args: None)
    monkeypatch.setattr(pipeline_module, "write_source_contract", lambda *_args: None)
    monkeypatch.setattr(pipeline_module, "select_training_pairs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        pipeline_module,
        "_gcg_training_source_identity",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        pipeline_module,
        "preflight_project_direction_fit",
        lambda *_args, **_kwargs: pytest.fail("zero-only GCG preflighted a project direction"),
    )
    monkeypatch.setattr(
        pipeline_module,
        "load_project_gcg_activation_direction",
        lambda *_args, **_kwargs: pytest.fail("zero-only GCG loaded a project direction"),
    )

    def stop_at_runtime(**kwargs: object) -> object:
        assert "gcg_activation_directions" not in kwargs
        raise _ReachedProjectRuntime

    with pytest.raises(_ReachedProjectRuntime):
        pipeline_module._execute(
            config=config,  # ty: ignore[invalid-argument-type]
            validated=validated,  # ty: ignore[invalid-argument-type]
            reflection_lm=None,
            worker_factory=object(),  # ty: ignore[invalid-argument-type]
            judge_factory=object(),  # ty: ignore[invalid-argument-type]
            self_conditioning_factory=stop_at_runtime,  # ty: ignore[invalid-argument-type]
            optimizer=object(),  # ty: ignore[invalid-argument-type]
            final_judge=object(),  # ty: ignore[invalid-argument-type]
            methods=(ROOT_METHODS[0],),
            concept_ids=(0,),
        )


def test_gcg_merge_source_compatibility_is_content_based_and_portable() -> None:
    """Absolute staging paths are provenance, while content identities gate merges."""

    first = {
        "train_data": {"path": "/host-a/train.parquet", "sha256": "a" * 64, "byte_length": 1},
        "metadata": {"path": "/host-a/metadata.jsonl", "sha256": "b" * 64, "byte_length": 2},
        "pair_sha256": {"0": "c" * 64},
    }
    relocated = {
        **first,
        "train_data": {**first["train_data"], "path": "/host-b/train.parquet"},
        "metadata": {**first["metadata"], "path": "/host-b/metadata.jsonl"},
    }
    mutated = {
        **relocated,
        "train_data": {**relocated["train_data"], "sha256": "d" * 64},
    }

    assert _gcg_training_source_compatibility(first) == _gcg_training_source_compatibility(
        relocated
    )
    assert _gcg_training_source_compatibility(first) != _gcg_training_source_compatibility(mutated)


@pytest.mark.parametrize(
    ("token_count", "field_path", "value"),
    [
        (1, ("direction_calibration", "factor"), 2.0),
        (1, ("direction_calibration", "resolved_scale"), 6.0),
        (1, ("direction_calibration", "resolved_scale"), True),
        (1, ("direction_calibration", "factor"), -1.0),
        (1, ("direction_calibration", "resolved_scale"), -2.0),
        (1, ("direction_calibration", "max_activation"), 0.0),
        (1, ("direction_calibration", "direction_norm"), 0.0),
        (
            1,
            ("direction_calibration", "coefficient_semantics"),
            "saved_diffmean * own_positive_max_activation * 1.0",
        ),
        (0, ("effective_config", "direction_factor_policy"), None),
        (1, ("effective_config", "direction_factor_policy"), "fixed_one"),
        (0, ("effective_config", "alpha"), 0.0),
        (0, ("effective_config", "unexpected"), True),
        (0, ("effective_config", "search"), None),
        (0, ("effective_config", "search"), []),
        (1, ("effective_config", "search"), {"optimized_token_counts": [0, 1]}),
        (1, ("effective_config", "search", "master_seed"), 43),
        (1, ("schema_version",), True),
        (1, ("concept_id",), False),
        (1, ("token_count",), True),
        (1, ("factor",), True),
        (1, ("model_tokenizer_identity", "concept_id"), False),
        (1, ("model_tokenizer_identity", "factor"), True),
        (1, ("direction_calibration", "concept_id"), False),
        (1, ("direction_calibration", "factor"), True),
        (1, ("suffix_token_ids",), [32]),
        (1, ("suffix",), ""),
        (1, ("suffix",), "  padded  "),
        (1, ("diagnostics", "accepted_steps"), FasterGcgConfig().max_steps + 1),
        (0, ("diagnostics", "accepted_steps"), 1),
        (0, ("diagnostics", "visited_sequences"), 1),
        (0, ("diagnostics", "accepted_steps"), False),
        (0, ("diagnostics", "extra"), 0),
    ],
)
def test_gcg_receipts_reject_rehashed_contract_violations(
    tmp_path: Path,
    token_count: int,
    field_path: tuple[str, ...],
    value: object,
) -> None:
    """A fresh hash must not authenticate producer-impossible receipt contents."""

    identity = "a" * 64
    effective = json.loads(
        json.dumps(
            AxBenchAdvGcgConfig(
                objective=GcgObjective.ACTIVATION_L2,
                search=replace(FasterGcgConfig(), optimized_token_counts=(0, 1)),
            ).effective_identity()
        )
    )
    training_source = {
        "train_data": {
            "path": "/source/train_data.parquet",
            "sha256": "b" * 64,
            "byte_length": 10,
        },
        "metadata": {
            "path": "/source/metadata.jsonl",
            "sha256": "c" * 64,
            "byte_length": 10,
        },
        "pair_sha256": {"0": identity},
    }
    common = {
        "schema_version": 1,
        "objective": "activation_l2",
        "effective_config": effective,
        "training_pair_identity": identity,
        "method": "SelfConditioning",
        "concept_id": 0,
        "concept": "topic",
        "factor": 1.0,
    }
    records: dict[int, dict[str, object]] = {
        0: {
            **common,
            "token_count": 0,
            "suffix_token_ids": [],
            "suffix": "",
            "diagnostics": {"control": True, "accepted_steps": 0, "visited_sequences": 0},
        },
        1: {
            **common,
            "token_count": 1,
            "suffix_token_ids": [3],
            "suffix": "x",
            "model_tokenizer_identity": {
                "runtime": "project_encoder_decoder",
                "rank_sentinel_id": 0,
                "vocab_size": 32,
                "embedding_dim": 8,
                "method": "SelfConditioning",
                "concept_id": 0,
                "factor": 1.0,
                "frozen_state": "clean_real_prompt_S0_before_assistant_marker",
            },
            "layouts_sha256": "d" * 64,
            "direction_calibration": {
                **_calibration(project=True, factor=1.0),
                "sha256": "e" * 64,
                "fit_manifest_sha256": "f" * 64,
                "direction_norm": 1.0,
            },
            "objective_value": 1.0,
            "diagnostics": {"control": False, "accepted_steps": 1, "visited_sequences": 2},
        },
    }

    def publish() -> dict[str, str]:
        files: dict[str, str] = {}
        for token_count, record in records.items():
            relative = Path("gcg") / "SelfConditioning" / "c0" / f"k{token_count}" / "search.json"
            path = tmp_path / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record, allow_nan=False), encoding="utf-8")
            files[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        return files

    manifest = {
        "max_concepts": 1,
        "master_seed": 0,
        "gcg": effective,
        "gcg_training_source": training_source,
    }
    files = publish()
    _validate_search_receipts(
        pressure=Pressure.GCG,
        root=tmp_path,
        manifest=manifest,
        files=files,
        methods=("SelfConditioning",),
        concept_ids=(0,),
        factors={"SelfConditioning": 1.0},
        token_counts=(0, 1),
    )

    target = records[token_count]
    for field in field_path[:-1]:
        nested = target[field]
        assert isinstance(nested, dict)
        target = nested
    target[field_path[-1]] = value
    files = publish()
    with pytest.raises(AxBenchAdvError):
        _validate_search_receipts(
            pressure=Pressure.GCG,
            root=tmp_path,
            manifest=manifest,
            files=files,
            methods=("SelfConditioning",),
            concept_ids=(0,),
            factors={"SelfConditioning": 1.0},
            token_counts=(0, 1),
        )
