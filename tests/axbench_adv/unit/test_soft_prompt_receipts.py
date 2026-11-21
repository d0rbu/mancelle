"""Authenticate actual tensors and reject semantically rewritten soft receipts."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

from scripts.eval.axbench_adv.merge_slices import (
    _validate_gcg_model_identity,
    _validate_search_receipts,
)
from scripts.eval.axbench_adv.pressures import Pressure
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_driver import training_pair_identity
from steering.eval.axbench_adv.soft_prompt_config import AxBenchAdvSoftPromptConfig
from steering.eval.axbench_adv.soft_prompt_driver import execute_soft_prompt_cell
from tests.axbench_adv.unit.test_soft_prompt_driver import ToySoftRuntime, training_pairs


@pytest.fixture
def receipt_tree(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    config = AxBenchAdvSoftPromptConfig(optimized_token_counts=(0, 1, 3), max_steps=2)
    training = training_pairs(members=1)
    execute_soft_prompt_cell(
        config=config,
        runtime=ToySoftRuntime(members=1),
        master_seed=7,
        method="DiffMean",
        concept_id=0,
        concept="topic 0",
        factor=1.0,
        training_pairs=training,
        evaluation_prompts=("held out",),
        evaluation_input_ids=(0,),
        output_dir=tmp_path,
        totals=dict.fromkeys(
            ("actual_generation_calls", "actual_evaluator_calls", "paid_judge_cache_misses"), 0
        ),
    )
    manifest = {
        "max_concepts": 1,
        "master_seed": 7,
        "soft_prompt": config.effective_identity(master_seed=7),
        "soft_prompt_training_source": {
            "train_data": {
                "path": "/source/train_data.parquet",
                "sha256": "b" * 64,
                "byte_length": 1,
            },
            "metadata": {"path": "/source/metadata.jsonl", "sha256": "c" * 64, "byte_length": 1},
            "pair_sha256": {"0": training_pair_identity(training)},
        },
        "files": {
            str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in tmp_path.rglob("*")
            if path.is_file()
        },
    }
    return tmp_path, manifest


def _validate(root: Path, manifest: dict[str, Any], **kwargs: Any) -> list[Path]:
    return _validate_search_receipts(
        pressure=Pressure.SOFT_PROMPT,
        root=root,
        manifest=manifest,
        files=manifest["files"],
        methods=("DiffMean",),
        concept_ids=(0,),
        factors={"DiffMean": 1.0},
        token_counts=(0, 1, 3),
        **kwargs,
    )


def _receipt(root: Path, count: int = 1) -> Path:
    return root / "soft_prompt/DiffMean/c0" / f"k{count}" / "search.json"


def _rewrite(
    root: Path, manifest: dict[str, Any], document: dict[str, Any], count: int = 1
) -> None:
    path = _receipt(root, count)
    path.write_text(json.dumps(document), encoding="utf-8")
    manifest["files"][str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()


def test_valid_receipts_require_every_tensor_without_relaxing_gcg(receipt_tree) -> None:
    root, manifest = receipt_tree
    required = _validate(root, manifest)
    assert len(required) == 5
    assert sum(path.suffix == ".safetensors" for path in required) == 2
    identity = json.loads(_receipt(root).read_text())["model_tokenizer_identity"]
    with pytest.raises(AxBenchAdvError, match="model identity"):
        _validate_gcg_model_identity(
            identity, method="DiffMean", concept_id=0, factor=1.0, relative=Path("search.json")
        )


@pytest.mark.parametrize(
    ("fields", "value"),
    [
        (("schema_version",), True),
        (("method",), "LoRA"),
        (("concept_id",), True),
        (("concept_id",), 1),
        (("concept",), ""),
        (("factor",), True),
        (("factor",), 2.0),
        (("token_count",), True),
        (("token_count",), 3),
        (("objective",), "activation_l2"),
        (("training_pair_identity",), "e" * 64),
        (("layouts_sha256",), "UPPER"),
        (("initialization_token_ids",), [True]),
        (("initialization_token_ids",), []),
        (("initialization_token_ids",), [999]),
        (("objective_value",), True),
        (("objective_value",), float("nan")),
        (("suffix",), "fake text"),
        (("diagnostics", "control"), True),
        (("diagnostics", "best_step"), -1),
        (("diagnostics", "best_step"), 3),
        (("diagnostics", "updates"), 1),
        (("diagnostics", "full_population_scores"), 2),
        (("diagnostics", "updates"), True),
        (("model_tokenizer_identity", "generation_conditioning"), "placeholder_text"),
        (
            ("model_tokenizer_identity", "embedding_layout"),
            [{"name": "wte", "width": 2, "dtype": "torch.float32"}],
        ),
        (
            ("model_tokenizer_identity", "embedding_layout"),
            [{"name": "input_embeddings", "width": 1, "dtype": "torch.float32"}],
        ),
        (
            ("model_tokenizer_identity", "embedding_layout"),
            [{"name": "input_embeddings", "width": 2, "dtype": "torch.int64"}],
        ),
        (("model_tokenizer_identity", "model_name"), ""),
        (("direction_calibration",), {}),
        (("tensor", "path"), "../../different.safetensors"),
        (("tensor", "shape"), [True, 2]),
        (("tensor", "key"), "wrong"),
        (("tensor", "byte_length"), 0),
    ],
)
def test_rehashed_receipt_contract_mutations_fail(
    receipt_tree, fields: tuple[str, ...], value: object
) -> None:
    root, manifest = receipt_tree
    document = json.loads(_receipt(root).read_text())
    target = document
    for field in fields[:-1]:
        target = target[field]
    target[fields[-1]] = value
    _rewrite(root, manifest, document)
    with pytest.raises(AxBenchAdvError):
        _validate(root, manifest)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "corrupt",
        "key",
        "extra_key",
        "rank",
        "shape",
        "dtype",
        "nan",
        "infinity",
        "native_overflow",
        "substituted",
    ],
)
def test_rehashed_tensor_bytes_must_match_descriptor_and_tensor_contract(
    receipt_tree, kind: str
) -> None:
    root, manifest = receipt_tree
    document = json.loads(_receipt(root).read_text())
    path = _receipt(root).with_name("embeddings.safetensors")
    if kind == "missing":
        path.unlink()
    elif kind == "corrupt":
        path.write_bytes(b"invalid safetensors bytes")
    else:
        values = {
            "key": {"wrong": t.zeros(1, 2)},
            "extra_key": {"embeddings": t.zeros(1, 2), "extra": t.ones(1)},
            "rank": {"embeddings": t.zeros(2)},
            "shape": {"embeddings": t.zeros(2, 2)},
            "dtype": {"embeddings": t.zeros(1, 2, dtype=t.float64)},
            "nan": {"embeddings": t.full((1, 2), float("nan"))},
            "infinity": {"embeddings": t.full((1, 2), float("inf"))},
            "native_overflow": {"embeddings": t.full((1, 2), 70000.0)},
            "substituted": {"embeddings": t.full((1, 2), 42.0)},
        }
        save_file(values[kind], str(path))
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][str(path.relative_to(root))] = digest
        if kind != "substituted":
            document["tensor"].update(sha256=digest, byte_length=path.stat().st_size)
    if kind == "native_overflow":
        document["model_tokenizer_identity"]["embedding_layout"][0]["dtype"] = "torch.float16"
    _rewrite(root, manifest, document)
    with pytest.raises(AxBenchAdvError, match="tensor"):
        _validate(root, manifest)


@pytest.mark.parametrize(
    "field,value",
    [
        ("master_seed", 8),
        ("optimizer", {"name": "SGD"}),
        ("native_cast_policy", "native_master"),
        ("learning_rate", 0.02),
        ("optimized_token_counts", [0, 1]),
    ],
)
def test_manifest_effective_config_mutations_fail(receipt_tree, field: str, value: object) -> None:
    root, manifest = receipt_tree
    manifest["soft_prompt"][field] = value
    with pytest.raises(AxBenchAdvError):
        _validate(root, manifest)


def test_optimizer_bool_alias_is_rejected_even_after_every_receipt_is_rehashed(
    receipt_tree,
) -> None:
    root, manifest = receipt_tree
    manifest["soft_prompt"]["optimizer"]["foreach"] = 0
    for count in (0, 1, 3):
        document = json.loads(_receipt(root, count).read_text())
        document["effective_config"] = manifest["soft_prompt"]
        _rewrite(root, manifest, document, count)
    with pytest.raises(AxBenchAdvError, match="effective config"):
        _validate(root, manifest)


@pytest.mark.parametrize("change", ["control_flag", "updates", "tensor"])
def test_zero_receipt_is_only_a_control(receipt_tree, change: str) -> None:
    root, manifest = receipt_tree
    document = json.loads(_receipt(root, 0).read_text())
    if change == "control_flag":
        document["diagnostics"]["control"] = False
    elif change == "updates":
        document["diagnostics"]["updates"] = 1
    else:
        save_file(
            {"embeddings": t.zeros(0, 2)},
            str(_receipt(root, 0).with_name("embeddings.safetensors")),
        )
    _rewrite(root, manifest, document, 0)
    with pytest.raises(AxBenchAdvError):
        _validate(root, manifest)


def _project_identity(
    identity: dict[str, Any], *, compute_dtype: str, partition: bool
) -> dict[str, Any]:
    return {
        "runtime": "project_encoder_decoder",
        "concept_id": identity["concept_id"],
        "method": identity["method"],
        "factor": identity["factor"],
        "rank_sentinel_id": 0,
        "vocab_size": 3,
        "embedding_dim": 2,
        "frozen_state": "clean_real_prompt_S0_before_assistant_marker",
        "generation_conditioning": "clean_prompt_S0_dynamic_decode",
        "compute_dtype": compute_dtype,
        "embedding_layout": (
            [
                {"name": "wte", "width": 1, "dtype": "torch.float32"},
                {"name": "value:0", "width": 1, "dtype": "torch.float32"},
            ]
            if partition
            else [{"name": "wte", "width": 2, "dtype": "torch.float32"}]
        ),
    }


@pytest.mark.parametrize("mismatch", ["compute_dtype", "partition"])
def test_cross_slice_identity_binds_equal_width_partitions_and_actual_compute_dtype(
    receipt_tree, mismatch: str
) -> None:
    root, manifest = receipt_tree
    for count in (1, 3):
        document = json.loads(_receipt(root, count).read_text())
        document["model_tokenizer_identity"] = _project_identity(
            document["model_tokenizer_identity"], compute_dtype="torch.float32", partition=False
        )
        _rewrite(root, manifest, document, count)
    compatibility = {}
    _validate(root, manifest, compatibility=compatibility)
    for count in (1, 3):
        document = json.loads(_receipt(root, count).read_text())
        document["model_tokenizer_identity"] = _project_identity(
            document["model_tokenizer_identity"],
            compute_dtype="torch.bfloat16" if mismatch == "compute_dtype" else "torch.float32",
            partition=mismatch == "partition",
        )
        _rewrite(root, manifest, document, count)
    with pytest.raises(AxBenchAdvError, match="runtime or calibration"):
        _validate(root, manifest, compatibility=copy.deepcopy(compatibility))
