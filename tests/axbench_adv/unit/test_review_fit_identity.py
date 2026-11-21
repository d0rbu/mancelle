"""Focused tests for the compact project-fit identity builder."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_adv.direction_fitting import build_shared_fit_identity
from steering.eval.axbench_upstream.methods import MethodSetup
from steering.stats_io import content_sha256


@pytest.fixture
def identity_case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    meta = {
        "step": 1125,
        "model_config": {"n_layer": 27, "n_embd": 1664},
    }
    companion = tmp_path / "meta_001125.json"
    companion.write_text(json.dumps(meta), encoding="utf-8")
    model = tmp_path / "model_001125.pt"
    model.write_bytes(b"checkpoint bytes are hashed, not unpickled")
    prepared = tmp_path / "prepared"
    (prepared / "generate").mkdir(parents=True)
    train = prepared / "generate/train_data.parquet"
    train.write_bytes(b"opaque training rows")
    tokenizer = tmp_path / "cache/tokenizer/tokenizer.pkl"
    tokenizer.parent.mkdir(parents=True)
    tokenizer.write_bytes(b"tokenizer identity bytes")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "cache"))
    stats = tmp_path / "feature_statistics.safetensors"
    save_file(
        {"counts": t.tensor([1, 2, 3], dtype=t.int64)},
        str(stats),
        metadata={"kind": "test"},
    )
    setup = SimpleNamespace(target_meta_path=companion, meta=meta)
    return {
        "setup": setup,
        "prepared": prepared,
        "companion": companion,
        "model": model,
        "tokenizer": tokenizer,
        "stats": stats,
        "train": train,
    }


def _identity(case: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return build_shared_fit_identity(
        setup=cast(MethodSetup, case["setup"]),
        prepared_dir=case["prepared"],
        feature_statistics_path=case["stats"],
        layer=21,
        seed=42,
        **overrides,
    )


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_binds_target_weights_metadata_and_records_reproduction_inputs(
    identity_case: dict[str, Any],
) -> None:
    identity = _identity(identity_case)
    checkpoint = identity["checkpoint"]
    assert checkpoint["target_model"]["sha256"] == _sha(identity_case["model"])
    assert checkpoint["companion_metadata"]["sha256"] == _sha(
        identity_case["companion"]
    )
    assert identity["fit"]["layer"] == 21
    assert identity["fit"]["decoder_width"] == 1664
    assert identity["fit"]["seed"] == 42
    reproduction = identity["reproducibility"]
    assert reproduction["prepared_train"]["sha256"] == _sha(identity_case["train"])
    assert reproduction["statistics_content_sha256"] == content_sha256(
        identity_case["stats"]
    )
    assert reproduction["tokenizer"]["sha256"] == _sha(identity_case["tokenizer"])


def test_builder_streams_checkpoint_instead_of_reading_it_whole(
    identity_case: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = identity_case["model"]
    payload = bytes(range(256)) * 16385
    model.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    original_read_bytes = Path.read_bytes

    def no_whole_checkpoint_read(path: Path) -> bytes:
        assert path != model, "multi-gigabyte checkpoint must be hashed as a stream"
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", no_whole_checkpoint_read)
    identity = _identity(identity_case)["checkpoint"]["target_model"]
    assert identity == {
        "name": model.name, "byte_length": len(payload), "sha256": expected,
    }


@pytest.mark.parametrize("target", ["model", "companion"])
def test_operational_target_mutation_changes_identity(
    identity_case: dict[str, Any], target: str
) -> None:
    before = _identity(identity_case)
    path = identity_case[target]
    path.write_bytes(path.read_bytes() + b"changed")
    after = _identity(identity_case)
    key = "target_model" if target == "model" else "companion_metadata"
    assert before["checkpoint"][key]["sha256"] != after["checkpoint"][key]["sha256"]


@pytest.mark.parametrize(
    ("key", "value"),
    [("layer", True), ("layer", -1), ("layer", 27), ("seed", True), ("seed", -1)],
)
def test_builder_rejects_invalid_recipe_coordinates(
    identity_case: dict[str, Any], key: str, value: Any
) -> None:
    kwargs = {"layer": 21, "seed": 42, key: value}
    with pytest.raises(ValueError):
        build_shared_fit_identity(
            setup=cast(MethodSetup, identity_case["setup"]),
            prepared_dir=identity_case["prepared"],
            feature_statistics_path=identity_case["stats"],
            **kwargs,
        )
