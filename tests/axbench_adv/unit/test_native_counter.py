"""Native decoder own-direction counter fit and generation contracts."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pandas as pd
import pytest
import torch as t
import torch.nn as nn
from omegaconf import OmegaConf

from steering.eval.axbench_adv import native_counter
from steering.eval.axbench_adv.direction_fitting import SIGNED_CALIBRATION_KEY
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    load_direction,
    save_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import RenderedFitRow
from steering.eval.axbench_upstream.native_contract import METHOD as NATIVE_METHOD
from steering.eval.axbench_upstream.native_contract import load_native_output
from steering.safetensors_io import load_safetensors_file_snapshot


class AddOneBlock(nn.Module):
    def forward(self, value: t.Tensor) -> t.Tensor:
        return value + 1


class RaiseBlock(nn.Module):
    def forward(self, value: t.Tensor) -> t.Tensor:
        del value
        raise RuntimeError("block failure")


class TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_layer=1, n_embd=3, sequence_len=32)
        self.anchor = nn.Parameter(t.zeros(()))
        self.blocks = nn.ModuleList([AddOneBlock()])
        self.transformer = nn.ModuleDict({"h": self.blocks})

    def forward(self, idx: t.Tensor) -> t.Tensor:
        values = t.arange(
            idx.shape[0] * idx.shape[1] * self.config.n_embd,
            dtype=t.float32,
            device=idx.device,
        ).reshape(idx.shape[0], idx.shape[1], self.config.n_embd)
        return self.blocks[0](values)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.slow)])
def test_counter_subtracts_on_prefill_and_decode_and_always_cleans_up(device: str) -> None:
    if device == "cuda" and not t.cuda.is_available():
        pytest.skip("CUDA unavailable")
    model = TinyDecoder().to(device)
    direction = t.tensor([1.0, 0.0, -1.0], dtype=t.float32)
    prefill_ids = t.tensor([[1, 2, 3]], dtype=t.int64, device=device)
    decode_ids = t.tensor([[4]], dtype=t.int64, device=device)
    base_prefill = model(prefill_ids)
    base_decode = model(decode_ids)

    with native_counter.scoped_native_counter(
        model,
        layer=0,
        direction=direction,
        magnitude=0.5,
    ):
        assert t.equal(model(prefill_ids), base_prefill - 0.5 * direction.to(device))
        assert t.equal(model(decode_ids), base_decode - 0.5 * direction.to(device))
    assert t.equal(model(prefill_ids), base_prefill)
    assert not model.blocks[0]._forward_hooks

    with pytest.raises(RuntimeError, match="inside"):
        with native_counter.scoped_native_counter(
            model,
            layer=0,
            direction=direction,
            magnitude=1.0,
        ):
            raise RuntimeError("inside")
    assert not model.blocks[0]._forward_hooks


def test_zero_counter_is_bit_exact_and_invalid_geometry_fails_before_forward() -> None:
    model = TinyDecoder()
    ids = t.tensor([[1, 2]], dtype=t.int64)
    direction = t.tensor([1.0, 0.0, -1.0], dtype=t.float32)
    expected = model(ids)
    with native_counter.scoped_native_counter(
        model,
        layer=0,
        direction=direction,
        magnitude=0.0,
    ):
        assert t.equal(model(ids), expected)
        assert not model.blocks[0]._forward_hooks
    for invalid in (
        direction.to(t.float64),
        direction.reshape(1, 3),
        t.ones(2, dtype=t.float32),
        t.tensor([1.0, float("nan"), 0.0], dtype=t.float32),
    ):
        with pytest.raises(AxBenchAdvError, match="direction"):
            with native_counter.scoped_native_counter(
                model,
                layer=0,
                direction=invalid,
                magnitude=1.0,
            ):
                pytest.fail("invalid direction entered the intervention scope")


def test_native_capture_uses_the_same_block_output_and_removes_hook() -> None:
    model = TinyDecoder()
    rendered = RenderedFitRow(
        ids=(4, 5, 6, 7),
        mask=(0, 0, 1, 1),
        prompt_token_count=2,
    )
    captured = native_counter.capture_native_post_prefix_activations(
        model,
        rendered,
        layer=0,
        prefix_length=2,
    )
    expected = model(t.tensor([[4, 5, 6, 7]], dtype=t.int64))[0, 2:, :]
    assert captured.dtype is t.float32
    assert captured.device.type == "cpu"
    assert t.equal(captured, expected)
    assert not model.blocks[0]._forward_hooks

    model.blocks[0] = RaiseBlock()
    with pytest.raises(RuntimeError, match="block failure"):
        native_counter.capture_native_post_prefix_activations(
            model,
            rendered,
            layer=0,
            prefix_length=2,
        )
    assert not model.blocks[0]._forward_hooks


def test_fitted_direction_rejects_wrong_norm_and_provenance() -> None:
    contract = {
        "concept_id": 0,
        "intervention_layer": 0,
        "n_embd": 3,
        "n_positive_rows": 2,
        "n_negative_rows": 2,
    }
    metadata = {
        "method": "diffmean",
        "intervention_layer": 0,
        "n_positive_rows": 2,
        "n_negative_rows": 2,
        "native_contract": contract,
    }
    valid = FittedDirection(
        direction=t.tensor([1.0, 0.0, 0.0]),
        max_activation=1.0,
        metadata=metadata,
    )
    native_counter._validate_direction(
        valid,
        method="DiffMean",
        contract=contract,
        seed=42,
    )
    with pytest.raises(AxBenchAdvError, match="provenance"):
        native_counter._validate_direction(
            FittedDirection(valid.direction, 1.0, {**metadata, "native_contract": {}}),
            method="DiffMean",
            contract=contract,
            seed=42,
        )
    with pytest.raises(AxBenchAdvError, match="LinearProbe"):
        native_counter._validate_direction(
            FittedDirection(
                t.tensor([2.0, 0.0, 0.0]),
                1.0,
                {
                    **metadata,
                    "method": "probe",
                    "seed": 42,
                    "concept_id": 0,
                    "estimator": {},
                },
            ),
            method="LinearProbe",
            contract=contract,
            seed=42,
        )


def test_fitted_direction_accepts_zero_tiny_diffmean_and_signed_probe() -> None:
    contract = {
        "concept_id": 0,
        "intervention_layer": 0,
        "n_embd": 3,
        "n_positive_rows": 2,
        "n_negative_rows": 2,
    }
    common = {
        "intervention_layer": 0,
        "n_positive_rows": 2,
        "n_negative_rows": 2,
        "native_contract": contract,
    }
    for direction, calibration in (
        (t.zeros(3), 0.0),
        (t.tensor([1e-9, 0.0, 0.0]), -0.25),
    ):
        native_counter._validate_direction(
            FittedDirection(direction, calibration, {**common, "method": "diffmean"}),
            method="DiffMean",
            contract=contract,
            seed=42,
        )
    native_counter._validate_direction(
        FittedDirection(
            t.tensor([1.0, 0.0, 0.0]),
            -0.75,
            {
                **common,
                "method": "probe",
                "seed": 42,
                "concept_id": 0,
                "estimator": {},
            },
        ),
        method="LinearProbe",
        contract=contract,
        seed=42,
    )


def test_native_direction_signed_roundtrip_and_scaled_consumer_guard(tmp_path) -> None:
    positive = FittedDirection(t.tensor([1.0, 0.0]), 0.5, {"method": "diffmean"})
    native_positive = tmp_path / "native-positive.safetensors"
    shared_positive = tmp_path / "shared-positive.safetensors"
    native_counter._save_native_direction(native_positive, positive)
    save_direction(shared_positive, positive)
    assert native_positive.read_bytes() == shared_positive.read_bytes()

    signed = FittedDirection(t.tensor([1.0, 0.0]), -0.75, {"method": "diffmean"})
    signed_path = tmp_path / "signed.safetensors"
    native_counter._save_native_direction(signed_path, signed)
    loaded = native_counter._load_native_direction(signed_path)
    assert t.equal(loaded.direction, signed.direction)
    assert loaded.max_activation == -0.75
    assert loaded.metadata == signed.metadata
    _, metadata = load_safetensors_file_snapshot(signed_path)
    assert metadata is not None
    assert metadata[SIGNED_CALIBRATION_KEY] == "-0.75"
    assert "max_activation" not in metadata
    with pytest.raises(ValueError, match="missing max_activation"):
        load_direction(signed_path)

    zero = FittedDirection(t.zeros(2), 0.0, {"method": "diffmean"})
    zero_path = tmp_path / "zero.safetensors"
    native_counter._save_native_direction(zero_path, zero)
    loaded_zero = native_counter._load_native_direction(zero_path)
    assert t.equal(loaded_zero.direction, zero.direction)
    assert loaded_zero.max_activation == 0.0


def test_magnitude_labels_preserve_legacy_paths_and_support_custom_values() -> None:
    assert [native_counter._magnitude_label(value) for value in (0.25, 0.5, 1.0, 2.0)] == [
        "m0p25",
        "m0p5",
        "m1",
        "m2",
    ]
    assert native_counter._magnitude_label(64.0) == "m64.0"
    assert native_counter._magnitude_label(4096.0) == "m4096.0"


def test_point_selection_is_placement_only_and_requires_strict_grid_subsets() -> None:
    cfg = OmegaConf.create({"point_methods": None, "point_magnitudes": None})
    assert native_counter._point_selection(cfg) == (
        native_counter.COUNTER_METHODS,
        native_counter.COUNTER_MAGNITUDES,
    )
    cfg.point_methods = ["LinearProbe"]
    configured = (64.0, 512.0, 4096.0)
    cfg.point_magnitudes = [512, 4096]
    assert native_counter._point_selection(cfg, configured) == (
        ("LinearProbe",),
        (512.0, 4096.0),
    )
    for methods, magnitudes in (
        ([], [64]),
        (["DiffMean", "DiffMean"], [64]),
        (["unknown"], [64]),
        (["DiffMean"], []),
        (["DiffMean"], [64, 64]),
        (["DiffMean"], [512, 64]),
        (["DiffMean"], [True]),
        (["DiffMean"], ["64"]),
        (["DiffMean"], [float("nan")]),
        (["DiffMean"], [float("inf")]),
        (["DiffMean"], [0]),
        (["DiffMean"], [-64]),
        (["DiffMean"], [128]),
    ):
        cfg.point_methods = methods
        cfg.point_magnitudes = magnitudes
        with pytest.raises(AxBenchAdvError, match="point_"):
            native_counter._point_selection(cfg, configured)


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_identity_rejects_wrong_checkpoint_tokenizer_geometry_and_layer(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_001003.pt").write_bytes(b"model")
    metadata = {
        "step": 1003,
        "sft_loss_mask": "completion_only",
        "model_config": {"n_layer": 12, "n_embd": 768, "sequence_len": 1024},
        "user_config": {"mixture": "standard_steered", "model_backend": "nanochat"},
    }
    meta_path = checkpoint / "meta_001003.json"
    meta_path.write_text(json.dumps(metadata))
    tokenizer = tmp_path / "tokenizer.pkl"
    tokenizer.write_bytes(b"tokenizer")
    expected = {
        "step": 1003,
        "model_sha256": _sha(checkpoint / "model_001003.pt"),
        "meta_sha256": _sha(meta_path),
        "tokenizer_sha256": _sha(tokenizer),
        "n_layer": 12,
        "n_embd": 768,
    }
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": str(checkpoint),
            "step": 1003,
            "tokenizer_path": str(tokenizer),
            "layer": 9,
            "expected": expected,
        }
    )
    source = {"checkpoint": dict(expected)}
    runtime, geometry = native_counter.validate_runtime_identity(
        cfg=cfg,
        source_manifest=source,
    )
    assert runtime["model_sha256"] == expected["model_sha256"]
    assert geometry == {"n_layer": 12, "n_embd": 768, "sequence_len": 1024}

    cfg.expected.tokenizer_sha256 = "0" * 64
    with pytest.raises(AxBenchAdvError, match="identity differs"):
        native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)
    cfg.expected.tokenizer_sha256 = expected["tokenizer_sha256"]
    cfg.layer = 12
    with pytest.raises(AxBenchAdvError, match="layer"):
        native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)


def test_d14_runtime_rejects_wrong_identity_geometry_layer_and_parameter_count(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_001777.pt").write_bytes(b"d14-model")
    metadata = {
        "step": 1777,
        "sft_loss_mask": "completion_only",
        "model_config": {"n_layer": 14, "n_embd": 896, "sequence_len": 1024},
        "user_config": {"mixture": "standard_steered", "model_backend": "nanochat"},
    }
    meta_path = checkpoint / "meta_001777.json"
    meta_path.write_text(json.dumps(metadata))
    tokenizer = tmp_path / "tokenizer.pkl"
    tokenizer.write_bytes(b"d14-tokenizer")
    expected = {
        "step": 1777,
        "model_sha256": _sha(checkpoint / "model_001777.pt"),
        "meta_sha256": _sha(meta_path),
        "tokenizer_sha256": _sha(tokenizer),
        "n_layer": 14,
        "n_embd": 896,
    }
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": str(checkpoint),
            "step": 1777,
            "tokenizer_path": str(tokenizer),
            "layer": 11,
            "expected": expected,
        }
    )
    source = {"checkpoint": dict(expected)}
    _, geometry = native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)
    assert geometry == {"n_layer": 14, "n_embd": 896, "sequence_len": 1024}

    cfg.layer = 9
    with pytest.raises(AxBenchAdvError, match="reviewed geometry mapping"):
        native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)
    cfg.layer = 11
    cfg.expected.model_sha256 = "0" * 64
    with pytest.raises(AxBenchAdvError, match="identity differs"):
        native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)
    cfg.expected.model_sha256 = expected["model_sha256"]
    metadata["model_config"]["sequence_len"] = 2048
    meta_path.write_text(json.dumps(metadata))
    cfg.expected.meta_sha256 = _sha(meta_path)
    source["checkpoint"]["meta_sha256"] = cfg.expected.meta_sha256
    with pytest.raises(AxBenchAdvError, match="geometry is not reviewed"):
        native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)

    class TinyD14(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = nn.Parameter(t.zeros(()))
            self.config = SimpleNamespace(n_layer=14, n_embd=896, sequence_len=1024)

    with pytest.raises(AxBenchAdvError, match="parameter count differs"):
        native_counter._validate_loaded_model(
            TinyD14(), {"n_layer": 14, "n_embd": 896, "sequence_len": 1024}
        )


def test_concept_scope_preserves_legacy50_and_selects_only_standard100_prefix() -> None:
    frame50 = pd.DataFrame({"concept_id": [concept for concept in range(50) for _ in range(10)]})
    legacy_cfg = OmegaConf.create({"max_concepts": 50, "concept_count": None})
    concept_ids, selected, scope = native_counter._concept_scope(
        cfg=legacy_cfg,
        source_manifest={"source": {"max_concepts": 50}},
        source_frame=frame50,
    )
    assert concept_ids == tuple(range(50))
    assert selected is frame50
    assert scope is None

    frame100 = pd.DataFrame({"concept_id": [concept for concept in range(100) for _ in range(10)]})
    prefix_cfg = OmegaConf.create({"max_concepts": 100, "concept_count": 50})
    concept_ids, selected, scope = native_counter._concept_scope(
        cfg=prefix_cfg,
        source_manifest={"source": {"max_concepts": 100}},
        source_frame=frame100,
    )
    assert concept_ids == tuple(range(50))
    assert selected.concept_id.unique().tolist() == list(range(50))
    assert scope == {
        "kind": "canonical_prefix",
        "concept_count": 50,
        "concept_ids": list(range(50)),
        "authenticated_source_max_concepts": 100,
    }
    for max_concepts, concept_count in ((100, 49), (100, 51), (50, 50)):
        cfg = OmegaConf.create({"max_concepts": max_concepts, "concept_count": concept_count})
        with pytest.raises(AxBenchAdvError, match="canonical standard100 prefix"):
            native_counter._concept_scope(
                cfg=cfg,
                source_manifest={"source": {"max_concepts": max_concepts}},
                source_frame=frame100,
            )


class FakeAdapter:
    def __init__(self) -> None:
        self.model = TinyDecoder()
        self.calls: list[tuple[str, int, t.Tensor]] = []

    def generate_user_text(
        self,
        user_text: str,
        *,
        seed: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str:
        del max_new_tokens, temperature, top_k
        residual = self.model(t.tensor([[1, 2]], dtype=t.int64))
        self.calls.append((user_text, seed, residual.detach().clone()))
        return f"generation-{seed}-{float(residual.sum())}"


def test_point_resume_has_exact_coverage_and_rejects_extra_cells(tmp_path, monkeypatch) -> None:
    source_frame = pd.DataFrame(
        [
            {
                "dataset_name": "AlpacaEval",
                "concept_id": 0,
                "input_id": input_id,
                "input_concept": "cats",
                "factor": 1.0,
                "original_prompt": f"question {input_id}",
                "steered_input": f"wrapped {input_id}",
                "raw_prompt": f"question {input_id}",
                "steering_prefix": "mention cats",
                "native_user_prompt": f"mention cats\n\nQuestion: question {input_id}",
                f"{NATIVE_METHOD}_steered_generation": f"control {input_id}",
            }
            for input_id in range(10)
        ]
    )
    source_manifest = {
        "schema_version": 1,
        "format": "axbench-native-decoder-v1",
        "method": NATIVE_METHOD,
        "factors": [1.0],
        "policy": {"name": "axbench_frozen_prompt_steering_v1"},
        "source": {"max_concepts": 1},
        "generation": {"seed": 42, "max_new_tokens": 8, "temperature": 1.0, "top_k": 5},
    }
    identity = {
        "inputs": {
            "source_manifest": {"path": "source-manifest", "sha256": "a" * 64},
            "source_parquet": {"path": "source-parquet", "sha256": "b" * 64},
        }
    }
    contract = {"n_embd": 3}
    fitted = {
        0: {
            "DiffMean": FittedDirection(
                direction=t.tensor([1.0, 0.0, 0.0]),
                max_activation=1.0,
                metadata=contract,
            )
        }
    }
    direction_path = native_counter._direction_path(tmp_path, 0, "DiffMean")
    direction_path.parent.mkdir(parents=True)
    direction_path.write_bytes(b"direction")
    monkeypatch.setattr(
        native_counter,
        "load_native_output",
        lambda manifest_path: (
            json.loads(manifest_path.read_text()),
            pd.read_parquet(manifest_path.parent / "steering_data.parquet"),
        ),
    )
    adapter = FakeAdapter()
    complete, generated = native_counter._generate_point(
        get_adapter=lambda: adapter,
        revalidate_inputs=lambda: None,
        source_manifest=source_manifest,
        source_frame=source_frame,
        output=tmp_path,
        identity=identity,
        runtime_checkpoint={"step": 1003},
        fitted=fitted,
        method="DiffMean",
        magnitude=0.25,
        layer=0,
        max_new_cells=3,
    )
    assert not complete and generated == 3
    complete, generated = native_counter._generate_point(
        get_adapter=lambda: adapter,
        revalidate_inputs=lambda: None,
        source_manifest=source_manifest,
        source_frame=source_frame,
        output=tmp_path,
        identity=identity,
        runtime_checkpoint={"step": 1003},
        fitted=fitted,
        method="DiffMean",
        magnitude=0.25,
        layer=0,
        max_new_cells=7,
    )
    assert complete and generated == 7 and len(adapter.calls) == 10
    point = tmp_path / "points/diffmean/m0p25"
    assert len(list((point / "cells").glob("*.json"))) == 10
    output = pd.read_parquet(point / "steering_data.parquet")
    assert output[f"{NATIVE_METHOD}_steered_generation"].nunique() == 10
    assert output.native_user_prompt.tolist() == source_frame.native_user_prompt.tolist()
    records = [json.loads(path.read_text()) for path in sorted((point / "cells").glob("*.json"))]
    assert (
        len({(entry["cell"]["concept_id"], entry["cell"]["input_id"]) for entry in records}) == 10
    )
    assert [entry["cell"]["seed"] for entry in records] == [
        native_counter.cell_seed(42, 0, input_id) for input_id in range(10)
    ]
    (point / "cells/unexpected.json").write_text("{}")
    with pytest.raises(AxBenchAdvError, match="unexpected cells"):
        native_counter._generate_point(
            get_adapter=lambda: adapter,
            revalidate_inputs=lambda: None,
            source_manifest=source_manifest,
            source_frame=source_frame,
            output=tmp_path,
            identity=identity,
            runtime_checkpoint={"step": 1003},
            fitted=fitted,
            method="DiffMean",
            magnitude=0.25,
            layer=0,
            max_new_cells=None,
        )


def test_standard100_prefix_point_keeps_parent_provenance_and_is_native_judge_compatible(
    tmp_path,
) -> None:
    rows = []
    for concept_id in range(50):
        prefix = f"Mention concept {concept_id}."
        for input_id in range(10):
            prompt = f"question {concept_id}-{input_id}"
            native_user = f"{prefix}\n\nQuestion: {prompt}"
            rows.append(
                {
                    "dataset_name": "AlpacaEval",
                    "concept_id": concept_id,
                    "input_concept": f"concept-{concept_id}",
                    "input_id": input_id,
                    "factor": 1.0,
                    "original_prompt": prompt,
                    "steered_input": (
                        f"<|im_start|>user\n{native_user}<|im_end|>\n<|im_start|>assistant\n"
                    ),
                    "input": prompt,
                    "suppress_original": "",
                    "suppress_rewrite": "",
                    "steered_prompt": native_user,
                    "defense": "",
                    "sae_link": "",
                    "sae_id": "",
                    "native_user_prompt": native_user,
                    "steering_prefix": prefix,
                    f"{NATIVE_METHOD}_steered_generation": (
                        f"authenticated-control-{concept_id}-{input_id}"
                    ),
                }
            )
    source_frame = pd.DataFrame(rows)
    source_manifest = {
        "schema_version": 1,
        "format": "axbench-native-decoder-v1",
        "method": NATIVE_METHOD,
        "factors": [1.0],
        "policy": {"name": "axbench_frozen_prompt_steering_v1"},
        "source": {
            "max_concepts": 100,
            "input_path": "/authenticated/standard100.parquet",
            "input_sha256": "c" * 64,
        },
        "generation": {"seed": 42, "max_new_tokens": 8, "temperature": 1.0, "top_k": 5},
    }
    scope = {
        "kind": "canonical_prefix",
        "concept_count": 50,
        "concept_ids": list(range(50)),
        "authenticated_source_max_concepts": 100,
    }
    identity = {
        "inputs": {
            "source_manifest": {"path": "/source/run_manifest.json", "sha256": "a" * 64},
            "source_parquet": {"path": "/source/steering_data.parquet", "sha256": "b" * 64},
        },
        "configuration": {"concept_scope": scope},
    }
    fitted = {}
    for concept_id in range(50):
        direction_path = native_counter._direction_path(tmp_path, concept_id, "DiffMean")
        direction_path.parent.mkdir(parents=True)
        direction_path.write_bytes(f"direction-{concept_id}".encode())
        fitted[concept_id] = {
            "DiffMean": FittedDirection(
                direction=t.tensor([1.0, 0.0, 0.0]),
                max_activation=1.0,
                metadata={},
            )
        }
    adapter = FakeAdapter()
    complete, generated = native_counter._generate_point(
        get_adapter=lambda: adapter,
        revalidate_inputs=lambda: None,
        source_manifest=source_manifest,
        source_frame=source_frame,
        output=tmp_path,
        identity=identity,
        runtime_checkpoint={"step": 1777},
        fitted=fitted,
        method="DiffMean",
        magnitude=0.25,
        layer=0,
        max_new_cells=None,
        concept_ids=tuple(range(50)),
    )
    assert complete and generated == 500
    point = tmp_path / "points/diffmean/m0p25"
    manifest, validated = load_native_output(point / "run_manifest.json")
    assert manifest["source"]["max_concepts"] == 50
    assert len(validated) == 500
    extension = manifest["counter_extension"]
    assert extension["source_scope"] == {
        **scope,
        "authenticated_source_manifest": identity["inputs"]["source_manifest"],
        "authenticated_source_parquet": identity["inputs"]["source_parquet"],
    }
    assert (
        source_frame[f"{NATIVE_METHOD}_steered_generation"]
        .str.startswith("authenticated-control-")
        .all()
    )

    legacy_identity = {**identity, "configuration": {}}
    with pytest.raises(AxBenchAdvError, match="invalid/conflicting native counter cell"):
        native_counter._generate_point(
            get_adapter=lambda: adapter,
            revalidate_inputs=lambda: None,
            source_manifest=source_manifest,
            source_frame=source_frame,
            output=tmp_path,
            identity=legacy_identity,
            runtime_checkpoint={"step": 1777},
            fitted=fitted,
            method="DiffMean",
            magnitude=0.25,
            layer=0,
            max_new_cells=None,
            concept_ids=tuple(range(50)),
        )
