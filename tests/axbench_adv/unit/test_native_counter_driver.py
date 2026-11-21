"""Native decoder own-direction counter fit and generation contracts."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch as t
import torch.nn as nn
from omegaconf import OmegaConf

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.eval.axbench_adv import native_counter
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_upstream import native_decoder as native_producer
from steering.eval.axbench_upstream.native_contract import METHOD as NATIVE_METHOD
from steering.eval.axbench_upstream.native_contract import load_native_output
from steering.immutable_io import ImmutableArtifactError


class AddOneBlock(nn.Module):
    def forward(self, value: t.Tensor) -> t.Tensor:
        return value + 1


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


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_identity_rejects_wrong_checkpoint_tokenizer_geometry_and_layer(
    counter_run,
) -> None:
    cfg, adapter = counter_run
    source = json.loads(Path(cfg.source_manifest).read_text())
    runtime, geometry = native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)
    assert runtime == adapter.identity
    assert geometry == {"n_layer": 1, "n_embd": 3, "sequence_len": 32}

    expected = dict(cfg.expected)
    for field, value in (
        ("tokenizer_sha256", "0" * 64),
        ("model_sha256", expected["model_sha256"].upper()),
        ("meta_sha256", None),
        ("step", float(cfg.step)),
        ("n_layer", True),
        ("extra_field", 1),
    ):
        cfg.expected = {**expected, field: value}
        with pytest.raises(AxBenchAdvError, match="expected|identity"):
            native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)
    cfg.expected = expected
    cfg.layer = 1
    with pytest.raises(AxBenchAdvError, match="layer"):
        native_counter.validate_runtime_identity(cfg=cfg, source_manifest=source)


class FakeAdapter:
    def __init__(self) -> None:
        self.model = TinyDecoder()
        self.identity = {}
        self.tokenizer = None
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


def test_point_resume_has_exact_coverage_and_rejects_extra_cells(counter_run) -> None:
    cfg, adapter = counter_run
    cfg.max_cells = 3
    assert native_counter.run_native_counter(cfg)["phase"] == "generation_incomplete"
    assert len(adapter.calls) == 3
    point = Path(cfg.output_dir) / "points/diffmean/m0p25"
    assert not (point / "run_manifest.json").exists()
    cfg.phase = "generate"
    cfg.max_cells = 7
    status = native_counter.run_native_counter(cfg)
    assert status["completed_points"] == ["diffmean/m0p25"]
    assert len(adapter.calls) == 10
    assert json.loads((point / "progress.json").read_text())["complete"] is True
    _, output = load_native_output(point / "run_manifest.json")
    source_frame = pd.read_parquet(cfg.source_parquet)
    assert output[f"{NATIVE_METHOD}_steered_generation"].nunique() == 10
    assert output.native_user_prompt.tolist() == source_frame.native_user_prompt.tolist()
    records = [json.loads(path.read_text()) for path in sorted((point / "cells").glob("*.json"))]
    assert len(records) == 10
    assert {(entry["cell"]["concept_id"], entry["cell"]["input_id"]) for entry in records} == {
        (0, input_id) for input_id in range(10)
    }
    assert [entry["cell"]["seed"] for entry in records] == [
        native_counter.cell_seed(42, 0, input_id) for input_id in range(10)
    ]
    (point / "cells/unexpected.json").write_text("{}")
    with pytest.raises(AxBenchAdvError, match="unexpected cells"):
        native_counter.run_native_counter(cfg)


@pytest.fixture
def counter_run(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model_001003.pt").write_bytes(b"native model")
    (checkpoint / "meta_001003.json").write_text(json.dumps({
        "step": 1003,
        "sft_loss_mask": "completion_only",
        "model_config": {"n_layer": 1, "n_embd": 3, "sequence_len": 32},
        "user_config": {"mixture": "standard_steered", "model_backend": "nanochat"},
    }))
    tokenizer = tmp_path / "tokenizer.pkl"
    tokenizer.write_bytes(b"trained tokenizer")
    adapter = FakeAdapter()
    adapter.identity = native_producer.checkpoint_identity(checkpoint, 1003, tokenizer)
    monkeypatch.setattr(native_producer, "NativeDecoderAdapter", lambda *a, **kw: adapter)
    monkeypatch.setattr(native_counter, "NativeDecoderAdapter", lambda **kw: adapter)
    monkeypatch.setattr(native_counter, "template_prefix_length", lambda tokenizer: 2)
    monkeypatch.setattr(native_counter, "resolve_native_model_contract", lambda geometry: SimpleNamespace(
        n_layer=1, n_embd=3, sequence_len=32, intervention_layer=0, parameter_count=None,
    ))
    monkeypatch.setattr(
        native_counter,
        "_capture_class",
        lambda *a, class_name, **kw: [
            t.tensor([[4.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
            * (1 if class_name == "positive" else -1)
        ],
    )
    rows = []
    for input_id in range(10):
        prompt = f"question {input_id}"
        row: dict[str, str | int | float] = dict.fromkeys(INPUT_COLUMNS, "")
        row.update(
            dataset_name="AlpacaEval", concept_id=0, input_id=input_id,
            input_concept="cats", factor=1.0, original_prompt=prompt,
            steered_input=("<|im_start|>user\nmention cats\n\nQuestion: " + prompt
                           + "<|im_end|>\n<|im_start|>assistant\n"),
        )
        rows.append(row)
    frozen_input = tmp_path / "frozen.parquet"
    pd.DataFrame(rows).to_parquet(frozen_input, index=False)
    source = tmp_path / "source"
    source_manifest = native_producer.run_native_decoder(OmegaConf.create({
        "source_path": str(frozen_input), "source_sha256": _sha(frozen_input),
        "checkpoint_dir": str(checkpoint), "step": 1003, "tokenizer_path": str(tokenizer),
        "output_dir": str(source), "max_concepts": 1, "device": "cpu", "seed": 42,
        "max_new_tokens": 8, "temperature": 1.0, "top_k": 5, "max_cells": None,
    }))
    adapter.calls.clear()
    prepared = tmp_path / "prepared" / "generate"
    prepared.mkdir(parents=True)
    (prepared / "metadata.jsonl").write_text(json.dumps({
        "concept": "cats", "concept_genres_map": {"cats": ["text"]},
    }) + "\n")
    pd.DataFrame([
        {"input": "positive", "output": "cats", "output_concept": "cats",
         "category": "positive", "concept_genre": "text"},
        {"input": "negative", "output": "no animals", "output_concept": "EEEEE",
         "category": "negative", "concept_genre": "text"},
    ]).to_parquet(prepared / "train_data.parquet", index=False)
    expected = {key: source_manifest["checkpoint"][key] for key in (
        "step", "model_sha256", "meta_sha256", "tokenizer_sha256",
    )}
    cfg = OmegaConf.create({
        "source_manifest": str(source / "run_manifest.json"),
        "source_parquet": str(source / "steering_data.parquet"),
        "prepared_dir": str(prepared.parent), "checkpoint_dir": str(checkpoint),
        "step": 1003, "tokenizer_path": str(tokenizer), "layer": 0,
        "expected": {**expected, "n_layer": 1, "n_embd": 3},
        "output_dir": str(tmp_path / "counter"), "max_concepts": 1, "device": "cpu",
        "seed": 42, "phase": "all", "max_cells": None,
        "concept_count": None, "point_methods": None, "point_magnitudes": None,
        "methods": list(native_counter.COUNTER_METHODS),
        "magnitudes": list(native_counter.COUNTER_MAGNITUDES),
    })
    return cfg, adapter


@pytest.mark.parametrize("corrupt", [None, "model", "meta", "tokenizer", "parquet"])
def test_driver_validates_relocated_artifacts_without_original_paths(counter_run, tmp_path, corrupt):
    cfg, adapter = counter_run
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for key in ("checkpoint_dir", "tokenizer_path", "source_parquet", "source_manifest"):
        original = Path(cfg[key])
        target = relocated / original.name
        original.rename(target)
        cfg[key] = str(target)
    if corrupt is not None:
        path = {
            "model": Path(cfg.checkpoint_dir) / "model_001003.pt",
            "meta": Path(cfg.checkpoint_dir) / "meta_001003.json",
            "tokenizer": Path(cfg.tokenizer_path),
            "parquet": Path(cfg.source_parquet),
        }[corrupt]
        path.write_bytes(b"changed bytes")
        with pytest.raises((AxBenchAdvError, ValueError), match="SHA|changed"):
            native_counter.run_native_counter(cfg)
        assert not adapter.calls
        assert not Path(cfg.output_dir).exists()
        return
    status = native_counter.run_native_counter(cfg)
    assert status["phase"] == "generation_selection_complete_pending_native_judges"
    assert len(status["completed_points"]) == 8
    assert len(adapter.calls) == 80
    for manifest_path in Path(cfg.output_dir).glob("points/*/*/run_manifest.json"):
        manifest, frame = load_native_output(manifest_path)
        assert manifest["checkpoint"]["checkpoint_dir"] == cfg.checkpoint_dir
        assert len(frame) == 10


@pytest.mark.parametrize("changed", [
    "model", "tokenizer", "prepared_metadata", "prepared_training_data",
    "source_manifest", "source_parquet", "source_pair",
])
def test_driver_rejects_input_changes_before_point_publication(counter_run, monkeypatch, changed):
    cfg, adapter = counter_run
    generate = adapter.generate_user_text

    def mutate_then_generate(*args, **kwargs):
        if not adapter.calls:
            if changed in {"model", "tokenizer"}:
                path = (Path(cfg.checkpoint_dir) / "model_001003.pt"
                        if changed == "model" else Path(cfg.tokenizer_path))
                path.write_bytes(b"changed bytes")
            elif changed == "prepared_metadata":
                path = Path(cfg.prepared_dir) / "generate/metadata.jsonl"
                path.write_text(path.read_text() + "\n")
            elif changed == "prepared_training_data":
                path = Path(cfg.prepared_dir) / "generate/train_data.parquet"
                frame = pd.read_parquet(path)
                frame.loc[0, "output"] = "changed training example"
                frame.to_parquet(path, index=False)
            else:
                manifest_path = Path(cfg.source_manifest)
                manifest = json.loads(manifest_path.read_text())
                if changed == "source_manifest":
                    manifest["generation"]["seed"] += 1
                else:
                    path = Path(cfg.source_parquet)
                    frame = pd.read_parquet(path)
                    frame.loc[0, f"{NATIVE_METHOD}_steered_generation"] = "replacement control"
                    frame.to_parquet(path, index=False)
                    manifest["output"]["sha256"] = _sha(path)
                if changed != "source_parquet":
                    manifest_path.write_text(json.dumps(manifest))
                    load_native_output(manifest_path)
        return generate(*args, **kwargs)

    monkeypatch.setattr(adapter, "generate_user_text", mutate_then_generate)
    with pytest.raises(AxBenchAdvError):
        native_counter.run_native_counter(cfg)
    assert not list(Path(cfg.output_dir).glob("points/*/*/run_manifest.json"))
    assert len(adapter.calls) == 10


def test_counter_points_remain_judgeable_but_cannot_be_zero_controls(counter_run, tmp_path):
    cfg, adapter = counter_run
    native_counter.run_native_counter(cfg)
    point = Path(cfg.output_dir) / "points/diffmean/m2"
    manifest, frame = load_native_output(point / "run_manifest.json")
    assert manifest["counter_extension"]["point_identity"]["magnitude_hex"] == (2.0).hex()
    assert len(frame) == 10
    cfg.source_manifest = str(point / "run_manifest.json")
    cfg.source_parquet = str(point / "steering_data.parquet")
    cfg.output_dir = str(tmp_path / "second_counter")
    adapter.calls.clear()
    with pytest.raises(AxBenchAdvError, match="standard|counter"):
        native_counter.run_native_counter(cfg)
    assert not adapter.calls
    assert not Path(cfg.output_dir).exists()


@pytest.mark.parametrize("initial_budget", [None, 3, 10, 79])
def test_driver_reports_completion_at_exact_remaining_budget(counter_run, monkeypatch, initial_budget):
    cfg, adapter = counter_run
    if initial_budget is not None:
        cfg.max_cells = initial_budget
        status = native_counter.run_native_counter(cfg)
        assert status["phase"] == "generation_incomplete"
        assert len(adapter.calls) == initial_budget
        cfg.phase = "generate"
    cfg.max_cells = 80 - (initial_budget or 0)
    status = native_counter.run_native_counter(cfg)
    assert status["phase"] == "generation_selection_complete_pending_native_judges"
    assert len(status["completed_points"]) == 8
    assert json.loads((Path(cfg.output_dir) / "status.json").read_text()) == status
    assert len(adapter.calls) == 80
    cfg.phase = "all"
    monkeypatch.setattr(native_counter, "NativeDecoderAdapter", lambda **kw: pytest.fail(
        "completed resume must not load the model"
    ))
    native_counter.run_native_counter(cfg)
    assert len(adapter.calls) == 80


def test_driver_fit_then_generate(counter_run):
    cfg, adapter = counter_run
    cfg.phase = "fit"
    assert native_counter.run_native_counter(cfg)["phase"] == "fit_complete"
    assert not adapter.calls
    cfg.phase = "generate"
    cfg.max_cells = 80
    status = native_counter.run_native_counter(cfg)
    assert status["phase"] == "generation_selection_complete_pending_native_judges"
    assert len(adapter.calls) == 80


def test_filtered_point_reports_completion_at_exact_budget(counter_run):
    cfg, adapter = counter_run
    cfg.point_methods = ["DiffMean"]
    cfg.point_magnitudes = [0.25]
    cfg.max_cells = 10
    status = native_counter.run_native_counter(cfg)
    assert status["phase"] == "generation_selection_complete_pending_native_judges"
    assert status["completed_points"] == ["diffmean/m0p25"]
    assert len(adapter.calls) == 10
    assert len(list(Path(cfg.output_dir).glob("points/*/*/run_manifest.json"))) == 1


def test_custom_grid_binds_full_identity_and_changed_grid_resume_fails(counter_run):
    cfg, adapter = counter_run
    cfg.magnitudes = [64, 512, 4096]
    cfg.point_methods = ["DiffMean"]
    cfg.point_magnitudes = None
    cfg.max_cells = 30
    status = native_counter.run_native_counter(cfg)
    assert status["phase"] == "generation_selection_complete_pending_native_judges"
    assert status["completed_points"] == [
        "diffmean/m64.0",
        "diffmean/m512.0",
        "diffmean/m4096.0",
    ]
    assert len(adapter.calls) == 30
    identity = json.loads((Path(cfg.output_dir) / "identity.json").read_text())
    assert identity["configuration"]["magnitudes"] == [
        float(value).hex() for value in (64, 512, 4096)
    ]
    assert all(
        (Path(cfg.output_dir) / f"points/diffmean/m{value}.0/run_manifest.json").is_file()
        for value in (64, 512, 4096)
    )

    cfg.phase = "generate"
    cfg.magnitudes = [64, 1024, 4096]
    cfg.point_magnitudes = [4096]
    with pytest.raises(ImmutableArtifactError, match="identity"):
        native_counter.run_native_counter(cfg)
    assert len(adapter.calls) == 30


@pytest.mark.parametrize(
    "magnitudes",
    [
        [],
        [True],
        ["64"],
        [float("nan")],
        [float("inf")],
        [0],
        [-64],
        [64, 64],
        [512, 64],
    ],
)
def test_driver_rejects_invalid_configured_magnitude_grids_before_writing(
    counter_run, magnitudes
):
    cfg, adapter = counter_run
    cfg.magnitudes = magnitudes
    with pytest.raises(AxBenchAdvError, match="counter_magnitudes"):
        native_counter.run_native_counter(cfg)
    assert not adapter.calls
    assert not Path(cfg.output_dir).exists()


@pytest.mark.parametrize("boundary", ["identity_before", "cell_before", "cell_after"])
def test_driver_resumes_after_process_exit_during_immutable_install(counter_run, boundary):
    cfg, adapter = counter_run
    output = Path(cfg.output_dir)
    if boundary == "identity_before":
        path, payload = output / "identity.json", b"uncommitted identity"
    else:
        cfg.max_cells = 1
        native_counter.run_native_counter(cfg)
        path = output / "points/diffmean/m0p25/cells/c00000_i00000.json"
        payload = path.read_bytes()
        path.unlink()

    result = subprocess.run([sys.executable, "-c", """
import os
import sys
from pathlib import Path
from steering.immutable_io import install_bytes_once

link = os.link
def interrupt_link(source, target):
    if sys.argv[2] == "cell_after":
        link(source, target)
    os._exit(73)

os.link = interrupt_link
install_bytes_once(sys.stdin.buffer.read(), Path(sys.argv[1]), surface="crash test")
""", str(path), boundary], input=payload, timeout=15, check=False)
    assert result.returncode == 73
    assert list(path.parent.glob(f".{path.name}.tmp.*"))
    assert path.exists() == (boundary == "cell_after")
    adapter.calls.clear()
    cfg.max_cells = 80
    assert native_counter.run_native_counter(cfg)["phase"] == "generation_selection_complete_pending_native_judges"
    assert len(adapter.calls) == 80 - (boundary == "cell_after")
    assert len(list(output.glob("points/*/*/run_manifest.json"))) == 8


@pytest.mark.parametrize("field,value", [
    ("phase", "unknown"), ("methods", ["DiffMean"]), ("magnitudes", [0.0]),
    ("seed", True), ("max_cells", 0), ("max_cells", True), ("max_cells", 1.5),
    ("step", 1003.0),
    ("max_concepts", 2),
])
def test_driver_rejects_invalid_controls_before_writing(counter_run, field, value):
    cfg, adapter = counter_run
    cfg[field] = value
    with pytest.raises(AxBenchAdvError):
        native_counter.run_native_counter(cfg)
    assert not adapter.calls
    assert not Path(cfg.output_dir).exists()


@pytest.mark.parametrize("mismatch", ["checkpoint", "template", "geometry"])
def test_driver_rejects_inconsistent_loaded_model(counter_run, monkeypatch, mismatch):
    cfg, adapter = counter_run
    if mismatch == "checkpoint":
        adapter.identity["model_sha256"] = "0" * 64
    elif mismatch == "template":
        monkeypatch.setattr(native_counter, "template_prefix_length", lambda tokenizer: 3)
    else:
        adapter.model.config.sequence_len -= 1
    with pytest.raises(AxBenchAdvError, match="loading|prefix|geometry"):
        native_counter.run_native_counter(cfg)
    assert not adapter.calls
    assert not (Path(cfg.output_dir) / "fit_manifest.json").exists()


@pytest.mark.parametrize("state", [
    "no_fits", "missing_direction", "extra_direction", "invalid_direction",
    "cell_identity", "cell_key", "cell_generation", "cell_fields", "nonempty", "overlap",
])
def test_driver_rejects_invalid_persisted_state_before_new_generation(counter_run, state):
    cfg, adapter = counter_run
    output = Path(cfg.output_dir)
    cfg.phase = "generate"
    if state == "nonempty":
        output.mkdir()
        (output / "unrelated.json").write_text("{}")
    elif state == "overlap":
        cfg.output_dir = cfg.prepared_dir
    elif state != "no_fits":
        cfg.phase, cfg.max_cells = "all", 1
        native_counter.run_native_counter(cfg)
        adapter.calls.clear()
        cfg.phase = "generate"
        direction = native_counter._direction_path(output, 0, "DiffMean")
        if state == "missing_direction":
            direction.unlink()
        elif state == "extra_direction":
            direction.with_name("unexpected.safetensors").write_bytes(direction.read_bytes())
        elif state == "invalid_direction":
            cfg.phase = "all"
            direction.write_bytes(b"invalid safetensors")
        else:
            path = output / "points/diffmean/m0p25/cells/c00000_i00000.json"
            cell = json.loads(path.read_text())
            field, value = {
                "cell_identity": ("point_identity_sha256", "wrong"),
                "cell_key": ("cell", {}), "cell_generation": ("generation", None),
                "cell_fields": ("unexpected", True),
            }[state]
            cell[field] = value
            path.write_text(json.dumps(cell))
    with pytest.raises((AxBenchAdvError, ImmutableArtifactError)):
        native_counter.run_native_counter(cfg)
    assert not adapter.calls
    assert not list(output.glob("points/*/*/run_manifest.json"))


def test_partial_fit_resume_rejects_changed_direction_bytes(counter_run, monkeypatch):
    cfg, _ = counter_run
    cfg.phase = "fit"
    publish = native_counter._fit_manifest

    def interrupt(**kwargs):
        raise RuntimeError("interrupted before fit manifest")

    monkeypatch.setattr(native_counter, "_fit_manifest", interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        native_counter.run_native_counter(cfg)
    assert not (Path(cfg.output_dir) / "fit_manifest.json").exists()
    monkeypatch.setattr(native_counter, "_fit_manifest", publish)
    path = native_counter._direction_path(Path(cfg.output_dir), 0, "DiffMean")
    fitted = native_counter._load_native_direction(path)
    fitted.direction.neg_()
    native_counter._save_native_direction(path, fitted)
    with pytest.raises(ImmutableArtifactError, match="fit receipt"):
        native_counter.run_native_counter(cfg)


@pytest.mark.parametrize("phase", ["fit", "all"])
def test_driver_revalidates_before_publishing_fit_manifest(counter_run, monkeypatch, phase):
    cfg, adapter = counter_run
    cfg.phase = phase
    capture = native_counter._capture_class

    def mutate_then_capture(*args, **kwargs):
        path = Path(cfg.prepared_dir) / "generate/metadata.jsonl"
        path.write_text(path.read_text() + "\n")
        return capture(*args, **kwargs)

    monkeypatch.setattr(native_counter, "_capture_class", mutate_then_capture)
    with pytest.raises(AxBenchAdvError, match="prepared_metadata"):
        native_counter.run_native_counter(cfg)
    assert not (Path(cfg.output_dir) / "fit_manifest.json").exists()
    assert not adapter.calls


@pytest.mark.parametrize("changed", ["source_manifest", "prepared_training_data"])
def test_driver_pins_inputs_before_loading(counter_run, monkeypatch, changed):
    cfg, adapter = counter_run
    select = native_counter.select_training_pairs

    def select_then_replace(*args, **kwargs):
        training = select(*args, **kwargs)
        if changed == "source_manifest":
            path = Path(cfg.source_manifest)
            manifest = json.loads(path.read_text())
            manifest["generation"]["seed"] += 1
            path.write_text(json.dumps(manifest))
        else:
            path = Path(cfg.prepared_dir) / "generate/train_data.parquet"
            frame = pd.read_parquet(path)
            frame.loc[0, "output"] = "replacement training example"
            frame.to_parquet(path, index=False)
        return training

    monkeypatch.setattr(native_counter, "select_training_pairs", select_then_replace)
    with pytest.raises(AxBenchAdvError, match=changed):
        native_counter.run_native_counter(cfg)
    assert not (Path(cfg.output_dir) / "fit_manifest.json").exists()
    assert not adapter.calls


def test_raw_counter_fit_does_not_require_positive_activation_scale(counter_run, monkeypatch):
    cfg, adapter = counter_run
    training_path = Path(cfg.prepared_dir) / "generate/train_data.parquet"
    training = pd.read_parquet(training_path)
    pd.concat([training] * 20, ignore_index=True).to_parquet(training_path, index=False)
    monkeypatch.setattr(
        native_counter,
        "_capture_class",
        lambda *a, class_name, **kw: [
            t.tensor([[-1.0 if class_name == "positive" else -10.0, 0.0, 0.0]])
        ] * 20,
    )
    cfg.max_cells = 1
    assert native_counter.run_native_counter(cfg)["phase"] == "generation_incomplete"
    for method in native_counter.COUNTER_METHODS:
        fitted = native_counter._load_native_direction(
            native_counter._direction_path(Path(cfg.output_dir), 0, method)
        )
        t.testing.assert_close(fitted.direction, t.tensor([1.0, 0.0, 0.0]))
        assert fitted.max_activation == -1.0
    assert len(adapter.calls) == 1
