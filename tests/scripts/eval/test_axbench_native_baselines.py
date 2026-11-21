"""Native standalone baseline fit, scaling, prompt, and resume contracts."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from scripts.eval.axbench_upstream.merge_methods import INPUT_COLUMNS
from steering.eval.axbench_upstream import native_baselines as baselines
from steering.eval.axbench_upstream.data import ConceptTrainingPairs
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    fit_diffmean_direction,
    save_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import RenderedFitRow


class TinyAttention(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_proj = nn.Linear(width, width, bias=False)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return self.c_proj(x)


class TinyBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.attn = TinyAttention(width)

    def forward(self, x: t.Tensor) -> t.Tensor:
        return x + self.attn(x)


class TinyNativeModel(nn.Module):
    def __init__(self, *, width: int = 4, vocab: int = 32, layers: int = 10) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_layer=layers, n_embd=width, sequence_len=32)
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab, width),
                "h": nn.ModuleList(TinyBlock(width) for _ in range(layers)),
            }
        )
        self.lm_head = nn.Linear(width, vocab, bias=False)

    def forward(
        self,
        idx: t.Tensor,
        targets: t.Tensor | None = None,
        *,
        loss_reduction: str = "mean",
    ) -> t.Tensor:
        x = self.transformer["wte"](idx)
        for block in cast(nn.ModuleList, self.transformer["h"]):
            x = cast(TinyBlock, block)(x)
        logits = self.lm_head(x).float()
        if targets is None:
            return logits
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )


def _source() -> pd.DataFrame:
    rows = []
    for input_id in range(10):
        row: dict[str, Any] = dict.fromkeys(INPUT_COLUMNS, "")
        row.update(
            dataset_name="AlpacaEval",
            concept_id=0,
            input_id=input_id,
            input_concept="flowers",
            factor=1.0,
            original_prompt=f"Question {input_id}?",
            steered_input=f"boosted {input_id}",
            sae_id=3,
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame["raw_prompt"] = frame["original_prompt"]
    frame["native_user_prompt"] = "boosted"
    return frame


def _training_rows() -> tuple[RenderedFitRow, ...]:
    return (
        RenderedFitRow(ids=(1, 2, 3, 4), mask=(0, 0, 1, 1), prompt_token_count=2),
        RenderedFitRow(ids=(1, 5, 6, 7), mask=(0, 0, 1, 1), prompt_token_count=2),
        RenderedFitRow(ids=(1, 8, 9, 10), mask=(0, 0, 1, 1), prompt_token_count=2),
    )


def test_original_prompt_grid_and_direction_factor_scale_are_independent() -> None:
    frame = baselines._method_frame(_source(), method="DiffMean", concepts=(0,))
    assert len(frame) == 120
    assert frame["factor"].tolist()[:12] == list(baselines.METHOD_FACTORS["DiffMean"])
    assert frame["native_user_prompt"].equals(frame["original_prompt"])
    assert not frame["native_user_prompt"].str.contains("boosted").any()

    positive = [t.tensor([[3.0, 0.0, 0.0, 0.0]])]
    negative = [t.tensor([[1.0, 0.0, 0.0, 0.0]])]
    fitted = fit_diffmean_direction(
        positive, negative, intervention_layer=9, surface="native standalone oracle"
    )
    model = TinyNativeModel().eval()
    tokens = t.tensor([[1, 2]])
    baseline = model(tokens)
    factor = 0.4
    with baselines.scoped_native_direction(
        model,
        layer=9,
        direction=fitted.direction,
        coefficient=factor * fitted.max_activation,
    ):
        steered = model(tokens)
    with baselines.scoped_native_direction(
        model, layer=9, direction=fitted.direction, coefficient=0.0
    ):
        zero = model(tokens)
    assert t.equal(zero, baseline)
    assert not t.equal(steered, baseline)
    assert factor * fitted.max_activation != factor


def test_baseline_source_accepts_completed_carrier_without_parsing_boosted_prompt() -> None:
    source = pd.concat(
        [
            _source().assign(
                concept_id=concept_id,
                input_concept=f"concept {concept_id}",
                steered_input="full unrelated system and boosted-user rendering",
            )
            for concept_id in range(100)
        ],
        ignore_index=True,
    )
    validated = baselines._baseline_source_frame(source, max_concepts=100)
    assert len(validated) == 1000
    assert validated["original_prompt"].str.startswith("Question ").all()
    assert "native_user_prompt" not in validated

    source.loc[0, "raw_prompt"] = "changed"
    with pytest.raises(Exception, match="raw_prompt differs"):
        baselines._baseline_source_frame(source, max_concepts=100)


def test_train_population_is_artifact_separated_not_text_deduplicated() -> None:
    source = pd.concat(
        [
            _source().assign(concept_id=concept_id, input_concept=f"concept {concept_id}")
            for concept_id in range(100)
        ],
        ignore_index=True,
    )
    training = {
        concept_id: ConceptTrainingPairs(
            concept_id=concept_id,
            concept=f"concept {concept_id}",
            genre="text",
            positive=(("Question 0?", "TRAIN positive output"),),
            negative=(("Question 0?", "TRAIN negative output"),),
        )
        for concept_id in range(100)
    }
    baselines._assert_train_only_population(source, training)

    training.pop(99)
    with pytest.raises(Exception, match="global IDs 0..99"):
        baselines._assert_train_only_population(source, training)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.slow)])
def test_native_lora_is_deterministic_updates_adapters_freezes_base_and_has_finite_logits(
    device: str,
) -> None:
    if device == "cuda" and not t.cuda.is_available():
        pytest.skip("CUDA unavailable")
    t.manual_seed(11)
    first_model = TinyNativeModel().to(device).eval()
    second_model = TinyNativeModel().to(device).eval()
    second_model.load_state_dict(first_model.state_dict())
    base_before = {name: value.detach().clone() for name, value in first_model.state_dict().items()}
    rng_before = t.get_rng_state().clone()
    device_rng_before = t.cuda.get_rng_state() if device == "cuda" else None

    first = baselines.train_native_lora(
        first_model, _training_rows(), base_seed=42, concept_id=7, epochs=2
    )
    second = baselines.train_native_lora(
        second_model, _training_rows(), base_seed=42, concept_id=7, epochs=2
    )

    assert first.keys() == second.keys()
    assert all(t.equal(first[key], second[key]) for key in first)
    assert any(t.count_nonzero(value).item() for key, value in first.items() if key.endswith("_B"))
    assert all(
        t.equal(value, base_before[name]) for name, value in first_model.state_dict().items()
    )
    assert all(parameter.requires_grad for parameter in first_model.parameters())
    assert all(parameter.grad is None for parameter in first_model.parameters())
    assert t.equal(t.get_rng_state(), rng_before)
    if device_rng_before is not None:
        assert t.equal(t.cuda.get_rng_state(), device_rng_before)
    idx = t.tensor([[1, 2, 3]], device=device)
    base_logits = first_model(idx)
    with baselines.scoped_native_lora(first_model, first):
        adapted_logits = first_model(idx)
    assert bool(t.isfinite(adapted_logits).all())
    assert not t.equal(adapted_logits, base_logits)
    blocks = cast(nn.ModuleList, first_model.transformer["h"])
    assert all(isinstance(cast(TinyBlock, block).attn.c_proj, nn.Linear) for block in blocks)


def test_generation_cells_resume_without_repeating_and_use_original_prompt(tmp_path) -> None:
    model = TinyNativeModel().eval()
    direction = FittedDirection(
        direction=t.tensor([1.0, 0.0, 0.0, 0.0]),
        max_activation=2.5,
        metadata={"method": "diffmean"},
    )
    output = tmp_path / "run"
    artifact = baselines._artifact_path(output, "DiffMean", 0)
    save_direction(artifact, direction)
    identity = {
        "inputs": {"source": {"path": "/source.parquet", "sha256": "a" * 64}},
        "geometry": {"n_layer": 12, "n_embd": 768, "sequence_len": 1024},
        "identity": "unit-test",
    }
    baselines.install_bytes_once(
        baselines.canonical_bytes(identity),
        output / "identity.json",
        surface="unit-test native identity",
    )
    baselines._publish_fit_manifests(
        output=output, identity=identity, methods=("DiffMean",), concepts=(0,)
    )
    calls: list[tuple[str, int]] = []

    class FakeAdapter:
        def __init__(self) -> None:
            self.model = model
            self.tokenizer = None

        def generate_user_text(
            self,
            text: str,
            *,
            seed: int,
            max_new_tokens: int = 128,
            temperature: float = 1.0,
            top_k: int = 50,
        ) -> str:
            del max_new_tokens, temperature, top_k
            assert text.startswith("Question ")
            assert "boosted" not in text
            calls.append((text, seed))
            return f"generation {seed}"

    kwargs = {
        "adapter": cast(baselines.NativeBaselineAdapter, FakeAdapter()),
        "revalidate_inputs": lambda: None,
        "source": _source(),
        "output": output,
        "identity": identity,
        "checkpoint": {"model_sha256": "b" * 64},
        "method": "DiffMean",
        "concepts": (0,),
        "fits": {0: direction},
        "seed": 42,
        "generation": {"max_new_tokens": 8, "temperature": 1.0, "top_k": 5},
    }
    complete, generated = baselines._generate_method(**kwargs, max_new_cells=2)
    assert complete is False and generated == 2

    def changed_input() -> None:
        raise baselines.AxBenchAdvError("source changed")

    with pytest.raises(baselines.AxBenchAdvError, match="source changed"):
        baselines._generate_method(
            **{**kwargs, "revalidate_inputs": changed_input},
            max_new_cells=118,
        )
    assert not (output / "methods/diffmean/run_manifest.json").exists()
    complete, generated = baselines._generate_method(**kwargs, max_new_cells=None)
    assert complete is True and generated == 0
    assert len(calls) == 120
    complete, generated = baselines._generate_method(**kwargs, max_new_cells=None)
    assert complete is True and generated == 0
    assert len(calls) == 120
    manifest, frame = baselines.load_native_baseline_output(
        output / "methods/diffmean/run_manifest.json"
    )
    assert manifest["method"] == "DiffMean"
    assert len(frame) == 120
    assert frame["native_user_prompt"].equals(frame["original_prompt"])

    run_manifest_path = output / "methods/diffmean/run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    run_manifest["identity_sha256"] = "0" * 64
    run_manifest_path.write_text(json.dumps(run_manifest))
    with pytest.raises(ValueError, match="authentication"):
        baselines.load_native_baseline_output(run_manifest_path)
    run_manifest_path.write_bytes(baselines.canonical_bytes(manifest))

    record_path = next((output / "methods/diffmean/cells").glob("*.json"))
    record = json.loads(record_path.read_text())
    record["cell"]["seed"] += 1
    record_path.write_text(json.dumps(record))
    with pytest.raises(Exception, match="invalid/conflicting"):
        baselines._generate_method(**kwargs, max_new_cells=None)


@pytest.mark.parametrize("fail_forward", [False, True])
def test_native_capture_matches_block_output_and_always_removes_hook(fail_forward) -> None:
    model = TinyNativeModel().eval()
    row = _training_rows()[0]
    tokens = t.tensor([row.ids])
    with t.no_grad():
        expected = model.transformer["wte"](tokens)
        for block in cast(nn.ModuleList, model.transformer["h"]):
            expected = block(expected)
    if fail_forward:
        def fail(_module, _inputs, _output):
            raise RuntimeError("downstream failure")
        handle = model.lm_head.register_forward_hook(fail)
        with pytest.raises(RuntimeError, match="downstream failure"):
            baselines.capture_native_post_prefix_activations(model, row, layer=9, prefix_length=2)
        handle.remove()
    else:
        actual = baselines.capture_native_post_prefix_activations(model, row, layer=9, prefix_length=2)
        assert t.equal(actual, expected[0, 2:])
        assert actual.dtype == t.float32 and actual.device.type == "cpu"
        assert not actual.requires_grad
    assert not baselines._get_block(model, 9)._forward_hooks


@pytest.mark.parametrize("layer,prefix", [(-1, 2), (10, 2), (9, -1), (9, 4)])
def test_native_capture_rejects_invalid_population(layer, prefix) -> None:
    with pytest.raises(baselines.AxBenchAdvError):
        baselines.capture_native_post_prefix_activations(
            TinyNativeModel(), _training_rows()[0], layer=layer, prefix_length=prefix,
        )


def test_reviewed_native_geometry_contracts_use_round_half_up_mapping() -> None:
    assert baselines.resolve_native_model_contract(
        {"n_layer": 12, "n_embd": 768, "sequence_len": 1024}
    ) == baselines.LEGACY_MODEL_CONTRACT
    assert baselines.D14_MODEL_CONTRACT.intervention_layer == 11
    assert baselines.D14_MODEL_CONTRACT.lora_layers == (3, 5, 8, 11)
    assert baselines.D14_MODEL_CONTRACT.parameter_count == 399_115_836
    with pytest.raises(baselines.AxBenchAdvError, match="not reviewed"):
        baselines.resolve_native_model_contract(
            {"n_layer": 14, "n_embd": 768, "sequence_len": 1024}
        )


def test_d14_runtime_identity_requires_actual_step_hashes_and_layers(monkeypatch) -> None:
    checkpoint = {
        "step": 1777,
        "model_sha256": "a" * 64,
        "meta_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
    }
    geometry = {"n_layer": 14, "n_embd": 896, "sequence_len": 1024}
    monkeypatch.setattr(baselines, "checkpoint_identity", lambda *_args: checkpoint)
    monkeypatch.setattr(baselines, "_metadata_geometry", lambda *_args: geometry)
    cfg = OmegaConf.create(
        {
            "checkpoint_dir": "/checkpoint",
            "step": 1777,
            "tokenizer_path": "/tokenizer.pkl",
            "expected": {**checkpoint, "n_layer": 14, "n_embd": 896},
            "layer": 11,
            "lora_layers": [3, 5, 8, 11],
        }
    )
    assert baselines._runtime_identity(cfg) == (checkpoint, geometry)

    cfg.expected.model_sha256 = "0" * 64
    with pytest.raises(baselines.AxBenchAdvError, match="identity differs"):
        baselines._runtime_identity(cfg)
    cfg.expected.model_sha256 = "a" * 64
    cfg.layer = 9
    with pytest.raises(baselines.AxBenchAdvError, match="direction layer differs"):
        baselines._runtime_identity(cfg)
    cfg.layer = 11
    cfg.step = 1777.0
    with pytest.raises(baselines.AxBenchAdvError, match="step must be an integer"):
        baselines._runtime_identity(cfg)
    cfg.step = 1777
    cfg.lora_layers = [3, 5.0, 8, 11]
    with pytest.raises(baselines.AxBenchAdvError, match="LoRA layers differ"):
        baselines._runtime_identity(cfg)


def test_runner_rejects_checkpoint_changed_while_loading_model(tmp_path, monkeypatch) -> None:
    source = tmp_path / "source.parquet"
    source.write_bytes(b"pinned source")
    prepared = tmp_path / "prepared" / "generate"
    prepared.mkdir(parents=True)
    (prepared / "metadata.jsonl").write_bytes(b"pinned metadata")
    (prepared / "train_data.parquet").write_bytes(b"pinned training")
    checkpoint = dict.fromkeys(("model_sha256", "meta_sha256", "tokenizer_sha256"), "a" * 64)
    checkpoint["step"] = 1003
    geometry = {"n_layer": 12, "n_embd": 768, "sequence_len": 1024}
    monkeypatch.setattr(baselines, "_source_frame", lambda _cfg: (pd.DataFrame(), baselines.sha256(source)))
    monkeypatch.setattr(baselines, "_runtime_identity", lambda _cfg: (checkpoint, geometry))
    monkeypatch.setattr(baselines, "select_training_pairs", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(baselines, "_assert_train_only_population", lambda *_args: None)
    monkeypatch.setattr(
        baselines, "NativeDecoderAdapter", lambda *_args: SimpleNamespace(identity={})
    )
    cfg = OmegaConf.create({
        "phase": "fit", "methods": ["DiffMean"], "concept_lo": 0, "concept_hi": 1,
        "max_concepts": 100, "seed": 42, "max_cells": None,
        "source_path": str(source), "prepared_dir": str(prepared.parent),
        "checkpoint_dir": str(tmp_path / "checkpoint"), "step": 1003,
        "tokenizer_path": str(tmp_path / "tokenizer.pkl"),
        "output_dir": str(tmp_path / "output"), "device": "cpu",
        "max_new_tokens": 8, "temperature": 1.0, "top_k": 5,
    })
    with pytest.raises(baselines.AxBenchAdvError, match="changed while loading the model"):
        baselines.run_native_baselines(cfg)
    assert not (tmp_path / "output/methods/diffmean/fit_manifest.json").exists()
