"""Contracts for native standalone methods under AxBench-Adv pressures."""

from __future__ import annotations

import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from steering.eval.axbench_adv import native_method_judge, native_methods
from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.best_point import BestPoint
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import CallAccounting, SuffixOptimizationResult
from steering.eval.axbench_adv.native_mailbox import ArtifactMailbox
from steering.eval.axbench_adv.native_methods import NativeFitRef, NativeMethodSources
from steering.eval.axbench_upstream.encdec_baselines.directions import FittedDirection
from steering.eval.axbench_upstream.native_baselines import (
    D14_MODEL_CONTRACT,
    LEGACY_MODEL_CONTRACT,
    METHOD_FACTORS,
    NativeModelContract,
)
from steering.eval.axbench_upstream.native_contract import sha256
from steering.immutable_io import ImmutableArtifactError


class _Block(nn.Module):
    def forward(self, value: t.Tensor) -> t.Tensor:
        return value


class _ToyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(t.zeros(()))
        self.transformer = nn.ModuleDict({"h": nn.ModuleList(_Block() for _ in range(12))})
        self.config = SimpleNamespace(n_layer=12, n_embd=2, sequence_len=64)

    def intervention_output(self) -> t.Tensor:
        blocks = self.transformer["h"]
        assert isinstance(blocks, nn.ModuleList)
        return blocks[native_methods.INTERVENTION_LAYER](t.zeros(1, 1, 2))


_RUNTIME_CONTROL_CONTRACTS = (LEGACY_MODEL_CONTRACT, D14_MODEL_CONTRACT)


def _geometry(contract: NativeModelContract) -> dict[str, int]:
    return {
        "n_layer": contract.n_layer,
        "n_embd": contract.n_embd,
        "sequence_len": contract.sequence_len,
    }


def _fit(method: str, tmp_path, *, max_activation: float = 2.0) -> NativeFitRef:
    direction = FittedDirection(t.tensor([1.0, 0.0]), max_activation, {})
    return NativeFitRef(
        method, 0, tmp_path / f"{method}.safetensors", "a" * 64, direction, None, None
    )


@pytest.mark.parametrize("method", ["DiffMean", "LinearProbe"])
def test_direction_positive_calibration_and_raw_negative_counter(method, tmp_path):
    model = _ToyDecoder()
    fit = _fit(method, tmp_path)
    assert fit.direction is not None
    with native_methods.positive_method_scope(model, method=method, factor=0.4, fit=fit):
        with native_methods.scoped_native_counter(
            model,
            layer=native_methods.INTERVENTION_LAYER,
            direction=fit.direction.direction,
            magnitude=0.25,
        ):
            assert t.equal(model.intervention_output(), t.tensor([[[0.55, 0.0]]]))
    assert t.equal(model.intervention_output(), t.zeros(1, 1, 2))


def test_lora_positive_uses_authenticated_weights_with_raw_negative_counter(tmp_path, monkeypatch):
    model = _ToyDecoder()
    state = {"actual_lora_weight": t.tensor([3.0])}
    metadata = {"authenticated": "metadata"}
    fit = NativeFitRef("LoRA", 0, tmp_path / "lora.safetensors", "b" * 64, None, state, metadata)
    observed = []

    @contextmanager
    def fake_lora(candidate_model, candidate_state, *, layers):
        observed.append((candidate_model, candidate_state, tuple(layers)))
        yield

    monkeypatch.setattr(native_methods, "scoped_native_lora", fake_lora)
    with native_methods.positive_method_scope(model, method="LoRA", factor=1.0, fit=fit):
        with native_methods.scoped_native_counter(
            model,
            layer=native_methods.INTERVENTION_LAYER,
            direction=t.tensor([1.0, 0.0]),
            magnitude=0.25,
        ):
            assert t.equal(model.intervention_output(), t.tensor([[[-0.25, 0.0]]]))
    assert observed == [(model, state, native_methods.LORA_LAYERS)]
    assert t.equal(model.intervention_output(), t.zeros(1, 1, 2))


def _judged_entries(method: str) -> tuple[dict, ...]:
    factors = list(METHOD_FACTORS[method])
    preferred = {"DiffMean": 0.2, "LinearProbe": 0.4, "LoRA": 1.0}[method]
    entries = []
    for concept_id in range(100):
        ratings = [1.0 if factor == preferred else 0.1 for factor in factors]
        if concept_id >= 50 and 6.0 in factors:
            ratings = [2.0 if factor == 6.0 else 0.1 for factor in factors]
        entries.append(
            {
                "concept_id": concept_id,
                "results": {
                    "LMJudgeEvaluator": {
                        method: {
                            "factor": factors,
                            "relevance_concept_ratings": ratings,
                            "fluency_ratings": ratings,
                        }
                    }
                },
            }
        )
    return tuple(entries)


def test_factor_selection_uses_exact_adv50_not_later_concepts():
    points = native_methods.select_adv50_best_points(
        {method: _judged_entries(method) for method in native_methods.METHODS}
    )
    assert {method: point.factor for method, point in points.items()} == {
        "DiffMean": 0.2,
        "LinearProbe": 0.4,
        "LoRA": 1.0,
    }


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "concept100"])
def test_factor_selection_rejects_population_mutations(mutation):
    entries = {method: list(_judged_entries(method)) for method in native_methods.METHODS}
    if mutation == "missing":
        entries["DiffMean"].pop()
    elif mutation == "duplicate":
        entries["DiffMean"][-1] = entries["DiffMean"][-2]
    else:
        entries["DiffMean"][-1] = {**entries["DiffMean"][-1], "concept_id": 100}
    with pytest.raises(AxBenchAdvError, match="population|duplicated"):
        native_methods.select_adv50_best_points(entries)


def test_fit_source_identity_mismatch_fails_closed():
    contract = {
        "format": native_methods.BASELINE_FORMAT,
        "identity_sha256": "a" * 64,
        "method": "DiffMean",
        "concept_id": 0,
        "concept": "concept zero",
        "training_split": "TRAIN",
        "capture_site": native_methods.CAPTURE_SITE,
        "intervention_layer": native_methods.INTERVENTION_LAYER,
        "n_layer": 12,
        "n_embd": 768,
    }
    fitted = FittedDirection(
        direction=t.nn.functional.normalize(t.ones(768), dim=0),
        max_activation=2.0,
        metadata={"method": "diffmean", "native_contract": contract},
    )
    native_methods._validate_direction_fit(
        fitted,
        method="DiffMean",
        concept_id=0,
        concept="concept zero",
        identity_sha256="a" * 64,
    )
    with pytest.raises(AxBenchAdvError, match="fit contract differs"):
        native_methods._validate_direction_fit(
            fitted,
            method="DiffMean",
            concept_id=0,
            concept="concept zero",
            identity_sha256="b" * 64,
        )


def _standalone_identity(
    prefix: str,
    *,
    adapted_runtime: bool,
    n_layer: int = 12,
    n_embd: int = 768,
    step: int = 1003,
) -> dict:
    contract = native_methods.resolve_native_model_contract(
        {"n_layer": n_layer, "n_embd": n_embd, "sequence_len": 1024}
    )
    configuration = {
        "max_concepts": 100,
        "direction_factors": list(native_methods.FULL_FACTORS),
        "lora_factors": [1.0],
        "layer": contract.intervention_layer,
        "capture_site": native_methods.CAPTURE_SITE,
        "capture_prefix_length": native_methods.CAPTURE_PREFIX_LENGTH,
        "lora_layers": list(contract.lora_layers),
        "seed": 42,
        "prompt": "original_prompt only",
        "training_split": "TRAIN only",
        "direction_scale": "factor * fit-row max_activation * unit direction",
        "methods": list(native_methods.METHODS),
        "concept_ids": [0],
        "generation": {"max_new_tokens": 128, "temperature": 1.0, "top_k": 50},
    }
    if adapted_runtime:
        configuration["renderer"] = "RustBPE.render_conversation(max_tokens=None) + assistant_start"
    runtime = {
        "torch_version": "2.13",
        "matmul_precision": "highest",
        "compute_dtype": "torch.bfloat16",
        "attention_backend": "sdpa" if adapted_runtime else None,
        "cuda_version": "13.0",
        "device": "cuda:0",
    }
    if adapted_runtime:
        runtime["full_prefix_attention_backend"] = "native_fa4"
    return {
        "schema_version": 1,
        "format": native_methods.BASELINE_FORMAT,
        "configuration": configuration,
        "inputs": {
            "checkpoint": {
                "checkpoint_dir": f"{prefix}/checkpoint",
                "tokenizer_path": f"{prefix}/tokenizer.pkl",
                "step": step,
                "model_sha256": "a" * 64,
                "meta_sha256": "b" * 64,
                "tokenizer_sha256": "c" * 64,
            },
            "source": {"path": f"{prefix}/source.parquet", "sha256": "d" * 64},
            "prepared_metadata": {"path": f"{prefix}/meta.jsonl", "sha256": "e" * 64},
            "prepared_training_data": {
                "path": f"{prefix}/train.parquet",
                "sha256": "f" * 64,
            },
        },
        "geometry": {"n_layer": n_layer, "n_embd": n_embd, "sequence_len": 1024},
        "runtime": runtime,
    }


def test_source_identity_allows_relocation_but_not_hash_mutation():
    local = native_methods._common_identity(_standalone_identity("/local", adapted_runtime=False))
    relocated = _standalone_identity("/remote", adapted_runtime=True)
    assert native_methods._common_identity(relocated) == local
    relocated["inputs"]["prepared_training_data"]["sha256"] = "0" * 64
    assert native_methods._common_identity(relocated) != local


def test_d14_source_identity_is_distinct_and_rejects_legacy_fitted_direction() -> None:
    d12 = native_methods._common_identity(
        _standalone_identity("/d12", adapted_runtime=True)
    )
    d14_identity = _standalone_identity(
        "/d14", adapted_runtime=True, n_layer=14, n_embd=896, step=1777
    )
    d14 = native_methods._common_identity(d14_identity)
    assert d14 != d12
    assert d14["inputs"]["checkpoint"]["step"] == 1777
    assert d14["geometry"] == {"n_layer": 14, "n_embd": 896, "sequence_len": 1024}

    contract = {
        "format": native_methods.BASELINE_FORMAT,
        "identity_sha256": "d" * 64,
        "method": "DiffMean",
        "concept_id": 0,
        "concept": "concept zero",
        "training_split": "TRAIN",
        "capture_site": native_methods.CAPTURE_SITE,
        "intervention_layer": 11,
        "n_layer": 14,
        "n_embd": 896,
    }
    old_width = FittedDirection(
        direction=t.nn.functional.normalize(t.ones(768), dim=0),
        max_activation=2.0,
        metadata={"method": "diffmean", "native_contract": contract},
    )
    with pytest.raises(AxBenchAdvError, match="fit contract differs"):
        native_methods._validate_direction_fit(
            old_width,
            method="DiffMean",
            concept_id=0,
            concept="concept zero",
            identity_sha256="d" * 64,
            model_contract=native_methods.resolve_native_model_contract(d14["geometry"]),
        )


def test_d14_identity_rejects_legacy_layers_even_with_new_hashes() -> None:
    identity = _standalone_identity(
        "/d14", adapted_runtime=True, n_layer=14, n_embd=896, step=1777
    )
    identity["configuration"]["layer"] = native_methods.INTERVENTION_LAYER
    identity["configuration"]["lora_layers"] = list(native_methods.LORA_LAYERS)
    with pytest.raises(AxBenchAdvError, match="semantic identity differs"):
        native_methods._common_identity(identity)


def test_archive_roots_are_bound_to_union_parent_bytes(tmp_path):
    root = tmp_path / "job" / "native"
    native_methods.write_json(root / "status.json", {"phase": "complete"})
    archive = root.parent / "judged-archive.tar.gz"
    with tarfile.open(archive, mode="w:gz") as bundle:
        bundle.add(root, arcname="native")
    remote = "campaign/native-c000-001-dmlp/judged.tar.gz"
    union = {"parents": [{"remote": remote, "sha256": sha256(archive)}]}
    native_methods._validate_archive_parents(union, [root])

    mutated = {"parents": [{"remote": remote, "sha256": "0" * 64}]}
    with pytest.raises(AxBenchAdvError, match="differ from union parent"):
        native_methods._validate_archive_parents(mutated, [root])
    native_methods.write_json(root / "status.json", {"phase": "mutated"})
    with pytest.raises(AxBenchAdvError, match="extracted file differs"):
        native_methods._validate_archive_parents(union, [root])


def test_aggregate_score_comparison_allows_only_csv_roundoff():
    expected = {
        "method": "DiffMean",
        "factor": 0.2,
        "n_concepts": 100,
        **dict.fromkeys(native_methods.SUMMARY, 0.33999999999999997),
    }
    rounded = {**expected, "lm_judge_rating": 0.34}
    assert native_methods._score_row_matches(rounded, expected)
    mutated = {**rounded, "lm_judge_rating": 0.34000001}
    assert not native_methods._score_row_matches(mutated, expected)


def test_zero_control_reuses_exact_standard_generation_and_judgment(tmp_path):
    generation = "authenticated standard output"
    frame = pd.DataFrame(
        [
            {
                "concept_id": 0,
                "input_id": 0,
                "factor": 0.2,
                "DiffMean_steered_generation": generation,
            }
        ]
    )
    entry = {
        "concept_id": 0,
        "results": {
            "LMJudgeEvaluator": {
                "DiffMean": {
                    "factor": [0.2],
                    "raw_relevance_concept_ratings": [1.25],
                    "raw_fluency_ratings": [1.75],
                    "relevance_concept_completions": ["concept rationale"],
                    "fluency_completions": ["fluency rationale"],
                }
            }
        },
    }
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={"DiffMean": frame},
        judged_entries={"DiffMean": (entry,)},
        best_points={"DiffMean": BestPoint("DiffMean", 0.2, 1.25, 1.75, 1.458333)},
        fits={},
        checkpoint={},
        geometry={},
        generation={},
        identity={},
    )
    raw = native_methods._selected_raw(sources, method="DiffMean", concept_id=0, input_id=0)
    assert (raw.generation, raw.concept_rating, raw.fluency_rating) == (generation, 1.25, 1.75)
    assert raw.generation_calls == raw.paid_judge_cache_misses == 0


def test_reused_zero_controls_replay_every_prompt(tmp_path, monkeypatch):
    prompts = ("first", "second")
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(
            {"concept_id": [0, 0], "input_id": [0, 1], "original_prompt": prompts}
        ),
        generation_frames={},
        judged_entries={},
        best_points={"DiffMean": BestPoint("DiffMean", 0.2, 1.0, 1.0, 1.0)},
        fits={("DiffMean", 0): _fit("DiffMean", tmp_path)},
        checkpoint={},
        geometry={"n_layer": 12, "n_embd": 768, "sequence_len": 1024},
        generation={"seed": 42, "max_new_tokens": 1},
        identity={},
    )
    calls = []

    @contextmanager
    def no_hook(*_args, **_kwargs):
        yield

    class Adapter:
        model = _ToyDecoder()

        def prompt_tokens(self, text):
            return list(text.encode())

        def generate_user_text(self, text, *, seed, **_kwargs):
            calls.append((text, seed))
            return text

    monkeypatch.setattr(native_methods, "positive_method_scope", no_hook)
    controls = {
        ("DiffMean", 0, input_id): native_methods.RawEvaluation(prompt, 1.0, 1.0, "", 0)
        for input_id, prompt in enumerate(prompts)
    }
    adapter = cast(native_methods.NativeMethodAdapter, Adapter())
    native_methods._verify_zero_controls(sources=sources, adapter=adapter, controls=controls)
    assert calls == [
        (prompt, native_methods.derive_cell_seed(42, "DiffMean", 0, input_id, 0.2))
        for input_id, prompt in enumerate(prompts)
    ]
    controls[("DiffMean", 0, 1)] = native_methods.RawEvaluation("drift", 1.0, 1.0, "", 0)
    with pytest.raises(AxBenchAdvError, match="replay differs: DiffMean c0 i1"):
        native_methods._verify_zero_controls(sources=sources, adapter=adapter, controls=controls)


def test_retained_gepa_suffix_must_fit_all_prompts_before_cell_write(tmp_path, monkeypatch):
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(
            {
                "concept_id": [0] * 10,
                "input_id": list(range(10)),
                "original_prompt": ["short"] * 9 + ["long" * 12],
                "input_concept": ["target"] * 10,
            }
        ),
        generation_frames={},
        judged_entries={},
        best_points={"DiffMean": BestPoint("DiffMean", 0.2, 1.0, 1.0, 1.0)},
        fits={("DiffMean", 0): _fit("DiffMean", tmp_path)},
        checkpoint={},
        geometry={"n_layer": 12, "n_embd": 768, "sequence_len": 1024},
        generation={"seed": 42, "max_new_tokens": 1},
        identity={},
    )

    class Adapter:
        model = _ToyDecoder()

        def prompt_tokens(self, text):
            return list(text.encode())

    monkeypatch.setattr(
        native_methods,
        "optimize_suffix",
        lambda **_kwargs: SuffixOptimizationResult(
            suffix="x" * 16,
            best_fitness=0.0,
            evaluations=(),
            accounting=CallAccounting(0, 0, 0, 0, 0),
        ),
    )
    with pytest.raises(AxBenchAdvError, match="retained native method GEPA suffix"):
        native_methods._gepa_cell(
            sources=sources,
            adapter=cast(native_methods.NativeMethodAdapter, Adapter()),
            mailbox=cast(ArtifactMailbox, None),
            reflection=cast(native_methods.MailboxReflection, None),
            output=tmp_path,
            identity={},
            method="DiffMean",
            concept_id=0,
            proposal_count=1,
            reflection_model="openai/gpt-5.6-luna",
            gepa_seed=42,
            seed_suffix="avoid",
            reflection_minibatch_size=3,
        )
    assert not (tmp_path / "gepa" / "DiffMean" / "c0" / "p1.json").exists()


@pytest.mark.parametrize(
    "model_contract", _RUNTIME_CONTROL_CONTRACTS, ids=("d12", "d14")
)
def test_archived_default_pilot_keeps_mismatch_report_and_fails(
    tmp_path, monkeypatch, model_contract
):
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(
            {"concept_id": [0], "input_id": [0], "original_prompt": ["prompt"]}
        ),
        generation_frames={},
        judged_entries={},
        best_points={
            method: BestPoint(method, 1.0, 1.0, 1.0, 1.0)
            for method in native_methods.METHODS
        },
        fits={(method, 0): _fit(method, tmp_path) for method in native_methods.METHODS},
        checkpoint={},
        geometry=_geometry(model_contract),
        generation={"seed": 42, "max_new_tokens": 1},
        identity={},
    )

    @contextmanager
    def no_hook(*_args, **_kwargs):
        yield

    class Adapter:
        model = _ToyDecoder()

        def generate_user_text(self, *_args, **_kwargs):
            return "different"

    monkeypatch.setattr(native_methods, "positive_method_scope", no_hook)
    monkeypatch.setattr(native_methods, "scoped_native_counter", no_hook)
    monkeypatch.setattr(
        native_methods,
        "_selected_raw",
        lambda *_args, **_kwargs: native_methods.RawEvaluation("control", 1.0, 1.0, "", 0),
    )
    with pytest.raises(AxBenchAdvError, match="pilot zero-control replay differs"):
        native_methods._pilot(
            sources=sources,
            adapter=cast(native_methods.NativeMethodAdapter, Adapter()),
            output=tmp_path,
            concept_id=0,
            pressures=(native_methods.Pressure.DIFFMEAN_COUNTER,),
            magnitudes=(64.0, 512.0, 4096.0),
        )
    report = native_methods.read_json(tmp_path / "pilot.json")
    assert "zero_control_source" not in report
    assert [
        row["equal_to_authenticated_control"]
        for row in report["records"]
        if row["pressure"] == "zero_control_replay"
    ] == [False] * len(native_methods.METHODS)
    assert {
        row["magnitude"]
        for row in report["records"]
        if row["pressure"] != "zero_control_replay"
    } == {64.0}
    assert {row["pressure"] for row in report["records"]} == {
        "zero_control_replay",
        "diffmean_counter",
    }


def test_run_identity_resumes_only_byte_identical_source(tmp_path):
    identity = {"format": native_methods.FORMAT, "source": {"sha256": "a" * 64}}
    native_methods._install_run_identity(tmp_path, identity)
    before = (tmp_path / "identity.json").read_bytes()
    native_methods._install_run_identity(tmp_path, identity)
    assert (tmp_path / "identity.json").read_bytes() == before
    with pytest.raises(ImmutableArtifactError):
        native_methods._install_run_identity(
            tmp_path, {"format": native_methods.FORMAT, "source": {"sha256": "b" * 64}}
        )


@pytest.mark.parametrize(
    "model_contract", _RUNTIME_CONTROL_CONTRACTS, ids=("d12", "d14")
)
def test_runtime_and_counter_cell_resume_are_identity_bound(tmp_path, model_contract):
    point = BestPoint("DiffMean", 1.0, 1.0, 1.0, 1.0)
    fit = _fit("DiffMean", tmp_path)
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={},
        judged_entries={},
        identity={"source": "authenticated", "geometry": _geometry(model_contract)},
        best_points=dict.fromkeys(native_methods.METHODS, point),
        generation={"seed": 42, "max_new_tokens": 128, "temperature": 1.0, "top_k": 50},
        fits={("DiffMean", 0): fit},
        checkpoint={},
        geometry=_geometry(model_contract),
    )
    cfg = OmegaConf.create(
        {
            "gepa_seed": 42,
            "seed_suffix": "avoid",
            "reflection_model": "openai/gpt-5.6-luna",
            "reflection_minibatch_size": 3,
        }
    )
    arguments = {
        "cfg": cfg,
        "sources": sources,
        "concept_ids": [0],
        "pressures": native_methods.PRESSURES,
        "magnitudes": native_methods.COUNTER_MAGNITUDES,
        "proposals": native_methods.PROPOSAL_COUNTS,
    }
    first = native_methods._runtime_identity(
        **arguments, runtime={"device": "cuda:0", "attention_backend": "sdpa"}
    )
    assert "zero_control_source" not in first
    runtime_cfg = cast(DictConfig, OmegaConf.merge(cfg, {"zero_control_source": "runtime"}))
    runtime_control = native_methods._runtime_identity(
        cfg=runtime_cfg,
        sources=sources,
        concept_ids=[0],
        pressures=native_methods.PRESSURES,
        magnitudes=native_methods.COUNTER_MAGNITUDES,
        proposals=native_methods.PROPOSAL_COUNTS,
        runtime={"device": "cuda:0", "attention_backend": "sdpa"},
    )
    assert runtime_control["zero_control_source"] == "runtime"
    assert runtime_control != first
    second = native_methods._runtime_identity(
        **arguments, runtime={"device": "cpu", "attention_backend": "sdpa"}
    )
    assert first != second
    changed_grid = native_methods._runtime_identity(
        cfg=cfg,
        sources=sources,
        concept_ids=[0],
        pressures=native_methods.PRESSURES,
        magnitudes=(64.0, 512.0, 4096.0),
        proposals=native_methods.PROPOSAL_COUNTS,
        runtime={"device": "cuda:0", "attention_backend": "sdpa"},
    )
    assert changed_grid["counter_magnitudes"] == [64.0, 512.0, 4096.0]
    assert changed_grid != first
    resume = tmp_path / "changed-grid-resume"
    native_methods._install_run_identity(resume, first)
    with pytest.raises(ImmutableArtifactError, match="identity"):
        native_methods._install_run_identity(resume, changed_grid)
    with pytest.raises(ImmutableArtifactError, match="identity"):
        native_methods._install_run_identity(resume, runtime_control)

    cell = tmp_path / "counter" / "diffmean_counter" / "DiffMean" / "c0" / "m0.25" / "point.json"
    native_methods.write_json(
        cell,
        {
            "method": "DiffMean",
            "concept_id": 0,
            "pressure": "diffmean_counter",
            "magnitude_hex": (0.25).hex(),
            "factor_hex": (1.0).hex(),
            "run_identity_sha256": "wrong",
            "positive_fit_sha256": fit.sha256,
            "counter_fit_sha256": fit.sha256,
            "mean_fitness": 1.0,
            "evaluations": [],
        },
    )
    with pytest.raises(AxBenchAdvError, match="counter cell identity"):
        native_methods._counter_cell(
            sources=sources,
            adapter=cast(native_methods.NativeMethodAdapter, None),
            mailbox=cast(ArtifactMailbox, None),
            output=tmp_path,
            identity=first,
            method="DiffMean",
            concept_id=0,
            pressure=native_methods.Pressure.DIFFMEAN_COUNTER,
            magnitude=0.25,
            reflection_model="openai/gpt-5.6-luna",
        )

    raw = {
        "generation": "retained generation",
        "concept_rating": 1.0,
        "fluency_rating": 2.0,
        "judge_rationale": '{"concept":"ok","fluency":"ok"}',
        "paid_judge_cache_misses": 0,
        "generation_calls": 0,
    }
    document = native_methods.read_json(cell)
    document["run_identity_sha256"] = native_methods.digest_json(first)
    document["evaluations"] = [
        {"input_id": input_id, "raw": raw} for input_id in range(10)
    ]
    native_methods.write_json(cell, document)
    with pytest.raises(AxBenchAdvError, match="retained fitness"):
        native_methods._counter_cell(
            sources=sources,
            adapter=cast(native_methods.NativeMethodAdapter, None),
            mailbox=cast(ArtifactMailbox, None),
            output=tmp_path,
            identity=first,
            method="DiffMean",
            concept_id=0,
            pressure=native_methods.Pressure.DIFFMEAN_COUNTER,
            magnitude=0.25,
            reflection_model="openai/gpt-5.6-luna",
        )


def test_d14_provider_request_is_bound_to_run_identity(tmp_path) -> None:
    fit = _fit("DiffMean", tmp_path)
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(
            [
                {
                    "concept_id": 0,
                    "input_id": 0,
                    "input_concept": "concept zero",
                    "original_prompt": "Question?",
                }
            ]
        ),
        generation_frames={},
        judged_entries={},
        best_points={"DiffMean": BestPoint("DiffMean", 0.4, 1.0, 1.0, 1.0)},
        fits={("DiffMean", 0): fit},
        checkpoint={},
        geometry={"n_layer": 14, "n_embd": 896, "sequence_len": 1024},
        generation={"seed": 42, "max_new_tokens": 8, "temperature": 1.0, "top_k": 5},
        identity={},
    )

    class Adapter:
        model = _ToyDecoder()

        @staticmethod
        def prompt_tokens(_text):
            return [1]

        @staticmethod
        def generate_user_text(_text, **_kwargs):
            return "D14 generation"

    class Mailbox:
        payload = None

        def request(self, operation, payload):
            assert operation == "judge"
            self.payload = payload
            return {
                "generation": "D14 generation",
                "concept_rating": 1.0,
                "fluency_rating": 2.0,
                "judge_rationale": '{"concept":"ok","fluency":"ok"}',
                "paid_judge_cache_misses": 0,
            }

    identity = {"format": native_methods.FORMAT, "source_identity": "D14"}
    mailbox = Mailbox()
    native_methods._evaluate_generated(
        sources=sources,
        adapter=cast(native_methods.NativeMethodAdapter, Adapter()),
        mailbox=cast(ArtifactMailbox, mailbox),
        output=tmp_path / "output",
        identity=identity,
        method="DiffMean",
        concept_id=0,
        input_id=0,
        suffix="",
        pressure=native_methods.Pressure.GEPA,
        magnitude=None,
        reflection_model="openai/gpt-5.6-luna",
    )
    assert mailbox.payload is not None
    assert mailbox.payload["run_identity_sha256"] == native_methods.digest_json(identity)
    saved = next((tmp_path / "output" / "generations").glob("*.json"))
    record = native_methods.read_json(saved)
    record["seed"] += 1
    native_methods.write_json(saved, record)
    with pytest.raises(AxBenchAdvError, match="saved generation identity"):
        native_methods._evaluate_generated(
            sources=sources,
            adapter=cast(native_methods.NativeMethodAdapter, Adapter()),
            mailbox=cast(ArtifactMailbox, mailbox),
            output=tmp_path / "output",
            identity=identity,
            method="DiffMean",
            concept_id=0,
            input_id=0,
            suffix="",
            pressure=native_methods.Pressure.GEPA,
            magnitude=None,
            reflection_model="openai/gpt-5.6-luna",
        )


@pytest.mark.parametrize(
    "model_contract", _RUNTIME_CONTROL_CONTRACTS, ids=("d12", "d14")
)
def test_runtime_shared_zero_generation_and_judgment_resume_durably(
    tmp_path, monkeypatch, model_contract
):
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(
            [
                {
                    "concept_id": 0,
                    "input_id": 0,
                    "input_concept": "concept zero",
                    "original_prompt": "Question?",
                }
            ]
        ),
        generation_frames={},
        judged_entries={},
        best_points={"DiffMean": BestPoint("DiffMean", 0.4, 1.0, 1.0, 1.0)},
        fits={("DiffMean", 0): _fit("DiffMean", tmp_path)},
        checkpoint={},
        geometry=_geometry(model_contract),
        generation={"seed": 42, "max_new_tokens": 8, "temperature": 1.0, "top_k": 50},
        identity={},
    )
    generated = []

    @contextmanager
    def no_hook(*_args, **_kwargs):
        yield

    class Adapter:
        model = _ToyDecoder()

        @staticmethod
        def prompt_tokens(_text):
            return [1]

        @staticmethod
        def generate_user_text(text, **kwargs):
            generated.append((text, kwargs))
            return "fresh runtime control"

    class Mailbox:
        calls = []

        def request(self, operation, payload):
            self.calls.append((operation, payload))
            return {
                "generation": "fresh runtime control",
                "concept_rating": 0.25,
                "fluency_rating": 1.75,
                "judge_rationale": '{"concept":"fresh concept","fluency":"fresh fluency"}',
                "paid_judge_cache_misses": 1,
            }

    monkeypatch.setattr(native_methods, "positive_method_scope", no_hook)
    identity = {
        "format": native_methods.FORMAT,
        "source_identity": f"D{model_contract.n_layer}",
        "zero_control_source": "runtime",
    }
    mailbox = Mailbox()
    kwargs = {
        "sources": sources,
        "adapter": cast(native_methods.NativeMethodAdapter, Adapter()),
        "mailbox": cast(ArtifactMailbox, mailbox),
        "output": tmp_path / "output",
        "identity": identity,
        "method": "DiffMean",
        "concept_id": 0,
        "input_id": 0,
        "suffix": "",
        "pressure": native_methods.Pressure.GEPA,
        "magnitude": None,
        "reflection_model": "openai/gpt-5.6-luna",
        "control_key": "shared_zero",
        "preserve_cached_accounting": True,
    }
    first = native_methods._evaluate_generated(**kwargs)
    second = native_methods._evaluate_generated(**kwargs)
    assert first == second
    assert (first.generation, first.concept_rating, first.fluency_rating) == (
        "fresh runtime control",
        0.25,
        1.75,
    )
    assert first.generation_calls == first.paid_judge_cache_misses == 1
    assert len(generated) == len(mailbox.calls) == 1
    record = native_methods.read_json(next((tmp_path / "output/generations").glob("*.json")))
    assert record["key"]["control"] == "shared_zero"
    assert "pressure" not in record["key"]
    assert record["generation"] == "fresh runtime control"
    assert record["raw"] == {
        "generation": "fresh runtime control",
        "concept_rating": 0.25,
        "fluency_rating": 1.75,
        "judge_rationale": '{"concept":"fresh concept","fluency":"fresh fluency"}',
        "paid_judge_cache_misses": 1,
        "generation_calls": 1,
    }


def test_unknown_zero_control_source_fails_before_source_or_runtime_work(monkeypatch):
    called = Mock(side_effect=AssertionError("source validation must not run"))
    monkeypatch.setattr(native_methods, "validate_native_method_sources", called)
    with pytest.raises(AxBenchAdvError, match="zero_control_source"):
        native_methods.run_native_method_adv(
            OmegaConf.create({"zero_control_source": "copied_archive_scores"})
        )
    called.assert_not_called()


def test_runtime_zero_control_rejects_unsupported_geometry_before_adapter_construction(
    tmp_path, monkeypatch
):
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={},
        judged_entries={},
        best_points={},
        fits={},
        checkpoint={},
        geometry={"n_layer": 13, "n_embd": 768, "sequence_len": 1024},
        generation={},
        identity={},
    )
    adapter = Mock(side_effect=AssertionError("unsupported geometry must not load an adapter"))
    monkeypatch.setattr(native_methods, "validate_native_method_sources", lambda **_k: sources)
    monkeypatch.setattr(native_methods, "_adapter", adapter)
    cfg = OmegaConf.create(
        {
            "zero_control_source": "runtime",
            "phase": "run",
            "methods": list(native_methods.METHODS),
            "pressures": ["diffmean_counter"],
            "counter_magnitudes": [64.0],
            "proposal_counts": list(native_methods.PROPOSAL_COUNTS),
            "reflection_model": "openai/gpt-5.6-luna",
            "final_root": str(tmp_path / "final"),
            "archive_roots": [str(tmp_path / "archive")],
            "checkpoint_dir": str(tmp_path / "checkpoint"),
            "tokenizer_path": str(tmp_path / "tokenizer.pkl"),
            "output_dir": str(tmp_path / "output"),
            "step": 1001,
        }
    )
    with pytest.raises(AxBenchAdvError, match="geometry is not reviewed"):
        native_methods.run_native_method_adv(cfg)
    adapter.assert_not_called()


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("step", 1004),
        ("model_sha256", "0" * 64),
        ("meta_sha256", "1" * 64),
        ("tokenizer_sha256", "2" * 64),
    ],
)
def test_loaded_adapter_must_match_authenticated_checkpoint(
    tmp_path, monkeypatch, field, bad_value
):
    expected = {
        "step": 1003,
        "model_sha256": "a" * 64,
        "meta_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
    }

    class FakeAdapter:
        def __init__(self, **_kwargs):
            self.identity = {**expected, field: bad_value}
            self.model = _ToyDecoder()

    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={},
        judged_entries={},
        best_points={},
        fits={},
        checkpoint=expected,
        geometry={"n_layer": 12, "n_embd": 2, "sequence_len": 64},
        generation={},
        identity={},
    )
    monkeypatch.setattr(native_methods, "NativeDecoderAdapter", FakeAdapter)
    with pytest.raises(AxBenchAdvError, match=f"checkpoint differs: {field}"):
        native_methods._adapter(
            sources=sources,
            checkpoint_dir=tmp_path / "checkpoint",
            tokenizer_path=tmp_path / "tokenizer.pkl",
            device="cpu",
        )


def test_loaded_d14_adapter_requires_exact_parameter_count(tmp_path, monkeypatch):
    expected = {
        "step": 1777,
        "model_sha256": "a" * 64,
        "meta_sha256": "b" * 64,
        "tokenizer_sha256": "c" * 64,
    }

    class TinyD14(nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = nn.Parameter(t.zeros(()))
            self.config = SimpleNamespace(n_layer=14, n_embd=896, sequence_len=1024)

    class FakeAdapter:
        def __init__(self, **_kwargs):
            self.identity = expected
            self.model = TinyD14()

    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={},
        judged_entries={},
        best_points={},
        fits={},
        checkpoint=expected,
        geometry={"n_layer": 14, "n_embd": 896, "sequence_len": 1024},
        generation={},
        identity={},
    )
    monkeypatch.setattr(native_methods, "NativeDecoderAdapter", FakeAdapter)
    with pytest.raises(AxBenchAdvError, match="parameter count differs"):
        native_methods._adapter(
            sources=sources,
            checkpoint_dir=tmp_path / "checkpoint",
            tokenizer_path=tmp_path / "tokenizer.pkl",
            device="cpu",
        )


def test_published_point_reaches_final_judge_and_detects_changed_emission(tmp_path):
    source = pd.DataFrame(
        [
            {
                "dataset_name": "fixture",
                "concept_id": 0,
                "input_concept": "target",
                "input_id": input_id,
                "factor": 1.0,
                "original_prompt": f"prompt {input_id}",
            }
            for input_id in range(10)
        ]
    )
    factors = {"DiffMean": 0.4, "LinearProbe": 1.5, "LoRA": 1.0}
    emissions = {
        (method, 0, input_id): native_methods.RawEvaluation(
            f"{method} output {input_id}",
            1.0,
            1.5,
            '{"concept":"concept rationale","fluency":"fluency rationale"}',
            0,
            0,
        )
        for method in native_methods.METHODS
        for input_id in range(10)
    }
    identity = {"format": native_methods.FORMAT, "concept_ids": [0], "best_factors": factors}
    relative = Path("points/p0")
    coordinate = {"pressure": "shared_zero", "proposal_count": 0, "magnitude": 0.0}
    native_methods._publish_point(
        sources=cast(NativeMethodSources, SimpleNamespace(source_frame=source)),
        output=tmp_path,
        relative=relative,
        identity_sha256=native_methods.digest_json(identity),
        coordinate=coordinate,
        best_factors=factors,
        concept_ids=[0],
        emissions=emissions,
    )
    table, loaded = native_method_judge._validate_point(
        run_dir=tmp_path, relative=relative, coordinate=coordinate, identity=identity
    )
    assert table.num_rows == 30
    assert loaded == emissions

    point = tmp_path / relative
    changed = native_methods.read_json(point / "emissions.json")
    changed["emissions"][0]["raw"]["generation"] = "changed"
    native_methods.write_json(point / "emissions.json", changed)
    point_identity = native_methods.read_json(point / "point.json")
    point_identity["emissions_sha256"] = sha256(point / "emissions.json")
    native_methods.write_json(point / "point.json", point_identity)
    with pytest.raises(AxBenchAdvError, match="point input differs from emissions"):
        native_method_judge._validate_point(
            run_dir=tmp_path, relative=relative, coordinate=coordinate, identity=identity
        )


def _write_judged_output(output, *, table, best_factors, input_logical_sha256):
    output.mkdir(parents=True)
    accounting_path = output / "judge_call_accounting.json"
    steering_path = output / "steering.jsonl"
    parquet_path = output / "steering_data.parquet"
    native_methods.write_json(
        accounting_path,
        {"actual_evaluator_calls": 3, "paid_judge_cache_misses": 0},
    )
    steering_path.write_text('{"concept_id":0}\n')
    pq.write_table(table, parquet_path)
    native_methods.write_json(
        output / "manifest.json",
        {
            "schema_version": 1,
            "best_factors": best_factors,
            "call_accounting": {
                "actual_evaluator_calls": 3,
                "paid_judge_cache_misses": 0,
            },
            "files": {
                "judge_call_accounting.json": {"sha256": sha256(accounting_path)},
                "steering.jsonl": {"sha256": sha256(steering_path)},
                "steering_data.parquet": {"logical_sha256": logical_table_digest(table)},
            },
            "input_logical_sha256": input_logical_sha256,
        },
    )


def test_completed_judge_manifest_authenticates_outputs(tmp_path):
    output = tmp_path / "judged"
    table = pa.table({"concept_id": [0], "score": [1.0]})
    factors = {"DiffMean": 0.2, "LinearProbe": 0.4, "LoRA": 1.0}
    source = {"input_logical_sha256": "a" * 64}
    _write_judged_output(
        output,
        table=table,
        best_factors=factors,
        input_logical_sha256=source["input_logical_sha256"],
    )
    receipt = native_method_judge._validated_judge_receipt(
        output=output,
        source=source,
        best_factors=factors,
        relative=Path("gepa/points/p1"),
        identity={"concept_ids": [0]},
    )
    assert receipt["actual_evaluator_calls"] == 3
    (output / "steering.jsonl").write_text('{"concept_id":1}\n')
    with pytest.raises(AxBenchAdvError, match="manifest differs"):
        native_method_judge._validated_judge_receipt(
            output=output,
            source=source,
            best_factors=factors,
            relative=Path("gepa/points/p1"),
            identity={"concept_ids": [0]},
        )


def test_archived_control_receipt_requires_replay_lineage(tmp_path):
    output = tmp_path / "points/p0/judged"
    table = pa.table({"concept_id": [0], "score": [1.0]})
    factors = {"DiffMean": 0.2, "LinearProbe": 0.4, "LoRA": 1.0}
    source = {"input_logical_sha256": "a" * 64}
    _write_judged_output(
        output,
        table=table,
        best_factors=factors,
        input_logical_sha256=source["input_logical_sha256"],
    )
    with pytest.raises(AxBenchAdvError, match="replay lineage"):
        native_method_judge._validated_judge_receipt(
            output=output,
            source=source,
            best_factors=factors,
            relative=Path("points/p0"),
            identity={"concept_ids": [0]},
        )


def test_completed_judge_recovers_missing_receipt_without_provider(tmp_path, monkeypatch):
    relative = Path("gepa/points/p1")
    point = tmp_path / relative
    input_path = point / "input" / "steering_data.parquet"
    input_path.parent.mkdir(parents=True)
    input_table = pa.table({"concept_id": [0], "input_id": [0]})
    pq.write_table(input_table, input_path)
    native_methods.write_json(point / "emissions.json", {})
    factors = {"DiffMean": 0.2, "LinearProbe": 0.4, "LoRA": 1.0}
    _write_judged_output(
        point / "judged",
        table=input_table,
        best_factors=factors,
        input_logical_sha256=logical_table_digest(input_table),
    )
    monkeypatch.setattr(native_method_judge, "validate_final_judge_outputs", lambda **_kwargs: None)
    judge = Mock(side_effect=AssertionError("provider path must not run"))
    result = native_method_judge._judge_point(
        run_dir=tmp_path,
        relative=relative,
        table=input_table,
        emissions={},
        identity={"best_factors": factors, "concept_ids": [0]},
        cache=tmp_path / "judge.sqlite",
        cache_only=True,
        api_base_url="unused",
        judge=judge,
    )
    assert result == point / "judged" / "steering.jsonl"
    assert (point / "judge-receipt.json").is_file()
    judge.assert_not_called()


def test_final_judge_routes_shared_zero_gepa_and_both_counter_families():
    specs = native_method_judge._point_specs(
        {
            "pressures": [str(pressure) for pressure in native_methods.PRESSURES],
            "proposal_counts": [1, 2, 4, 8],
            "counter_magnitudes": [0.25, 0.5, 1.0, 2.0],
        }
    )
    assert [str(path) for path, _coordinate in specs] == [
        "points/p0",
        "gepa/points/p1",
        "gepa/points/p2",
        "gepa/points/p4",
        "gepa/points/p8",
        "counter/linear_probe_counter/points/m0.25",
        "counter/linear_probe_counter/points/m0.5",
        "counter/linear_probe_counter/points/m1.0",
        "counter/linear_probe_counter/points/m2.0",
        "counter/diffmean_counter/points/m0.25",
        "counter/diffmean_counter/points/m0.5",
        "counter/diffmean_counter/points/m1.0",
        "counter/diffmean_counter/points/m2.0",
    ]


@pytest.mark.parametrize(
    "pressures",
    [
        [],
        ["diffmean_counter", "diffmean_counter"],
        ["gcg"],
        ["unknown"],
    ],
)
def test_native_pressure_subset_rejects_empty_duplicate_and_unsupported(pressures):
    with pytest.raises(AxBenchAdvError, match="pressures"):
        native_methods._validate_pressures(pressures)


@pytest.mark.parametrize(
    "identity",
    [
        {
            "pressures": ["diffmean_counter", "linear_probe_counter"],
            "counter_magnitudes": [64.0],
        },
        {
            "pressures": ["linear_probe_counter", "linear_probe_counter"],
            "counter_magnitudes": [64.0],
        },
        {
            "pressures": ["gcg"],
            "counter_magnitudes": [64.0],
        },
        {
            "pressures": ["linear_probe_counter"],
            "counter_magnitudes": [64],
        },
        {
            "pressures": ["linear_probe_counter"],
            "counter_magnitudes": [64.0, 64.0],
        },
    ],
)
def test_final_judge_rejects_noncanonical_pressure_and_magnitude_identity(identity):
    with pytest.raises(AxBenchAdvError, match="identity|pressures"):
        native_method_judge._identity_grid(identity)


@pytest.mark.parametrize(
    "model_contract", _RUNTIME_CONTROL_CONTRACTS, ids=("d12", "d14")
)
def test_runtime_policy_builds_fresh_shared_zero_without_archive_replay(
    tmp_path, monkeypatch, model_contract
):
    output = tmp_path / "output"
    source_paths = {
        "final_root": tmp_path / "inputs/final",
        "archive": tmp_path / "inputs/archive",
        "checkpoint": tmp_path / "inputs/checkpoint",
        "tokenizer": tmp_path / "inputs/tokenizer.pkl",
        "mailbox": tmp_path / "mailbox",
    }
    factors = {"DiffMean": 0.2, "LinearProbe": 0.4, "LoRA": 1.0}
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={},
        judged_entries={},
        best_points={
            method: BestPoint(method, factor, 1.0, 1.0, 1.0)
            for method, factor in factors.items()
        },
        fits={},
        checkpoint={},
        geometry=_geometry(model_contract),
        generation={"seed": 42, "max_new_tokens": 128, "temperature": 1.0, "top_k": 50},
        identity={"authenticated": True},
    )

    class Adapter:
        model = _ToyDecoder()

    class Mailbox:
        def __init__(self, _path, *, timeout_seconds):
            assert timeout_seconds == 5.0

    fresh_calls = []

    def fresh_zero(**kwargs):
        fresh_calls.append(kwargs)
        return native_methods.RawEvaluation(
            f"fresh {kwargs['method']} c{kwargs['concept_id']} i{kwargs['input_id']}",
            0.25,
            1.75,
            '{"concept":"fresh concept","fluency":"fresh fluency"}',
            1,
            1,
        )

    archived = Mock(side_effect=AssertionError("runtime policy must not read archived controls"))
    replay = Mock(side_effect=AssertionError("runtime policy must not replay archived controls"))
    published = []
    counter_raw = native_methods.RawEvaluation("counter", 1.0, 1.0, "{}", 0, 0)
    monkeypatch.setattr(native_methods, "validate_native_method_sources", lambda **_k: sources)
    monkeypatch.setattr(native_methods, "_adapter", lambda **_k: Adapter())
    monkeypatch.setattr(native_methods, "native_runtime_settings", lambda **_k: {})
    monkeypatch.setattr(native_methods, "ArtifactMailbox", Mailbox)
    monkeypatch.setattr(native_methods, "_selected_raw", archived)
    monkeypatch.setattr(native_methods, "_verify_zero_controls", replay)
    monkeypatch.setattr(native_methods, "_evaluate_generated", fresh_zero)
    monkeypatch.setattr(native_methods, "_counter_cell", lambda **_k: [counter_raw] * 10)
    monkeypatch.setattr(
        native_methods,
        "_publish_point",
        lambda **kwargs: published.append(kwargs),
    )
    monkeypatch.setattr(native_methods, "_validate_union", lambda _path: ({}, {}))
    cfg = OmegaConf.create(
        {
            "phase": "run",
            "methods": list(native_methods.METHODS),
            "pressures": ["diffmean_counter"],
            "counter_magnitudes": [64.0],
            "proposal_counts": list(native_methods.PROPOSAL_COUNTS),
            "gepa_seed": 42,
            "seed_suffix": "unused",
            "reflection_model": "openai/gpt-5.6-luna",
            "reflection_minibatch_size": 3,
            "final_root": str(source_paths["final_root"]),
            "archive_roots": [str(source_paths["archive"])],
            "checkpoint_dir": str(source_paths["checkpoint"]),
            "tokenizer_path": str(source_paths["tokenizer"]),
            "output_dir": str(output),
            "mailbox_dir": str(source_paths["mailbox"]),
            "provider_timeout_seconds": 5.0,
            "step": 1003 if model_contract == LEGACY_MODEL_CONTRACT else 1001,
            "concept_ids": [0],
            "device": "cpu",
            "zero_control_source": "runtime",
        }
    )
    native_methods.run_native_method_adv(cfg)

    assert len(fresh_calls) == len(native_methods.METHODS) * 10
    assert all(call["control_key"] == "shared_zero" for call in fresh_calls)
    assert all(call["preserve_cached_accounting"] is True for call in fresh_calls)
    assert all(call["suffix"] == "" and call["magnitude"] is None for call in fresh_calls)
    control = published[0]
    assert control["relative"] == Path("points/p0")
    assert {raw.generation for raw in control["emissions"].values()} == {
        f"fresh {method} c0 i{input_id}"
        for method in native_methods.METHODS
        for input_id in range(10)
    }
    identity = native_methods.read_json(output / "identity.json")
    assert identity["zero_control_source"] == "runtime"
    assert identity["best_factors"] == factors
    assert identity["generation"] == sources.generation
    archived.assert_not_called()
    replay.assert_not_called()


def test_latent_only_driver_and_final_judge_route_only_bound_counter_grid(
    tmp_path, monkeypatch
):
    magnitudes = (64.0, 256.0, 512.0, 1024.0, 2048.0, 4096.0)
    output = tmp_path / "output"
    final_root = tmp_path / "inputs" / "final"
    archive = tmp_path / "inputs" / "archive"
    checkpoint = tmp_path / "inputs" / "checkpoint"
    tokenizer = tmp_path / "inputs" / "tokenizer.pkl"
    mailbox_dir = tmp_path / "mailbox"
    sources = NativeMethodSources(
        source_frame=pd.DataFrame(),
        generation_frames={},
        judged_entries={},
        best_points={
            method: BestPoint(method, 1.0, 1.0, 1.0, 1.0)
            for method in native_methods.METHODS
        },
        fits={},
        checkpoint={},
        geometry={"n_layer": 12, "n_embd": 768, "sequence_len": 1024},
        generation={"seed": 42, "max_new_tokens": 8, "temperature": 1.0, "top_k": 5},
        identity={"authenticated": True},
    )

    class Adapter:
        model = _ToyDecoder()

        @staticmethod
        def prompt_tokens(_text):
            raise AssertionError("counter-only run must not validate GEPA seed context")

    mailbox_calls = []
    mailbox_instances = []

    class Mailbox:
        def __init__(self, path, *, timeout_seconds):
            mailbox_calls.append((path, timeout_seconds))
            mailbox_instances.append(self)

    def no_gepa(*_args, **_kwargs):
        raise AssertionError("counter-only run must not construct or execute GEPA")

    raw = native_methods.RawEvaluation("generation", 1.0, 1.0, "{}", 0, 1)
    published = []
    counter_calls = []
    monkeypatch.setattr(native_methods, "validate_native_method_sources", lambda **_kwargs: sources)
    monkeypatch.setattr(native_methods, "_adapter", lambda **_kwargs: Adapter())
    monkeypatch.setattr(native_methods, "native_runtime_settings", lambda **_kwargs: {})
    monkeypatch.setattr(native_methods, "ArtifactMailbox", Mailbox)
    monkeypatch.setattr(native_methods, "MailboxReflection", no_gepa)
    monkeypatch.setattr(native_methods, "_gepa_cell", no_gepa)
    monkeypatch.setattr(native_methods, "_selected_raw", lambda *_args, **_kwargs: raw)
    monkeypatch.setattr(native_methods, "_verify_zero_controls", lambda **_kwargs: None)
    def counter_cell(**kwargs):
        counter_calls.append(kwargs)
        return [raw] * 10

    monkeypatch.setattr(native_methods, "_counter_cell", counter_cell)
    monkeypatch.setattr(
        native_methods,
        "_publish_point",
        lambda **kwargs: published.append((kwargs["relative"], kwargs["coordinate"])),
    )
    monkeypatch.setattr(native_methods, "_validate_union", lambda _path: ({}, {}))
    cfg = OmegaConf.create(
        {
            "phase": "run",
            "methods": list(native_methods.METHODS),
            "pressures": ["diffmean_counter", "linear_probe_counter"],
            "counter_magnitudes": list(magnitudes),
            "proposal_counts": list(native_methods.PROPOSAL_COUNTS),
            "gepa_seed": 42,
            "seed_suffix": "unused",
            "reflection_model": "openai/gpt-5.6-luna",
            "reflection_minibatch_size": 3,
            "final_root": str(final_root),
            "archive_roots": [str(archive)],
            "checkpoint_dir": str(checkpoint),
            "tokenizer_path": str(tokenizer),
            "output_dir": str(output),
            "mailbox_dir": str(mailbox_dir),
            "provider_timeout_seconds": 5.0,
            "step": 1001,
            "concept_ids": [0],
            "device": "cpu",
            "zero_control_source": "archived",
        }
    )
    status = native_methods.run_native_method_adv(cfg)

    expected = [Path("points/p0")] + [
        Path("counter") / pressure / "points" / native_methods.magnitude_label(magnitude)
        for pressure in (
            native_methods.Pressure.LINEAR_PROBE_COUNTER,
            native_methods.Pressure.DIFFMEAN_COUNTER,
        )
        for magnitude in magnitudes
    ]
    assert [relative for relative, _coordinate in published] == expected
    assert status["pressures"] == ["linear_probe_counter", "diffmean_counter"]
    assert mailbox_calls == [(mailbox_dir.resolve(), 5.0)]
    assert len(counter_calls) == len(native_methods.METHODS) * 2 * len(magnitudes)
    assert all(call["mailbox"] is mailbox_instances[0] for call in counter_calls)
    assert {call["reflection_model"] for call in counter_calls} == {"openai/gpt-5.6-luna"}

    judged = []
    aggregates = []
    emissions = {
        (method, 0, 0): raw
        for method in native_methods.METHODS
    }
    monkeypatch.setattr(
        native_method_judge,
        "_validate_point",
        lambda **kwargs: (object(), emissions),
    )

    def judge_point(**kwargs):
        judged.append(kwargs["relative"])
        return output / kwargs["relative"] / "judged" / "steering.jsonl"

    monkeypatch.setattr(native_method_judge, "_judge_point", judge_point)
    monkeypatch.setattr(native_method_judge, "build_aggregate", no_gepa)
    monkeypatch.setattr(
        native_method_judge,
        "build_counter_aggregate",
        lambda **kwargs: aggregates.append(kwargs["output_path"]) or pd.DataFrame(),
    )
    monkeypatch.setattr(native_method_judge, "plot_concept_vs_fluency", lambda *_a, **_k: None)
    judge_cfg = OmegaConf.create(
        {
            "run_dir": str(output),
            "judge_cache": str(tmp_path / "judge-cache.sqlite"),
            "cache_only": True,
            "api_base_url": "",
            "final_root": str(final_root),
        }
    )
    result = native_method_judge.run_native_method_final_judge(judge_cfg)
    assert judged == expected
    assert aggregates == [
        output / "counter" / pressure / "aggregate.parquet"
        for pressure in ("linear_probe_counter", "diffmean_counter")
    ]
    assert result["counter_direction_families"] == {
        "linear_probe_counter": "LinearProbe",
        "diffmean_counter": "DiffMean",
    }
