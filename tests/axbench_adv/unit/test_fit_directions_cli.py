"""Closed Hydra boundary tests for the issue-1673 direction fitter."""

from __future__ import annotations

import json
import weakref
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_adv import fit_directions
from scripts.eval.axbench_adv.pressures import Pressure
from steering.eval.axbench_adv import direction_fitting as fitting_module
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_counter_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.direction_fitting import (
    fit_recipe_identity,
)
from steering.eval.axbench_upstream.encdec_baselines.capture import FitRowConditioning
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import build_tiny_model


def _config() -> dict:
    return {
        "prepared_dir": "/prepared",
        "target_meta_path": "/checkpoint/meta_001125.json",
        "feature_statistics_path": "/stats/feature_statistics.safetensors",
        "output_root": "/output/directions",
        "max_concepts": 50,
        "layer": 21,
        "seed": 42,
        "device": "cuda:0",
    }


def test_cli_passes_exact_resolved_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    observed = {}

    def fake_fit(**kwargs):
        observed.update(kwargs)
        return SimpleNamespace(manifest_path=Path("/output/fit_manifest.json"), artifacts=())

    monkeypatch.setattr(fit_directions, "fit_project_directions", fake_fit)
    result = fit_directions.run_from_config(OmegaConf.create(_config()))
    assert result.manifest_path == Path("/output/fit_manifest.json")
    assert observed == {
        "prepared_dir": Path("/prepared"),
        "target_meta_path": Path("/checkpoint/meta_001125.json"),
        "feature_statistics_path": Path("/stats/feature_statistics.safetensors"),
        "output_root": Path("/output/directions"),
        "max_concepts": 50,
        "layer": 21,
        "seed": 42,
        "device": "cuda:0",
    }


def test_cli_rejects_unknown_or_unresolved_fields() -> None:
    unknown = {**_config(), "fallback_checkpoint": "/wrong"}
    with pytest.raises(ValueError, match="fields must be exactly"):
        fit_directions.run_from_config(OmegaConf.create(unknown))
    unresolved = {**_config(), "prepared_dir": "???"}
    with pytest.raises(ValueError, match="prepared_dir"):
        fit_directions.run_from_config(OmegaConf.create(unresolved))


def test_cli_rejects_out_of_scope_pretrain_metadata_before_fit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_fit(**_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(fit_directions, "fit_project_directions", fake_fit)
    config = {**_config(), "pretrain_meta_path": "/checkpoint/pretrain_meta.json"}
    with pytest.raises(ValueError, match="fields must be exactly"):
        fit_directions.run_from_config(OmegaConf.create(config))
    assert not called


def _functional_identity() -> dict:
    return {
        "checkpoint": {
            "target_model": {"name": "model_000001.pt", "byte_length": 1, "sha256": "a" * 64},
            "companion_metadata": {
                "name": "meta_000001.json",
                "byte_length": 1,
                "sha256": "b" * 64,
            },
        },
        "fit": fit_recipe_identity(layer=1, decoder_width=64, seed=42),
        "reproducibility": {},
    }


def test_cli_functional_fit_to_preflight_and_both_pressures(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    prepared_train = prepared / "generate/train_data.parquet"
    prepared_train.parent.mkdir()
    (prepared / "prepare_manifest.json").write_text("{}")
    (prepared / "generate/metadata.jsonl").write_text(
        json.dumps(
            {
                "concept_id": 0,
                "concept": "gardening",
                "concept_genres_map": {"gardening": ["hobby"]},
            }
        )
        + "\n"
    )
    pd.DataFrame(
        [
            {
                "input": "garden one",
                "output": "plants grow",
                "output_concept": "gardening",
                "category": "positive",
                "concept_genre": "hobby",
            },
            {
                "input": "garden two",
                "output": "water soil",
                "output_concept": "gardening",
                "category": "positive",
                "concept_genre": "hobby",
            },
            {
                "input": "neutral one",
                "output": "plain answer",
                "output_concept": "EEEEE",
                "category": "negative",
                "concept_genre": "hobby",
            },
            {
                "input": "neutral two",
                "output": "other reply",
                "output_concept": "EEEEE",
                "category": "negative",
                "concept_genre": "hobby",
            },
        ]
    ).to_parquet(prepared_train, index=False)
    target_meta = tmp_path / "meta_000001.json"
    target_meta.write_text("{}")
    statistics = tmp_path / "stats.safetensors"
    statistics.write_bytes(b"stats")
    output = tmp_path / "directions"
    identity = _functional_identity()
    setup = SimpleNamespace(meta={"model_config": {"sequence_len": 128}})
    model = build_tiny_model(n_layer=3, n_embd=64, xattn_frequency=1).eval()
    model.init_weights()
    extraction_refs: list[weakref.ReferenceType] = []

    class _ExtractionResource:
        pass

    def load_extraction(*_args):
        extraction = _ExtractionResource()
        sae = _ExtractionResource()
        extraction_refs.extend((weakref.ref(extraction), weakref.ref(sae)))
        return extraction, sae

    monkeypatch.setattr(fitting_module, "load_method_setup", lambda *_args, **_kwargs: setup)
    def build_identity(**_kwargs):
        return deepcopy(identity)

    monkeypatch.setattr(fitting_module, "build_shared_fit_identity", build_identity)
    monkeypatch.setattr(fitting_module, "get_tokenizer", MockTokenizer)
    monkeypatch.setattr(fitting_module, "load_extraction_stack", load_extraction)

    def fake_extracted_conditioning(
        _setup,
        _extraction_model,
        _sae,
        _tokenizer,
        *,
        token_ids,
        surface,
    ) -> FitRowConditioning:
        assert "row" in surface
        exact = tuple(token_ids)
        return FitRowConditioning(
            token_ids=exact,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(2, 5),
                    buckets=(1, 2),
                    char_start=0,
                    char_end=len(exact),
                ),
            ),
            segment_assignments=(0,) * len(exact),
        )

    monkeypatch.setattr(
        fitting_module,
        "build_fit_row_conditioning",
        fake_extracted_conditioning,
    )
    monkeypatch.setattr(fitting_module, "load_target_model", lambda *_args: (model, object()))
    config = {
        **_config(),
        "prepared_dir": str(prepared),
        "target_meta_path": str(target_meta),
        "feature_statistics_path": str(statistics),
        "output_root": str(output),
        "max_concepts": 1,
        "layer": 1,
        "device": "cpu",
    }
    publication = fit_directions.run_from_config(OmegaConf.create(config))
    assert publication.manifest_path == output / "fit_manifest.json"
    assert len(publication.artifacts) == 2
    assert all(reference() is None for reference in extraction_refs)
    preflight = preflight_project_direction_fit(
        output,
        ProjectCounterExpectations(
            target_model_sha256="a" * 64,
            target_metadata_sha256="b" * 64,
            concept_texts={0: "gardening"},
            decoder_width=64,
            layer=1,
            fit_seed=42,
        ),
    )
    for pressure in (Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER):
        loaded = load_counter_direction(
            pressure=pressure,
            method="SelfConditioning",
            concept_id=0,
            baseline_train=tmp_path,
            baseline_layer=28,
            project_fit=preflight,
        )
        assert loaded.direction.shape == (64,)
    assert model._post_cross_attention_eval_callback is None
