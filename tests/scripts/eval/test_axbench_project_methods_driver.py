"""Driver contracts for corrected AxBench SelfConditioning."""

from __future__ import annotations

import hashlib
import json
import weakref
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch as t
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors.torch import save_file

from scripts.benchmark import benchmark_axbench_upstream as benchmark
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    NL_PROJECT_METHODS,
    validate_steering_frame,
)
from scripts.eval.axbench_upstream.run_baselines import AXBENCH_UPSTREAM_COMMIT
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
    SelfCondScoreMasking,
)
from steering.conditional.sampling import Distribution, NegativeSamplingSelection
from steering.eval.axbench_upstream import run_methods as driver
from steering.eval.axbench_upstream.comparison import compare_project_bundle
from steering.eval.axbench_upstream.data import ConceptTrainingTexts, load_metadata
from steering.eval.axbench_upstream.methods import (
    SELF_CONDITIONING_MASKED_POLICY,
    SELF_CONDITIONING_POLICY,
)
from steering.eval.axbench_upstream.project_input import (
    ALPACA_EVAL_FILE,
    ALPACA_EVAL_REPO,
    ALPACA_EVAL_REVISION,
    AXBENCH_BENCHMARK_REPO,
    AXBENCH_BENCHMARK_REVISION,
    write_project_input,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.encdec_test_helpers import (
    EXPECTED_TURN_GEOMETRY,
    TURN_FIELD_DEFAULTS,
    turn_prompt_selection_context,
)
from tests.encdec_test_helpers import method_setup_double as _dispatch_setup

_REPO_ROOT = Path(__file__).resolve().parents[3]
_GENERATION_SEED_GOLDEN = 320911683
_SELECTION_SEED_GOLDEN = 448567669


def test_runtime_identity_binds_exact_python_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver.platform, "python_version", lambda: "3.13.test")

    identity = driver._runtime_identity("cpu")

    assert identity["python"] == "3.13.test"
    assert identity["device_type"] == "cpu"


def test_runtime_identity_binds_numeric_modes_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    original_precision = t.get_float32_matmul_precision()
    try:
        t.set_float32_matmul_precision("highest")
        highest = driver._runtime_identity("cpu")
        t.set_float32_matmul_precision("high")
        high = driver._runtime_identity("cpu")
    finally:
        t.set_float32_matmul_precision(original_precision)

    assert highest["float32_matmul_precision"] == "highest"
    assert high["float32_matmul_precision"] == "high"
    assert highest != high
    assert high["deterministic_algorithms"] is t.are_deterministic_algorithms_enabled()
    assert high["deterministic_warn_only"] is t.is_deterministic_algorithms_warn_only_enabled()
    assert high["cublas_workspace_config"] == ":4096:8"
    assert high["nvidia_tf32_override"] == "0"
    assert high["torch_allow_tf32_cublas_override"] == "1"


def test_runtime_identity_uses_explicit_cuda_device_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(driver.t.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(driver.t.cuda, "get_device_name", lambda index: f"gpu-{index}")
    monkeypatch.setattr(driver.t.cuda, "get_device_capability", lambda index: (9, index))

    identity = driver._runtime_identity("cuda:3")

    assert identity["gpu_name"] == "gpu-3"
    assert identity["gpu_capability"] == [9, 3]


def test_tokenizer_artifact_identity_binds_exact_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_path = tokenizer_dir / "tokenizer.pkl"
    tokenizer_path.write_bytes(b"tokenizer-v1")
    monkeypatch.setattr(driver, "get_base_dir", lambda: str(tmp_path))

    first = driver._tokenizer_artifact_identity()
    tokenizer_path.write_bytes(b"tokenizer-v2")
    second = driver._tokenizer_artifact_identity()

    assert first == {
        "name": "tokenizer.pkl",
        "byte_length": 12,
        "sha256": "7e8c1ee480d271048e345dae82d0df012f3fc9a0a6fb4597b2d5c005fe304f87",
    }
    assert second["byte_length"] == first["byte_length"]
    assert second["sha256"] != first["sha256"]


def test_tokenizer_artifact_identity_rejects_missing_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver, "get_base_dir", lambda: str(tmp_path))

    with pytest.raises(FileNotFoundError, match="tokenizer.pkl"):
        driver._tokenizer_artifact_identity()


def _main_frame(max_concepts: int = 2) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for concept_id in range(max_concepts):
        for input_id in range(10):
            for factor in FULL_FACTORS:
                rows.append(
                    {
                        "dataset_name": "AlpacaEval",
                        "concept_id": concept_id,
                        "input_concept": f"concept-{concept_id}",
                        "input_id": input_id,
                        "factor": factor,
                        "original_prompt": f"prompt {concept_id}/{input_id}",
                        "steered_input": f"steered {concept_id}/{input_id}",
                        "input": f"input {concept_id}/{input_id}",
                        "suppress_original": f"suppress original {concept_id}/{input_id}",
                        "suppress_rewrite": f"suppress rewrite {concept_id}/{input_id}",
                        "steered_prompt": f"steered prompt {concept_id}/{input_id}",
                        "defense": "none",
                        "sae_link": f"https://example.test/{concept_id}",
                        "sae_id": 100 + concept_id,
                        "DiffMean_steered_generation": (
                            f"DiffMean healthy generation {concept_id}/{input_id}/{factor}"
                        ),
                        "DiffMean_strength": factor,
                        "LinearProbe_steered_generation": (
                            f"LinearProbe healthy generation {concept_id}/{input_id}/{factor}"
                        ),
                        "LinearProbe_strength": factor,
                    }
                )
    frame = pd.DataFrame(rows)
    for column in ("concept_id", "input_id", "sae_id"):
        frame[column] = frame[column].astype("int64")
    for column in ("factor", "DiffMean_strength", "LinearProbe_strength"):
        frame[column] = frame[column].astype("float64")
    return frame.loc[
        :,
        [
            *INPUT_COLUMNS,
            "DiffMean_steered_generation",
            "DiffMean_strength",
            "LinearProbe_steered_generation",
            "LinearProbe_strength",
        ],
    ]


def _training(max_concepts: int = 2) -> dict[int, ConceptTrainingTexts]:
    return {
        concept_id: ConceptTrainingTexts(
            concept_id=concept_id,
            concept=f"concept-{concept_id}",
            genre="text",
            positive=(f"positive-{concept_id}",),
            negative=(f"negative-{concept_id}",),
        )
        for concept_id in range(max_concepts)
    }


def _write_inputs(tmp_path: Path, *, max_concepts: int = 2) -> dict[str, Any]:
    prepared_dir = tmp_path / "prepared"
    (prepared_dir / "generate").mkdir(parents=True)
    (prepared_dir / "data").mkdir()
    metadata_path = prepared_dir / "generate" / "metadata.jsonl"
    metadata_path.write_text(
        "".join(
            json.dumps(
                {
                    "concept_id": concept_id,
                    "concept": f"concept-{concept_id}",
                    "ref": f"https://example.test/sae/{100 + concept_id}",
                }
            )
            + "\n"
            for concept_id in range(max_concepts)
        ),
        encoding="utf-8",
    )
    alpaca_path = prepared_dir / "data" / "alpaca_eval.json"
    alpaca_path.write_text(
        json.dumps([{"instruction": f"instruction-{index}"} for index in range(20)]),
        encoding="utf-8",
    )
    (prepared_dir / "prepare_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "benchmark": {
                    "repo": AXBENCH_BENCHMARK_REPO,
                    "revision": AXBENCH_BENCHMARK_REVISION,
                },
                "alpaca_eval": {
                    "repo": ALPACA_EVAL_REPO,
                    "revision": ALPACA_EVAL_REVISION,
                    "file": ALPACA_EVAL_FILE,
                },
                "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
                "artifacts": {
                    "generate/metadata.jsonl": {
                        "bytes": metadata_path.stat().st_size,
                        "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    },
                    "data/alpaca_eval.json": {
                        "bytes": alpaca_path.stat().st_size,
                        "sha256": hashlib.sha256(alpaca_path.read_bytes()).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    (prepared_dir / "generate" / "train_data.parquet").write_bytes(b"training")
    main_path = tmp_path / "main.parquet"
    _main_frame(max_concepts).to_parquet(main_path, index=False)
    target_meta = tmp_path / "meta_000007.json"
    target_meta.write_text('{"step": 7}', encoding="utf-8")
    (tmp_path / "model_000007.pt").write_bytes(b"target-model")
    statistics = tmp_path / "feature_statistics.safetensors"
    save_file({"weights": t.tensor([1.0, 2.0, 3.0])}, str(statistics), metadata={"k": "v"})
    output_path = tmp_path / "project_v2" / "inference" / "steering_data.parquet"
    manifest_path = tmp_path / "project_v2" / "run_manifest.json"
    return {
        "main_path": main_path,
        "prepared_dir": prepared_dir,
        "output_path": output_path,
        "manifest_path": manifest_path,
        "target_meta_path": target_meta,
        "feature_statistics_path": statistics,
        "max_concepts": max_concepts,
        "device": "cpu",
        "seed": 42,
        "max_new_tokens": 128,
        "extraction_batch_size": 3,
        "self_conditioning_policy": SELF_CONDITIONING_POLICY,
        "self_conditioning_masked_policy": SELF_CONDITIONING_MASKED_POLICY,
        "force": False,
        "pretrain_meta_path": None,
        "require_exact_statistics_identity": True,
    }


def _enable_nl_receipt(kwargs: dict[str, Any]) -> Path:
    root = Path(kwargs["manifest_path"]).parent / "nl-inputs"
    root.mkdir(parents=True, exist_ok=True)
    descriptions = root / "descriptions.npz"
    concepts = root / "concepts.npz"
    np.savez(
        descriptions,
        fids=np.arange(8, dtype=np.int64),
        emb=np.asarray(
            [
                [1.0, 0.0],
                [0.8, 0.2],
                [0.5, 0.5],
                [0.2, 0.8],
                [0.0, 1.0],
                [-0.2, 0.8],
                [-0.5, 0.5],
                [-1.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    concept_texts = [f"concept-{index}" for index in range(int(kwargs["max_concepts"]))]
    np.savez(
        concepts,
        texts=np.asarray(concept_texts),
        emb=np.asarray(
            [[1.0, float(index)] for index in range(len(concept_texts))],
            dtype=np.float32,
        ),
    )
    embedding_space = {
        "model": "test/model",
        "revision": "1" * 40,
        "encoding": "test-encoding-v1",
        "library_versions": {"numpy": "test"},
    }
    receipt = root / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "description_embeddings": {
                    "path": descriptions.name,
                    "sha256": hashlib.sha256(descriptions.read_bytes()).hexdigest(),
                    "source_texts_sha256": "2" * 64,
                    "embedding_space": embedding_space,
                },
                "concept_embeddings": {
                    "path": concepts.name,
                    "sha256": hashlib.sha256(concepts.read_bytes()).hexdigest(),
                    "source_texts_sha256": hashlib.sha256(
                        json.dumps(
                            concept_texts,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest(),
                    "embedding_space": embedding_space,
                },
            }
        ),
        encoding="utf-8",
    )
    kwargs["nl_target_source_manifest_path"] = receipt
    return receipt


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, Any],
    *,
    generation: str = "a healthy SelfConditioning generation",
) -> dict[str, Any]:
    events: list[str] = []
    fitted_concepts: list[tuple[int, ...]] = []
    prepared_tokens: list[tuple[int, ...]] = []
    selection_seeds: dict[tuple[int, int], set[int]] = {}
    token_seeds: dict[tuple[int, int], set[int]] = {}
    setup: Any = SimpleNamespace(
        target_meta_path=Path(kwargs["target_meta_path"]).resolve(),
        meta={
            "step": 7,
            "user_config": {
                "model_tag": "test-target",
                "feature_selection": {"mode": "top_k_percentile"},
                "negative_selection": {"distribution": "geometric"},
            },
        },
        n_sae_features=8,
        feature_contract=CheckpointFeatureContract(feature_mask=None, resolved_clusters=None),
        consolidation=None,
        feature_vocab=None,
        feature_mask=None,
        allowed_ids=None,
        output_selection=SimpleNamespace(min_k=1, mean=2.0),
        output_negative_selection=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=0,
            max_k=4,
        ),
        **TURN_FIELD_DEFAULTS,
    )
    target_model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=1024))

    def render(_tokenizer: Any, prompt: str) -> Any:
        concept, input_id = (int(value) for value in prompt.removeprefix("prompt ").split("/"))
        return SimpleNamespace(token_ids=[1, 10 + concept, 100 + input_id, 2], assistant_end_id=9)

    def prepare(
        _setup: Any,
        _model: Any,
        _sae: Any,
        _tokenizer: Any,
        *,
        token_ids: tuple[int, ...],
        assistant_end_id: int,
    ) -> Any:
        events.append("prepare")
        exact = tuple(token_ids)
        prepared_tokens.append(exact)
        return driver.PreparedPromptContext(
            token_ids=exact,
            assistant_end_id=assistant_end_id,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(1, 2),
                    buckets=(0, 1),
                    char_start=0,
                    char_end=4,
                ),
            ),
        )

    def fit(
        _setup: Any, _model: Any, _sae: Any, training: Any, **_kwargs: Any,
    ) -> dict[int, t.Tensor]:
        assert _kwargs["extraction_batch_size"] == kwargs["extraction_batch_size"]
        events.append("fit")
        fitted_concepts.append(tuple(training))
        return {concept_id: t.linspace(0.8, -0.8, 8) for concept_id in training}

    def load_target(_setup: Any, _device: str) -> tuple[Any, Any]:
        events.append("target")
        return target_model, object()

    masked_selection_seeds: dict[tuple[int, int], set[int]] = {}
    masked_token_seeds: dict[tuple[int, int], set[int]] = {}
    scalar_calls: list[tuple[tuple[int, int], Any, int, int]] = []
    batch_calls: list[tuple[tuple[int, int], tuple[Any, ...]]] = []

    def generate(
        _setup: Any,
        _model: Any,
        _tokenizer: Any,
        prepared: Any,
        policy: Any,
        *,
        selection_seed: int,
        seed: int,
        max_new_tokens: int,
    ) -> str:
        assert max_new_tokens == 128
        pair = (prepared.token_ids[1] - 10, prepared.token_ids[2] - 100)
        scalar_calls.append((pair, policy, selection_seed, seed))
        # Recover the driving factor from the frozen policy value: the mapped
        # interpolation weight or the method-scoped token seed identifies it.
        if isinstance(policy, SelfCondScoreMasking):
            factor = next(
                candidate
                for concept_id in range(int(kwargs["max_concepts"]))
                for input_id in range(10)
                for candidate in FULL_FACTORS
                if driver.derive_generation_seed(
                    int(kwargs["seed"]),
                    "SelfConditioningMasked",
                    concept_id,
                    input_id,
                    candidate,
                )
                == seed
            )
            masked_selection_seeds.setdefault(pair, set()).add(selection_seed)
            masked_token_seeds.setdefault(pair, set()).add(seed)
            prefix = "masked "
        else:
            assert isinstance(policy, SelfCondScoreInterpolation)
            factor = min(FULL_FACTORS, key=lambda f: abs(f / 6.0 - float(policy.weight)))
            selection_seeds.setdefault(pair, set()).add(selection_seed)
            token_seeds.setdefault(pair, set()).add(seed)
            prefix = ""
        if not generation.strip():
            return generation
        return f"{prefix}{generation} {pair} factor={float(factor)}"

    def generate_batch(
        _setup: Any,
        _model: Any,
        _tokenizer: Any,
        prepared: Any,
        requests: Any,
        *,
        max_new_tokens: int,
    ) -> tuple[str, ...]:
        request_tuple = tuple(requests)
        pair = (prepared.token_ids[1] - 10, prepared.token_ids[2] - 100)
        batch_calls.append((pair, request_tuple))
        first_nested_call = len(scalar_calls)
        generated = tuple(
            generate(
                _setup,
                _model,
                _tokenizer,
                prepared,
                request.policy,
                selection_seed=request.selection_seed,
                seed=request.token_seed,
                max_new_tokens=max_new_tokens,
            )
            for request in request_tuple
        )
        del scalar_calls[first_nested_call:]
        return generated

    monkeypatch.setattr(driver, "get_tokenizer", lambda: object())
    monkeypatch.setattr(
        driver,
        "_tokenizer_artifact_identity",
        lambda: {"name": "tokenizer.pkl", "byte_length": 9, "sha256": "a" * 64},
    )
    monkeypatch.setattr(driver, "render_prompt_for_chat_completion", render)
    monkeypatch.setattr(driver, "select_training_texts", lambda *_args, **_kwargs: _training())

    monkeypatch.setattr(driver, "load_method_setup", lambda *_args, **_kwargs: setup)
    monkeypatch.setattr(
        driver,
        "load_extraction_stack",
        lambda *_args, **_kwargs: (object(), object()),
    )
    monkeypatch.setattr(driver, "prepare_prompt_context", prepare)
    monkeypatch.setattr(driver, "fit_concept_dense_diffmeans", fit)
    monkeypatch.setattr(driver, "load_target_model", load_target)
    monkeypatch.setattr(driver, "assert_prepared_prompt_matches", lambda *_args: None)
    monkeypatch.setattr(driver, "generate_self_conditioning", generate)
    monkeypatch.setattr(driver, "generate_self_conditioning_batch", generate_batch)
    return {
        "setup": setup,
        "events": events,
        "fitted_concepts": fitted_concepts,
        "prepared_tokens": prepared_tokens,
        "selection_seeds": selection_seeds,
        "token_seeds": token_seeds,
        "masked_selection_seeds": masked_selection_seeds,
        "masked_token_seeds": masked_token_seeds,
        "scalar_calls": scalar_calls,
        "batch_calls": batch_calls,
    }


def test_driver_prepares_once_fits_once_releases_then_generates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)

    result = driver.run_project_methods(**kwargs)

    assert result.shape == (240, 18)
    assert tuple(result.columns) == (
        *INPUT_COLUMNS,
        "SelfConditioning_steered_generation",
        "SelfConditioning_strength",
        "SelfConditioningMasked_steered_generation",
        "SelfConditioningMasked_strength",
    )
    assert result["SelfConditioning_strength"].equals(result["factor"])
    assert result["SelfConditioning_steered_generation"].str.strip().ne("").all()
    assert result["SelfConditioningMasked_strength"].equals(result["factor"])
    assert result["SelfConditioningMasked_steered_generation"].str.strip().ne("").all()
    assert len(observed["prepared_tokens"]) == 20
    assert len(set(observed["prepared_tokens"])) == 20
    assert observed["events"].count("fit") == 1
    assert observed["events"].index("target") > max(
        index for index, event in enumerate(observed["events"]) if event in {"prepare", "fit"}
    )
    assert all(len(seeds) == 1 for seeds in observed["selection_seeds"].values())
    assert all(len(seeds) == len(FULL_FACTORS) for seeds in observed["token_seeds"].values())
    assert observed["scalar_calls"] == []
    assert len(observed["batch_calls"]) == 20
    assert {len(requests) for _pair, requests in observed["batch_calls"]} == {24}
    first_requests = next(requests for pair, requests in observed["batch_calls"] if pair == (0, 0))
    expected_cells = tuple(
        cell
        for cell in driver.build_project_cell_grid(_main_frame(), seed=42)
        if (cell.concept_id, cell.input_id) == (0, 0)
    )
    assert len(first_requests) == len(expected_cells) == 24
    for request, cell in zip(first_requests, expected_cells, strict=True):
        assert request.selection_seed == cell.selection_seed
        assert request.token_seed == cell.token_seed
        if cell.method == "SelfConditioning":
            assert isinstance(request.policy, SelfCondScoreInterpolation)
        else:
            assert isinstance(request.policy, SelfCondScoreMasking)
            factor = float.fromhex(cell.factor_hex)
            assert int(
                request.policy.positive_pool_limit
            ) == driver.self_conditioning_masked_pool_limit(
                factor,
                n_candidates=8,
                mean=2.0,
            )
            assert int(
                request.policy.negative_pool_limit
            ) == driver.self_conditioning_masked_pool_limit(
                factor,
                n_candidates=8,
                mean=3.0,
            )

    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 3
    assert set(manifest) == {
        "schema_version",
        "repository_git_commit",
        "repository_git_worktree",
        "vendored_axbench_upstream_commit",
        "package_lock_sha256",
        "target",
        "dataset_revision",
        "prepared_manifest",
        "configuration",
        "generation",
        "inputs",
        "method_order",
        "methods",
    }
    assert set(manifest["inputs"]) == {
        "main_steering_parquet_sha256",
        "feature_statistics_sha256",
    }
    assert "project_input_path" not in manifest["configuration"]
    assert "project_input_manifest_path" not in manifest["configuration"]
    identity = json.loads(
        (Path(kwargs["manifest_path"]).parent / "_project_cell_resume_v1" / "identity.json")
        .read_text(encoding="utf-8")
    )["semantic_identity"]
    assert identity["schema_version"] == 1


    assert set(identity["inputs"]) >= {"main_sha256"}
    assert "source_manifest_sha256" not in identity["inputs"]
    assert manifest["method_order"] == [
        "DiffMean",
        "LinearProbe",
        "PromptSteering",
        "LoRA",
        "SelfConditioning",
        "SelfConditioningMasked",
    ]
    assert set(manifest["methods"]) == {"SelfConditioning", "SelfConditioningMasked"}
    assert manifest["methods"]["SelfConditioning"]["factor_6"] == (
        "pure_dense_feature_space_diffmean"
    )
    assert manifest["methods"]["SelfConditioning"]["factor_0"] == (
        "pure_prompt_prediction_logit_ranking"
    )
    assert manifest["methods"]["SelfConditioning"]["policy"] == SELF_CONDITIONING_POLICY
    assert manifest["generation"] == {"temperature": 1.0, "top_k": 50}
    assert "self_conditioning_segment_geometry" not in manifest
    assert manifest["methods"]["SelfConditioning"]["prompt_features"] == (
        "real context-independent fixed-128 Gemma l22 SAE active/"
        "true-inactive features from nested checkpoint conditioning"
    )
    assert manifest["methods"]["SelfConditioning"]["completion_slots"] == (
        "one_predicted_slot_for_complete_continuation"
    )
    masked_block = manifest["methods"]["SelfConditioningMasked"]
    assert masked_block == {
        "factors": list(FULL_FACTORS),
        "policy": SELF_CONDITIONING_MASKED_POLICY,
        "strength_semantics": "linear_rank_pool_limit_from_N_to_2x_side_mean",
        "rounding": "nearest_ties_to_even",
        "clamp": {"minimum": 1, "maximum": "eligible_head_rows"},
        "eligible_head_rows": 8,
        "positive_mean": 2.0,
        "negative_mean": 3.0,
        "pool_limits": [
            {
                "factor": factor,
                "positive": min(8, max(1, round(8 * (1 - factor / 6) + 4 * factor / 6))),
                "negative": min(8, max(1, round(8 * (1 - factor / 6) + 6 * factor / 6))),
            }
            for factor in FULL_FACTORS
        ],
        "candidate_eligibility": "post_feature_mask_sign_consistent_stable_diffmean_rank",
        "final_ranking": "last_prompt_segment_max_pooled_float32_model_logits",
        "exact_zero_diffmean_eligible": False,
        "selection_seed": "sha256(domain,base_seed,concept_id,input_id); factor excluded",
        "token_seed": "sha256(base_seed,method,concept_id,input_id,float_factor_hex)",
    }
    assert "masked_threshold_scale" not in manifest["configuration"]


def test_nl_driver_uses_real_scorer_scalar_cohort_and_publishes_bound_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    receipt_path = _enable_nl_receipt(kwargs)
    observed = _install_fakes(monkeypatch, kwargs)
    real_builder = driver.build_retrieval_target_scores
    build_calls: list[dict[str, Any]] = []

    def record_build(**named: Any) -> dict[int, t.Tensor]:
        build_calls.append(named)
        return real_builder(**named)

    monkeypatch.setattr(driver, "build_retrieval_target_scores", record_build)

    raw = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    result = driver.run_project_methods_from_dictconfig(OmegaConf.create(raw))

    assert len(build_calls) == 1
    assert build_calls[0]["concepts"] == {0: "concept-0", 1: "concept-1"}
    assert result.shape == (240, 20)
    assert tuple(result.columns) == (*INPUT_COLUMNS, *driver.NL_PROJECT_COLUMNS)
    assert result["NLSteering_strength"].equals(result["factor"])
    assert result["NLSteering_steered_generation"].str.strip().ne("").all()
    assert len(observed["batch_calls"]) == 20
    assert {len(requests) for _pair, requests in observed["batch_calls"]} == {24}
    assert len(observed["scalar_calls"]) == 240
    first_pair = [call for call in observed["scalar_calls"] if call[0] == (0, 0)]
    assert len(first_pair) == len(FULL_FACTORS)
    assert all(isinstance(call[1], SelfCondScoreInterpolation) for call in first_pair)
    assert all(call[2] == first_pair[0][2] for call in first_pair)
    assert [call[3] for call in first_pair] == [
        driver.derive_generation_seed(42, "NLSteering", 0, 0, factor)
        for factor in FULL_FACTORS
    ]
    target = first_pair[0][1].target_scores
    assert target.dtype is t.float32
    assert target.device.type == "cpu"
    assert target.shape == (8,)
    assert t.equal(
        target,
        t.tensor(
            [
                float.fromhex("0x1.0000000000000p+0"),
                float.fromhex("0x1.f0b6860000000p-1"),
                float.fromhex("0x1.6a09e80000000p-1"),
                float.fromhex("0x1.f0b6860000000p-3"),
                float.fromhex("0x0.0p+0"),
                float.fromhex("-0x1.f0b6860000000p-3"),
                float.fromhex("-0x1.6a09e80000000p-1"),
                float.fromhex("-0x1.0000000000000p+0"),
            ],
            dtype=t.float32,
        ),
    )
    second_target = next(
        call[1].target_scores for call in observed["scalar_calls"] if call[0][0] == 1
    )
    assert t.equal(
        second_target,
        t.tensor(
            [
                float.fromhex("0x0.0p+0"),
                float.fromhex("0x1.b390a00000000p-4"),
                float.fromhex("0x1.a827a00000000p-3"),
                float.fromhex("0x1.b390a00000000p-4"),
                float.fromhex("0x0.0p+0"),
                float.fromhex("-0x1.16ee380000000p-3"),
                float.fromhex("-0x1.0000000000000p-1"),
                float.fromhex("-0x1.0000000000000p+0"),
            ],
            dtype=t.float32,
        ),
    )
    assert all(t.equal(call[1].target_scores, target) for call in first_pair)

    manifest_path = Path(kwargs["manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["method_order"][-3:] == list(NL_PROJECT_METHODS)
    assert set(manifest["methods"]) == set(NL_PROJECT_METHODS)
    nl_block = manifest["methods"]["NLSteering"]
    assert set(nl_block) == {"factors", "policy", "target_source"}
    assert nl_block["policy"] == driver.NL_STEERING_POLICY
    source = nl_block["target_source"]
    assert set(source) == {
        "algorithm",
        "feature_contract",
        "concepts",
        "description_embeddings",
        "concept_embeddings",
    }
    assert source["algorithm"] == driver.NL_RETRIEVAL_ALGORITHM
    assert source["concepts"] == [
        {"concept_id": 0, "text": "concept-0"},
        {"concept_id": 1, "text": "concept-1"},
    ]
    for name in ("description_embeddings", "concept_embeddings"):
        record = source[name]
        assert set(record) == {"path", "sha256", "source_texts_sha256", "embedding_space"}
        assert (manifest_path.parent / record["path"]).resolve().is_file()
    assert manifest["configuration"]["nl_target_source_manifest_path"] == str(
        receipt_path.resolve()
    )

    identity = json.loads(
        (manifest_path.parent / "_project_cell_resume_v1" / "identity.json").read_text(
            encoding="utf-8"
        )
    )["semantic_identity"]
    assert identity["configuration"]["project_methods"] == list(NL_PROJECT_METHODS)
    identity_source = identity["inputs"]["nl_target_source"]
    assert "path" not in identity_source["description_embeddings"]
    assert "path" not in identity_source["concept_embeddings"]
    assert identity_source["receipt_sha256"] == hashlib.sha256(
        receipt_path.read_bytes()
    ).hexdigest()
    assert identity_source["target_scores_sha256"] == driver.retrieval_targets_sha256(
        {
            concept_id: next(
                call[1].target_scores
                for call in observed["scalar_calls"]
                if call[0][0] == concept_id
            )
            for concept_id in range(2)
        }
    )


def test_nl_resume_identity_is_invariant_to_manifest_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    receipt_path = _enable_nl_receipt(kwargs)
    _install_fakes(monkeypatch, kwargs)
    monkeypatch.setattr(driver, "_repository_source_sha256", lambda: "f" * 64)

    driver.run_project_methods(**kwargs)
    first_manifest_path = Path(kwargs["manifest_path"])
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    first_identity = json.loads(
        (first_manifest_path.parent / "_project_cell_resume_v1" / "identity.json").read_text(
            encoding="utf-8"
        )
    )["semantic_identity"]

    second_manifest_path = tmp_path / "relocated" / "at" / "another-depth" / "manifest.json"
    relocated: dict[str, Any] = {
        **kwargs,
        "output_path": second_manifest_path.with_name("steering_data.parquet"),
        "manifest_path": second_manifest_path,
        "nl_target_source_manifest_path": receipt_path,
    }
    driver.run_project_methods(**relocated)
    second_manifest = json.loads(second_manifest_path.read_text(encoding="utf-8"))
    second_identity = json.loads(
        (second_manifest_path.parent / "_project_cell_resume_v1" / "identity.json").read_text(
            encoding="utf-8"
        )
    )["semantic_identity"]

    assert (
        first_manifest["methods"]["NLSteering"]["target_source"]["description_embeddings"]["path"]
        != second_manifest["methods"]["NLSteering"]["target_source"]["description_embeddings"][
            "path"
        ]
    )
    assert first_identity == second_identity


def test_nl_artifact_change_after_receipt_parse_fails_before_stores_or_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _enable_nl_receipt(kwargs)
    observed = _install_fakes(monkeypatch, kwargs)
    real_load = driver.load_nl_target_source_receipt

    def load_then_change(path: Path) -> Any:
        receipt = real_load(path)
        receipt.description_embeddings.path.write_bytes(
            receipt.description_embeddings.path.read_bytes() + b"changed-after-parse"
        )
        return receipt

    monkeypatch.setattr(driver, "load_nl_target_source_receipt", load_then_change)
    monkeypatch.setattr(
        driver,
        "CellResultStore",
        lambda *_args, **_kwargs: pytest.fail("NL preflight reached resume-store creation"),
    )

    with pytest.raises(ValueError, match="description embeddings SHA-256 mismatch"):
        driver.run_project_methods(**kwargs)

    assert observed["events"] == []
    assert not (Path(kwargs["manifest_path"]).parent / "_project_cell_resume_v1").exists()


@pytest.mark.parametrize(
    "input_name",
    ["receipt", "description_embeddings", "concept_embeddings"],
)
def test_force_rejects_hardlinked_nl_input_output_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    input_name: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    receipt_path = _enable_nl_receipt(kwargs)
    receipt = driver.load_nl_target_source_receipt(receipt_path)
    source_path = (
        receipt.path if input_name == "receipt" else getattr(receipt, input_name).path
    )
    output_path = Path(kwargs["output_path"])
    output_path.parent.mkdir(parents=True)
    output_path.hardlink_to(source_path)
    kwargs["force"] = True
    _install_fakes(monkeypatch, kwargs)

    with pytest.raises(ValueError, match="NL target inputs.*must be path-disjoint"):
        driver.run_project_methods(**kwargs)

    assert output_path.samefile(source_path)
    assert output_path.read_bytes() == source_path.read_bytes()
    assert not Path(kwargs["manifest_path"]).exists()


def test_nl_artifact_change_during_generation_blocks_canonical_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    receipt_path = _enable_nl_receipt(kwargs)
    _install_fakes(monkeypatch, kwargs)
    receipt = driver.load_nl_target_source_receipt(receipt_path)
    real_validate = driver.validate_steering_frame

    def validate_then_change(*args: Any, **named: Any) -> None:
        real_validate(*args, **named)
        if named.get("source") == "project":
            path = receipt.description_embeddings.path
            path.write_bytes(path.read_bytes() + b"changed-before-publication")

    monkeypatch.setattr(driver, "validate_steering_frame", validate_then_change)

    with pytest.raises(ValueError, match="description_embeddings SHA-256 mismatch"):
        driver.run_project_methods(**kwargs)

    assert not Path(kwargs["output_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


def test_valid_updated_nl_receipt_during_generation_blocks_canonical_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    receipt_path = _enable_nl_receipt(kwargs)
    _install_fakes(monkeypatch, kwargs)
    real_validate = driver.validate_steering_frame
    updated_receipts: list[Any] = []

    def validate_then_replace_source(*args: Any, **named: Any) -> None:
        real_validate(*args, **named)
        if named.get("source") != "project":
            return
        value = json.loads(receipt_path.read_text(encoding="utf-8"))
        description_path = receipt_path.parent / value["description_embeddings"]["path"]
        with np.load(description_path, allow_pickle=False) as payload:
            fids = payload["fids"].copy()
            embeddings = payload["emb"].copy()
        embeddings[0, 0] = np.float32(0.75)
        np.savez(description_path, fids=fids, emb=embeddings)
        value["description_embeddings"]["sha256"] = hashlib.sha256(
            description_path.read_bytes()
        ).hexdigest()
        receipt_path.write_text(json.dumps(value), encoding="utf-8")
        updated_receipts.append(driver.load_nl_target_source_receipt(receipt_path))

    monkeypatch.setattr(driver, "validate_steering_frame", validate_then_replace_source)

    with pytest.raises(ValueError, match="receipt identity changed during generation"):
        driver.run_project_methods(**kwargs)

    assert len(updated_receipts) == 1
    assert not Path(kwargs["output_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


def test_cross_rank_semantic_identity_requires_exact_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(driver.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(driver.dist, "get_world_size", lambda: 2)

    def divergent(output: list[str | None], value: str) -> None:
        output[:] = [value, "0" * 64]

    monkeypatch.setattr(driver.dist, "all_gather_object", divergent)

    with pytest.raises(ValueError, match="differs across distributed ranks"):
        driver._require_cross_rank_semantic_identity({"same": "locally"})


def test_invalid_checkpoint_geometry_precedes_resume_store_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)

    def reject_setup(*_args: Any, **_kwargs: Any) -> Any:
        raise ValueError("invalid turn checkpoint cadence")

    monkeypatch.setattr(driver, "load_method_setup", reject_setup)
    monkeypatch.setattr(
        driver,
        "CellResultStore",
        lambda *_args, **_kwargs: pytest.fail(
            "invalid checkpoint geometry reached resume-store construction"
        ),
    )

    with pytest.raises(ValueError, match="invalid turn checkpoint cadence"):
        driver.run_project_methods(**kwargs)

    assert not (Path(kwargs["manifest_path"]).parent / "_project_cell_resume_v1").exists()


def test_driver_assigns_complete_grid_through_project_prompt_group_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)
    ownership_calls: list[tuple[tuple[Any, ...], int, int]] = []
    original_owner = driver.owned_project_cells

    def record_owner(
        cells: tuple[Any, ...],
        *,
        rank: int,
        world_size: int,
        methods: tuple[str, ...] = driver.PROJECT_METHODS,
    ) -> tuple[Any, ...]:
        ownership_calls.append((cells, rank, world_size))
        return original_owner(cells, rank=rank, world_size=world_size, methods=methods)

    monkeypatch.setattr(driver, "owned_project_cells", record_owner)

    result = driver.run_project_methods(**kwargs)

    assert result.shape == (240, 18)
    assert len(ownership_calls) == 1
    cells, rank, world_size = ownership_calls[0]
    assert (rank, world_size) == (0, 1)
    assert cells == driver.build_project_cell_grid(_main_frame(), seed=42)
    assert len(cells) == 480
    assert observed["scalar_calls"] == []
    assert len(observed["batch_calls"]) == 20
    assert {len(requests) for _pair, requests in observed["batch_calls"]} == {24}


def test_turn_mode_manifest_records_both_token_currencies_and_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)
    setup = observed["setup"]
    setup.segment_mode = driver.SegmentMode.turn_delimited_fixed_tokens
    setup.sae_model = "gemma-3-4b-pt"
    setup.sae_layer = 22
    setup.prompt_segment_size = 17
    setup.prompt_context_window = 333
    setup.turn_self_cond_segment_size_rustbpe = 37
    setup.prompt_statistics_raw_sha256 = "a" * 64
    setup.prompt_selection_context = turn_prompt_selection_context()

    driver.run_project_methods(**kwargs)

    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["generation"] == {"temperature": 1.0, "top_k": 50}
    assert manifest["self_conditioning_segment_geometry"] == EXPECTED_TURN_GEOMETRY
    ordinary = manifest["methods"]["SelfConditioning"]
    assert ordinary["prompt_features"] == (
        "real context-independent turn-delimited gemma-3-4b-pt l22 SAE active/"
        "true-inactive features from checkpoint-recorded online-SAE lineage"
    )
    assert ordinary["completion_slots"] == (
        "marker_anchored_predicted_slots_refreshed_after_each_complete_"
        "checkpoint_rustbpe_cadence_window"
    )
    assert "fixed-128" not in ordinary["prompt_features"]
    assert ordinary["completion_slots"] != "one_predicted_slot_for_complete_continuation"


def test_fixed_mode_manifest_geometry_is_a_noop_without_turn_fields() -> None:
    # kills: adding a SegmentMode member without deciding what the AxBench
    # manifest publishes for it. `_turn_self_conditioning_segment_geometry`
    # handles exactly these two and raises "unsupported AxBench prompt segment
    # mode" for anything else, so that raise is unreachable only while this
    # assertion holds -- and returning a record instead of None for the
    # fixed-token member, which would splice a turn block into every legacy
    # manifest.
    assert set(driver.SegmentMode) == {
        driver.SegmentMode.fixed_tokens,
        driver.SegmentMode.turn_delimited_fixed_tokens,
    }
    setup: Any = SimpleNamespace(segment_mode=driver.SegmentMode.fixed_tokens)

    assert driver._turn_self_conditioning_segment_geometry(setup) is None


def test_turn_manifest_geometry_requires_the_prompt_selector() -> None:
    # kills: reading the published policy/seed off anything but the selector the
    # run actually conditioned with -- a `getattr(..., None)` or a null default
    # would advertise provenance no prompt was built from.
    setup: Any = SimpleNamespace(
        segment_mode=driver.SegmentMode.turn_delimited_fixed_tokens,
        prompt_selection_context=None,
    )

    with pytest.raises(ValueError, match=r"turn-delimited AxBench setup has no prompt selector"):
        driver._turn_self_conditioning_segment_geometry(setup)


def test_project_fit_phase_has_one_exact_owned_concept_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    observed: list[tuple[str, dict[str, Any]]] = []

    @contextmanager
    def record_region(name: str, **attrs: Any) -> Any:
        observed.append((name, attrs))
        yield

    monkeypatch.setattr(driver, "profile_region", record_region)

    driver.run_project_methods(**kwargs)

    assert [item for item in observed if item[0] == "axbench_upstream.project.concept_fits"] == [
        ("axbench_upstream.project.concept_fits", {"owned_concepts": 2})
    ]


def test_driver_default_factor_six_trace_keeps_prompt_group_scalar_in_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)
    selectors = benchmark._default_trace_selectors("project")
    assert len(selectors) == 1
    selector = selectors[0]
    assert selector.factor_hex == (6.0).hex()

    monkeypatch.setattr(driver, "cell_trace_requested", selector.matches)

    driver.run_project_methods(**kwargs)

    expected_cells = tuple(
        cell
        for cell in driver.build_project_cell_grid(_main_frame(), seed=42)
        if (cell.concept_id, cell.input_id) == (0, 0)
    )
    assert [index for index, cell in enumerate(expected_cells) if selector.matches(cell)] == [22]
    assert len(observed["scalar_calls"]) == len(expected_cells) == 24
    for (pair, policy, selection_seed, token_seed), cell in zip(
        observed["scalar_calls"],
        expected_cells,
        strict=True,
    ):
        assert pair == (0, 0)
        assert selection_seed == cell.selection_seed
        assert token_seed == cell.token_seed
        expected_type = (
            SelfCondScoreInterpolation
            if cell.method == "SelfConditioning"
            else SelfCondScoreMasking
        )
        assert isinstance(policy, expected_type)
    assert all(pair != (0, 0) for pair, _requests in observed["batch_calls"])
    assert len(observed["batch_calls"]) == 19
    assert {len(requests) for _pair, requests in observed["batch_calls"]} == {24}


@pytest.mark.parametrize(
    "selector_offsets",
    [
        pytest.param((11,), id="middle"),
        pytest.param((23,), id="last"),
        pytest.param((5, 18), id="multiple"),
    ],
)
def test_any_trace_selector_keeps_whole_group_scalar_and_unrelated_group_batched(
    selector_offsets: tuple[int, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(frame, seed=42)[:48]
    first_group = tuple(cell for cell in cells if (cell.concept_id, cell.input_id) == (0, 0))
    second_group = tuple(cell for cell in cells if (cell.concept_id, cell.input_id) == (0, 1))
    selected = {first_group[offset] for offset in selector_offsets}
    scalar_calls: list[Any] = []
    batch_calls: list[tuple[Any, ...]] = []
    appended: list[tuple[Any, str]] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return cells

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    def scalar(cell: Any, **_kwargs: Any) -> str:
        scalar_calls.append(cell)
        return f"scalar:{cell.method}:{cell.factor_hex}"

    def batch(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        frozen = tuple(group)
        batch_calls.append(frozen)
        return tuple(f"batch:{cell.method}:{cell.factor_hex}" for cell in frozen)

    monkeypatch.setattr(driver, "cell_trace_requested", selected.__contains__)
    monkeypatch.setattr(driver, "_generate_project_cell", scalar)
    monkeypatch.setattr(driver, "_generate_project_cells", batch)

    driver._append_missing_project_cells(
        Store(),  # ty: ignore[invalid-argument-type]
        cells,
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        setup=cast(Any, _dispatch_setup()),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    assert scalar_calls == list(first_group)
    assert batch_calls == [second_group]
    assert appended == [
        *[(cell, f"scalar:{cell.method}:{cell.factor_hex}") for cell in first_group],
        *[(cell, f"batch:{cell.method}:{cell.factor_hex}") for cell in second_group],
    ]


def test_resumed_singleton_uses_scalar_fallback_without_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(frame, seed=42)
    remaining = (cells[0],)
    appended: list[tuple[Any, str]] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells[:24]
            return remaining

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(
        driver,
        "_generate_project_cell",
        lambda cell, **_kwargs: f"scalar:{cell.method}:{cell.factor_hex}",
    )
    monkeypatch.setattr(
        driver,
        "_generate_project_cells",
        lambda *_args, **_kwargs: pytest.fail("resumed singleton reached batch generation"),
    )

    driver._append_missing_project_cells(
        Store(),  # ty: ignore[invalid-argument-type]
        cells[:24],
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        setup=cast(Any, _dispatch_setup()),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    assert appended == [(cells[0], f"scalar:{cells[0].method}:{cells[0].factor_hex}")]


@pytest.mark.parametrize(
    ("segment_mode", "expected_dispatch"),
    (
        pytest.param(driver.SegmentMode.fixed_tokens, "batch", id="fixed-batches"),
        pytest.param(
            driver.SegmentMode.turn_delimited_fixed_tokens,
            "scalar",
            id="turn-falls-back-to-scalar",
        ),
    ),
)
def test_project_batch_dispatch_is_limited_to_fixed_token_mode(
    monkeypatch: pytest.MonkeyPatch,
    segment_mode: driver.SegmentMode,
    expected_dispatch: str,
) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(frame, seed=42)[:2]
    dispatches: list[tuple[str, tuple[Any, ...]]] = []
    appended: list[tuple[Any, str]] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return cells

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    def scalar(cell: Any, **_kwargs: Any) -> str:
        dispatches.append(("scalar", (cell,)))
        return f"generated:{cell.selection_seed}:{cell.token_seed}"

    def batch(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        grouped = tuple(group)
        dispatches.append(("batch", grouped))
        return tuple(f"generated:{cell.selection_seed}:{cell.token_seed}" for cell in grouped)

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(driver, "_generate_project_cell", scalar)
    monkeypatch.setattr(driver, "_generate_project_cells", batch)

    driver._append_missing_project_cells(
        Store(),  # ty: ignore[invalid-argument-type]
        cells,
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        setup=cast(Any, _dispatch_setup(segment_mode=segment_mode)),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    expected_dispatches = (
        [("batch", cells)]
        if expected_dispatch == "batch"
        else [("scalar", (cell,)) for cell in cells]
    )
    assert dispatches == expected_dispatches
    assert appended == [
        (cell, f"generated:{cell.selection_seed}:{cell.token_seed}") for cell in cells
    ]


@pytest.mark.parametrize("methods", [driver.PROJECT_METHODS, NL_PROJECT_METHODS])
@pytest.mark.parametrize(
    ("segment_mode", "trace_old"),
    [
        (driver.SegmentMode.turn_delimited_fixed_tokens, False),
        (driver.SegmentMode.fixed_tokens, True),
    ],
)
def test_interleaved_scalar_paths_preserve_exact_missing_cell_order(
    monkeypatch: pytest.MonkeyPatch,
    methods: tuple[str, ...],
    segment_mode: driver.SegmentMode,
    trace_old: bool,
) -> None:
    base = _main_frame(max_concepts=1)
    frame = base.iloc[[0, len(FULL_FACTORS), 1, len(FULL_FACTORS) + 1]].reset_index(
        drop=True
    )
    cells = driver.build_project_cell_grid(frame, seed=42, methods=methods)
    scalar_calls: list[Any] = []
    appended: list[Any] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return cells

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    def scalar(cell: Any, **_kwargs: Any) -> str:
        scalar_calls.append(cell)
        return f"scalar:{cell.source_row_position}:{cell.method}"

    monkeypatch.setattr(
        driver,
        "cell_trace_requested",
        lambda cell: trace_old and cell.method in driver.PROJECT_METHODS,
    )
    monkeypatch.setattr(driver, "_generate_project_cell", scalar)
    monkeypatch.setattr(
        driver,
        "_generate_project_cells",
        lambda *_args, **_kwargs: pytest.fail("scalar path reached batch generation"),
    )

    driver._append_missing_project_cells(
        Store(),  # ty: ignore[invalid-argument-type]
        cells,
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        nl_targets={},
        setup=cast(Any, _dispatch_setup(segment_mode=segment_mode)),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    assert tuple(scalar_calls) == cells
    assert tuple(cell for cell, _generation in appended) == cells


def test_resumed_one_old_plus_nl_cells_stay_scalar_in_missing_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1).iloc[:1].copy()
    cells = driver.build_project_cell_grid(frame, seed=42, methods=NL_PROJECT_METHODS)
    remaining = (cells[1], cells[2])
    appended: list[Any] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return remaining

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(
        driver,
        "_generate_project_cell",
        lambda cell, **_kwargs: f"scalar:{cell.method}",
    )
    monkeypatch.setattr(
        driver,
        "_generate_project_cells",
        lambda *_args, **_kwargs: pytest.fail("singleton old cohort reached batch generation"),
    )

    driver._append_missing_project_cells(
        Store(),  # ty: ignore[invalid-argument-type]
        cells,
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        nl_targets={},
        setup=cast(Any, _dispatch_setup()),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    assert tuple(cell for cell, _generation in appended) == remaining


def test_nl_cells_remain_scalar_without_changing_old_fixed_batch_cohort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1).iloc[:2].copy()
    cells = driver.build_project_cell_grid(frame, seed=42, methods=NL_PROJECT_METHODS)
    batches: list[tuple[Any, ...]] = []
    scalars: list[Any] = []
    appended: list[Any] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return cells

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    def batch(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        grouped = tuple(group)
        batches.append(grouped)
        return tuple(f"old:{cell.method}:{cell.factor_hex}" for cell in grouped)

    def scalar(cell: Any, **_kwargs: Any) -> str:
        scalars.append(cell)
        return f"scalar:{cell.method}:{cell.factor_hex}"

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(driver, "_generate_project_cells", batch)
    monkeypatch.setattr(driver, "_generate_project_cell", scalar)

    driver._append_missing_project_cells(
        Store(),  # ty: ignore[invalid-argument-type]
        cells,
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        nl_targets={},
        setup=cast(Any, _dispatch_setup()),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    expected_old = tuple(cell for cell in cells if cell.method in driver.PROJECT_METHODS)
    expected_nl = tuple(cell for cell in cells if cell.method == "NLSteering")
    assert batches == [expected_old]
    assert tuple(scalars) == expected_nl
    assert tuple(cell for cell, _generation in appended) == (*expected_old, *expected_nl)


def test_project_batch_rejects_singleton_before_lookup() -> None:
    cell = driver.build_project_cell_grid(_main_frame(max_concepts=1), seed=42)[0]

    with pytest.raises(ValueError, match="requires at least two cells"):
        driver._generate_project_cells(
            (cell,),
            rows=(),
            rendered_tokens_by_pair={},
            prepared_by_tokens={},
            dense_diffmeans={},
            setup=cast(Any, _dispatch_setup()),
            model=object(),
            tokenizer=object(),
            max_new_tokens=128,
        )


def test_project_batch_rejects_nl_cells_before_lookup() -> None:
    cells = driver.build_project_cell_grid(
        _main_frame(max_concepts=1),
        seed=42,
        methods=NL_PROJECT_METHODS,
    )
    nl_cells = tuple(cell for cell in cells if cell.method == "NLSteering")[:2]

    with pytest.raises(ValueError, match="NLSteering cells use the scalar generation path"):
        driver._generate_project_cells(
            nl_cells,
            rows=(),
            rendered_tokens_by_pair={},
            prepared_by_tokens={},
            dense_diffmeans={},
            setup=cast(Any, _dispatch_setup()),
            model=object(),
            tokenizer=object(),
            max_new_tokens=128,
        )


def test_project_batch_rejects_mixed_prompt_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(frame, seed=42)
    first = cells[0]
    second = next(cell for cell in cells if cell.input_id != first.input_id)
    monkeypatch.setattr(driver, "_project_score_policy", lambda *_args, **_kwargs: object())

    with pytest.raises(ValueError, match="must share one concept/input pair"):
        driver._generate_project_cells(
            (first, second),
            rows=tuple(frame.itertuples(index=False)),
            rendered_tokens_by_pair={(first.concept_id, first.input_id): ()},
            prepared_by_tokens={},
            dense_diffmeans={first.concept_id: t.zeros(1)},
            setup=cast(Any, _dispatch_setup()),
            model=object(),
            tokenizer=object(),
            max_new_tokens=128,
        )


def test_project_score_policy_rejects_noncanonical_method() -> None:
    cell = driver.build_project_cell_grid(_main_frame(max_concepts=1), seed=42)[0]

    with pytest.raises(AssertionError, match="unsupported project method"):
        driver._project_score_policy(
            replace(cell, method="UnknownProjectMethod"),
            setup=cast(Any, _dispatch_setup()),
            diffmean=t.zeros(1),
            factor=1.0,
        )


@pytest.mark.parametrize("factor", [0.0, 2.0, 6.0])
def test_nl_score_policy_preserves_raw_interpolation_math_and_target(factor: float) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(
        frame,
        seed=42,
        methods=NL_PROJECT_METHODS,
    )
    cell = next(
        cell
        for cell in cells
        if cell.method == "NLSteering" and float.fromhex(cell.factor_hex) == factor
    )
    target = t.tensor([-1.0, -0.0, 0.25, 1.0], dtype=t.float32)

    policy = driver._project_score_policy(
        cell,
        setup=cast(
            Any,
            SimpleNamespace(
                output_negative_selection=NegativeSamplingSelection(
                    distribution=Distribution.geometric,
                    mean=1.0,
                    std=1.0,
                    min_k=0,
                    max_k=2,
                )
            ),
        ),
        diffmean=t.full_like(target, 0.75),
        factor=factor,
        nl_target=target,
    )

    assert isinstance(policy, SelfCondScoreInterpolation)
    assert t.equal(policy.target_scores, target)
    assert float(policy.weight) == factor / 6.0
    assert t.equal(target, t.tensor([-1.0, -0.0, 0.25, 1.0], dtype=t.float32))


def test_nl_score_policy_requires_prevalidated_concept_target() -> None:
    cell = next(
        cell
        for cell in driver.build_project_cell_grid(
            _main_frame(max_concepts=1),
            seed=42,
            methods=NL_PROJECT_METHODS,
        )
        if cell.method == "NLSteering"
    )

    with pytest.raises(ValueError, match="no validated retrieval target"):
        driver._project_score_policy(
            cell,
            setup=cast(Any, _dispatch_setup()),
            diffmean=t.zeros(1),
            factor=0.0,
        )


def test_project_batch_cardinality_failure_publishes_no_partial_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(frame, seed=42)[:2]
    appended: list[tuple[Any, str]] = []

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return cells

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(
        driver,
        "_generate_project_cells",
        lambda *_args, **_kwargs: ("only-one",),
    )

    with pytest.raises(RuntimeError, match="wrong number of generations"):
        driver._append_missing_project_cells(
            Store(),  # ty: ignore[invalid-argument-type]
            cells,
            rows=tuple(frame.itertuples(index=False)),
            rendered_tokens_by_pair={},
            prepared_by_tokens={},
            dense_diffmeans={},
            setup=cast(Any, _dispatch_setup()),
            model=object(),
            tokenizer=object(),
            max_new_tokens=128,
        )
    assert appended == []


def test_interleaved_partial_resume_batches_each_missing_pair_and_reconstructs_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = _main_frame(max_concepts=1)
    frame = (
        base.loc[base["input_id"].isin((0, 1, 2))]
        .sort_values(["factor", "input_id"], kind="stable")
        .reset_index(drop=True)
    )
    cells = driver.build_project_cell_grid(frame, seed=42)
    by_pair = {
        pair: tuple(cell for cell in cells if (cell.concept_id, cell.input_id) == pair)
        for pair in ((0, 0), (0, 1), (0, 2))
    }
    selected = (
        {by_pair[(0, 0)][index] for index in (1, 4, 9, 22)}
        | {by_pair[(0, 1)][index] for index in (0, 7, 12)}
        | {by_pair[(0, 2)][5]}
    )
    remaining = tuple(cell for cell in cells if cell in selected)
    batch_calls: list[tuple[Any, ...]] = []
    scalar_calls: list[Any] = []

    class Store:
        def __init__(self) -> None:
            self.results = {
                cell: f"existing:{index}"
                for index, cell in enumerate(cells)
                if cell not in selected
            }

        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            return tuple(cell for cell in requested if cell not in self.results)

        def append(self, cell: Any, generation: str) -> None:
            assert cell not in self.results
            self.results[cell] = generation

    store = Store()

    def batch(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        frozen = tuple(group)
        batch_calls.append(frozen)
        return tuple(f"resumed:{cell.token_seed}" for cell in frozen)

    def scalar(cell: Any, **_kwargs: Any) -> str:
        scalar_calls.append(cell)
        return f"resumed:{cell.token_seed}"

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(driver, "_generate_project_cells", batch)
    monkeypatch.setattr(driver, "_generate_project_cell", scalar)

    driver._append_missing_project_cells(
        store,  # ty: ignore[invalid-argument-type]
        cells,
        rows=tuple(frame.itertuples(index=False)),
        rendered_tokens_by_pair={},
        prepared_by_tokens={},
        dense_diffmeans={},
        setup=cast(Any, _dispatch_setup()),
        model=object(),
        tokenizer=object(),
        max_new_tokens=128,
    )

    assert batch_calls == [
        tuple(cell for cell in remaining if (cell.concept_id, cell.input_id) == (0, 0)),
        tuple(cell for cell in remaining if (cell.concept_id, cell.input_id) == (0, 1)),
    ]
    assert scalar_calls == [by_pair[(0, 2)][5]]
    assert len(store.results) == len(cells)
    assert tuple(store.results[cell] for cell in cells) == tuple(
        (f"resumed:{cell.token_seed}" if cell in selected else f"existing:{index}")
        for index, cell in enumerate(cells)
    )


def test_project_batch_failure_publishes_nothing_and_retry_is_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = _main_frame(max_concepts=1)
    cells = driver.build_project_cell_grid(frame, seed=42)[:4]
    appended: list[tuple[Any, str]] = []
    attempts = 0

    class Store:
        def missing(self, requested: Any) -> tuple[Any, ...]:
            assert tuple(requested) == cells
            completed = {cell for cell, _generation in appended}
            return tuple(cell for cell in requested if cell not in completed)

        def append(self, cell: Any, generation: str) -> None:
            appended.append((cell, generation))

    def generate(group: Any, **_kwargs: Any) -> tuple[str, ...]:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected batched decode failure")
        return tuple(f"retry:{cell.token_seed}" for cell in group)

    monkeypatch.setattr(driver, "cell_trace_requested", lambda _cell: False)
    monkeypatch.setattr(driver, "_generate_project_cells", generate)

    store = Store()
    call_kwargs: dict[str, Any] = {
        "rows": tuple(frame.itertuples(index=False)),
        "rendered_tokens_by_pair": {},
        "prepared_by_tokens": {},
        "dense_diffmeans": {},
        "setup": cast(Any, _dispatch_setup()),
        "model": object(),
        "tokenizer": object(),
        "max_new_tokens": 128,
    }
    with pytest.raises(RuntimeError, match="injected batched decode failure"):
        driver._append_missing_project_cells(
            store,  # ty: ignore[invalid-argument-type]
            cells,
            **call_kwargs,
        )
    assert appended == []

    driver._append_missing_project_cells(
        store,  # ty: ignore[invalid-argument-type]
        cells,
        **call_kwargs,
    )

    assert attempts == 2
    assert appended == [(cell, f"retry:{cell.token_seed}") for cell in cells]


def test_extraction_objects_are_released_before_target_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)

    class ExtractionObject:
        pass

    model_ref: weakref.ReferenceType[ExtractionObject] | None = None
    sae_ref: weakref.ReferenceType[ExtractionObject] | None = None

    def load_extraction(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        nonlocal model_ref, sae_ref
        model = ExtractionObject()
        sae = ExtractionObject()
        model_ref = weakref.ref(model)
        sae_ref = weakref.ref(sae)
        return model, sae

    def load_target(_setup: Any, _device: str) -> tuple[Any, Any]:
        assert model_ref is not None and model_ref() is None
        assert sae_ref is not None and sae_ref() is None
        return SimpleNamespace(dec_config=SimpleNamespace(sequence_len=512)), object()

    monkeypatch.setattr(driver, "load_extraction_stack", load_extraction)
    monkeypatch.setattr(driver, "load_target_model", load_target)

    driver.run_project_methods(**kwargs)


def test_source_row_order_is_restored_after_first_concept_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    frame = pd.read_parquet(kwargs["main_path"])
    shuffled = frame.sample(frac=1.0, random_state=4).reset_index(drop=True)
    shuffled.to_parquet(kwargs["main_path"], index=False)
    _install_fakes(monkeypatch, kwargs)

    result = driver.run_project_methods(**kwargs)

    assert result.loc[:, list(INPUT_COLUMNS)].equals(shuffled.loc[:, list(INPUT_COLUMNS)])


def test_model_free_project_input_needs_no_baseline_columns(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    project_input = tmp_path / "project_input.parquet"
    project_input_manifest = tmp_path / "project_input_manifest.json"
    write_project_input(
        prepared_dir=kwargs["prepared_dir"],
        output_path=project_input,
        manifest_path=project_input_manifest,
        max_concepts=kwargs["max_concepts"],
    )
    kwargs["main_path"] = None
    kwargs["project_input_path"] = project_input
    kwargs["project_input_manifest_path"] = project_input_manifest
    observed = _install_fakes(monkeypatch, kwargs)
    prompt_ids: dict[str, int] = {}

    def render_project_prompt(_tokenizer: Any, prompt: str) -> Any:
        prompt_id = prompt_ids.setdefault(prompt, len(prompt_ids))
        return SimpleNamespace(token_ids=[1, 10, 100 + prompt_id, 2], assistant_end_id=9)

    monkeypatch.setattr(driver, "render_prompt_for_chat_completion", render_project_prompt)

    result = driver.run_project_methods(**kwargs)

    assert tuple(result.columns[: len(INPUT_COLUMNS)]) == INPUT_COLUMNS
    assert "DiffMean_steered_generation" not in result.columns
    assert observed["scalar_calls"] == []
    assert len(observed["batch_calls"]) == 20
    assert [len(requests) for _pair, requests in observed["batch_calls"]] == [24] * 20
    manifest = json.loads(Path(kwargs["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 4
    assert manifest["inputs"]["project_source_kind"] == "project_input"
    assert "project_source_manifest_sha256" in manifest["inputs"]
    assert manifest["configuration"]["project_input_path"] == str(project_input.resolve())
    assert manifest["configuration"]["project_input_manifest_path"] == str(
        project_input_manifest.resolve()
    )
    identity = json.loads(
        (Path(kwargs["manifest_path"]).parent / "_project_cell_resume_v1" / "identity.json")
        .read_text(encoding="utf-8")
    )["semantic_identity"]
    assert identity["schema_version"] == 2
    assert identity["inputs"]["source_manifest_sha256"] == hashlib.sha256(
        project_input_manifest.read_bytes()
    ).hexdigest()


@pytest.mark.parametrize(
    ("main_enabled", "project_enabled"),
    [(False, False), (True, True)],
)
def test_project_input_source_is_exactly_one_of_legacy_or_model_free(
    tmp_path: Path,
    main_enabled: bool,
    project_enabled: bool,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["project_input_path"] = kwargs["main_path"] if project_enabled else None
    kwargs["project_input_manifest_path"] = (
        tmp_path / "project_input_manifest.json" if project_enabled else None
    )
    if not main_enabled:
        kwargs["main_path"] = None

    with pytest.raises(ValueError, match="exactly one of main_path or project_input_path"):
        driver.run_project_methods(**kwargs)


@pytest.mark.parametrize(
    ("main_enabled", "project_enabled", "manifest_enabled", "expected"),
    [
        (False, True, False, "manifest_path is required"),
        (True, False, True, "manifest_path requires project_input_path"),
    ],
)
def test_project_input_manifest_presence_matches_the_selected_source(
    tmp_path: Path,
    main_enabled: bool,
    project_enabled: bool,
    manifest_enabled: bool,
    expected: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    project_path = kwargs["main_path"]
    kwargs["main_path"] = project_path if main_enabled else None
    kwargs["project_input_path"] = project_path if project_enabled else None
    kwargs["project_input_manifest_path"] = (
        tmp_path / "project-input-manifest.json" if manifest_enabled else None
    )

    with pytest.raises(ValueError, match=expected):
        driver.run_project_methods(**kwargs)


def test_shuffled_source_prepares_unique_prompts_in_canonical_pair_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    frame = pd.read_parquet(kwargs["main_path"])
    frame.sample(frac=1.0, random_state=19).reset_index(drop=True).to_parquet(
        kwargs["main_path"],
        index=False,
    )
    observed = _install_fakes(monkeypatch, kwargs)

    driver.run_project_methods(**kwargs)

    assert observed["prepared_tokens"] == [
        (1, 10 + concept_id, 100 + input_id, 2) for concept_id in range(2) for input_id in range(10)
    ]


def test_exact_prompt_token_twins_are_prepared_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)
    monkeypatch.setattr(
        driver,
        "render_prompt_for_chat_completion",
        lambda *_args: SimpleNamespace(token_ids=[1, 10, 100, 2], assistant_end_id=9),
    )

    driver.run_project_methods(**kwargs)

    assert observed["prepared_tokens"] == [(1, 10, 100, 2)]


def test_metadata_ids_must_match_jsonl_positions(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.jsonl"
    metadata_path.write_text(
        "\n".join(
            (
                json.dumps({"concept_id": 0, "concept": "zero"}),
                json.dumps({"concept_id": 7, "concept": "one"}),
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="concept_id must match JSONL position 1"):
        load_metadata(metadata_path)


def test_empty_rendered_prompt_fails_before_model_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    monkeypatch.setattr(
        driver,
        "render_prompt_for_chat_completion",
        lambda *_args: SimpleNamespace(token_ids=[], assistant_end_id=9),
    )
    monkeypatch.setattr(driver, "get_tokenizer", lambda: object())
    monkeypatch.setattr(
        driver,
        "_tokenizer_artifact_identity",
        lambda: {"name": "tokenizer.pkl", "byte_length": 1, "sha256": "a" * 64},
    )
    monkeypatch.setattr(driver, "select_training_texts", lambda *_args, **_kwargs: _training())
    setup_calls = 0

    def forbidden_setup(*_args: Any) -> Any:
        nonlocal setup_calls
        setup_calls += 1
        raise AssertionError("empty prompt reached model setup")

    monkeypatch.setattr(driver, "load_method_setup", forbidden_setup)

    with pytest.raises(ValueError, match="token tuple must be nonempty"):
        driver.run_project_methods(**kwargs)
    assert setup_calls == 0


@pytest.mark.parametrize(
    ("body", "error"),
    [
        ("[]", "expected a JSON object"),
        ('{"key": 1, "key": 2}', "strict JSON.*duplicate JSON object key"),
        ('{"value": NaN}', "strict JSON.*non-finite JSON constant"),
    ],
)
def test_json_object_loader_rejects_non_strict_objects(
    tmp_path: Path,
    body: str,
    error: str,
) -> None:
    path = tmp_path / "invalid.json"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=error):
        driver._json_object(path)


def test_repository_source_digest_detects_same_path_content_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tracked_diff = b"first version"

    def check_output(args: list[str], **_kwargs: Any) -> bytes:
        if args[1] == "diff":
            return tracked_diff
        if args[1] == "ls-files":
            return b""
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    first = driver._repository_source_sha256()
    tracked_diff = b"second version"
    second = driver._repository_source_sha256()

    assert len(first) == len(second) == 64
    assert first != second


def test_repository_source_digest_binds_untracked_file_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "new_source.py"
    source.write_bytes(b"version-a")

    def check_output(args: list[str], **_kwargs: Any) -> bytes:
        if args[1] == "diff":
            return b""
        if args[1] == "ls-files":
            return b"new_source.py\0"
        raise AssertionError(f"unexpected git command: {args}")

    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    first = driver._repository_source_sha256()
    source.write_bytes(b"version-b")

    assert driver._repository_source_sha256() != first


def test_repository_source_digest_binds_untracked_symlink_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    link = tmp_path / "new_link.py"
    link.symlink_to("target-a.py")

    def check_output(args: list[str], **_kwargs: Any) -> bytes:
        return b"" if args[1] == "diff" else b"new_link.py\0"

    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    first = driver._repository_source_sha256()
    link.unlink()
    link.symlink_to("target-b.py")

    assert driver._repository_source_sha256() != first


def test_repository_source_digest_is_stable_under_untracked_listing_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.py").write_bytes(b"a")
    (tmp_path / "b.py").write_bytes(b"b")
    listing = b"b.py\0a.py\0"

    def check_output(args: list[str], **_kwargs: Any) -> bytes:
        return b"" if args[1] == "diff" else listing

    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    first = driver._repository_source_sha256()
    listing = b"a.py\0b.py\0"

    assert driver._repository_source_sha256() == first


def test_repository_source_digest_rejects_untracked_nonfile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "directory.py").mkdir()

    def check_output(args: list[str], **_kwargs: Any) -> bytes:
        return b"" if args[1] == "diff" else b"directory.py\0"

    monkeypatch.setattr(driver, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(driver.subprocess, "check_output", check_output)
    with pytest.raises(ValueError, match="not a regular file"):
        driver._repository_source_sha256()


def test_runtime_factor_grid_check_is_independent_of_frame_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    monkeypatch.setattr(driver, "SELF_CONDITIONING_FACTORS", FULL_FACTORS[:-1])
    with pytest.raises(ValueError, match="not in the declared factor grid"):
        driver.run_project_methods(**kwargs)


def test_rendered_tokens_must_be_invariant_across_factors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    calls = 0

    def render(_tokenizer: Any, _prompt: str) -> Any:
        nonlocal calls
        calls += 1
        return SimpleNamespace(token_ids=[1, 2, calls == 2], assistant_end_id=9)

    monkeypatch.setattr(driver, "render_prompt_for_chat_completion", render)
    with pytest.raises(ValueError, match="token tuple varies across factors"):
        driver.run_project_methods(**kwargs)


def test_rendered_assistant_end_must_be_invariant_across_factors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    calls = 0

    def render(_tokenizer: Any, _prompt: str) -> Any:
        nonlocal calls
        calls += 1
        return SimpleNamespace(token_ids=[1, 2, 3], assistant_end_id=9 + int(calls == 2))

    monkeypatch.setattr(driver, "render_prompt_for_chat_completion", render)
    with pytest.raises(ValueError, match="assistant-end ID varies across factors"):
        driver.run_project_methods(**kwargs)


def test_deduplicated_token_twins_require_one_assistant_end_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)

    def render(_tokenizer: Any, prompt: str) -> Any:
        _concept_id, input_id = (int(value) for value in prompt.removeprefix("prompt ").split("/"))
        return SimpleNamespace(token_ids=[1, 2, 3], assistant_end_id=9 + input_id)

    monkeypatch.setattr(driver, "render_prompt_for_chat_completion", render)
    with pytest.raises(ValueError, match="inconsistent assistant-end IDs"):
        driver.run_project_methods(**kwargs)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("overlength", "exceeds decoder sequence length"),
        ("empty_segments", "no real feature segments"),
        ("duplicate_features", "duplicate feature IDs"),
    ],
)
def test_prepared_context_structure_is_validated_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    match: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    if mutation == "overlength":
        monkeypatch.setattr(
            driver,
            "load_target_model",
            lambda *_args: (
                SimpleNamespace(dec_config=SimpleNamespace(sequence_len=100)),
                object(),
            ),
        )
    else:
        segments = ()
        if mutation == "duplicate_features":
            segments = (
                ContextFeatureSegment(
                    feature_ids=(1, 1),
                    buckets=(0, 1),
                    char_start=0,
                    char_end=4,
                ),
            )

        monkeypatch.setattr(
            driver,
            "prepare_prompt_context",
            lambda *_args, token_ids, assistant_end_id, **_kwargs: driver.PreparedPromptContext(
                token_ids=tuple(token_ids),
                assistant_end_id=assistant_end_id,
                segments=segments,
            ),
        )
    with pytest.raises(ValueError, match=match):
        driver.run_project_methods(**kwargs)


def test_prepared_concepts_must_match_main_frame_before_model_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    training = _training()
    first = training[0]
    training[0] = ConceptTrainingTexts(
        concept_id=first.concept_id,
        concept="different-concept",
        genre=first.genre,
        positive=first.positive,
        negative=first.negative,
    )
    monkeypatch.setattr(driver, "select_training_texts", lambda *_args, **_kwargs: training)
    monkeypatch.setattr(
        driver,
        "load_method_setup",
        lambda *_args: pytest.fail("concept drift reached model setup"),
    )

    with pytest.raises(ValueError, match="prepared metadata concept does not match"):
        driver.run_project_methods(**kwargs)


def test_main_intake_ignores_baseline_values_but_preserves_input_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    main = pd.read_parquet(kwargs["main_path"])
    main["DiffMean_steered_generation"] = None
    main["LinearProbe_strength"] = float("inf")
    main.to_parquet(kwargs["main_path"], index=False)
    _install_fakes(monkeypatch, kwargs)

    result = driver.run_project_methods(**kwargs)

    assert result.loc[:, list(INPUT_COLUMNS)].equals(main.loc[:, list(INPUT_COLUMNS)])
    assert result["SelfConditioning_steered_generation"].notna().all()


def test_first_concept_collapse_stops_before_later_concepts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs, generation=" ")

    with pytest.raises(ValueError, match="concept_id=0.*collapse detected"):
        driver.run_project_methods(**kwargs)

    assert set(observed["selection_seeds"]) == {(0, input_id) for input_id in range(10)}
    assert not Path(kwargs["output_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


@pytest.mark.parametrize(
    "identity_input",
    [
        "model_000007.pt",
        "prepared/generate/metadata.jsonl",
        "prepared/generate/train_data.parquet",
    ],
)
def test_resume_identity_rejects_mutated_scientific_input_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_input: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    driver.run_project_methods(**kwargs)
    path = tmp_path / identity_input
    original = path.read_bytes()
    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
    kwargs["force"] = True

    with pytest.raises(driver.CellStoreError, match="differs from the current result"):
        driver.run_project_methods(**kwargs)


def test_resume_identity_rejects_valid_changed_nl_npz_and_updated_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    receipt_path = _enable_nl_receipt(kwargs)
    _install_fakes(monkeypatch, kwargs)
    driver.run_project_methods(**kwargs)
    canonical_output = Path(kwargs["output_path"]).read_bytes()
    canonical_manifest = Path(kwargs["manifest_path"]).read_bytes()

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    concept_path = receipt_path.parent / receipt["concept_embeddings"]["path"]
    with np.load(concept_path, allow_pickle=False) as payload:
        texts = payload["texts"].copy()
        embeddings = payload["emb"].copy()
    embeddings[0] = np.asarray([0.0, 1.0], dtype=np.float32)
    np.savez(concept_path, texts=texts, emb=embeddings)
    receipt["concept_embeddings"]["sha256"] = hashlib.sha256(
        concept_path.read_bytes()
    ).hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    kwargs["force"] = True

    with pytest.raises(driver.CellStoreError, match="differs from the current result"):
        driver.run_project_methods(**kwargs)

    assert Path(kwargs["output_path"]).read_bytes() == canonical_output
    assert Path(kwargs["manifest_path"]).read_bytes() == canonical_manifest


@pytest.mark.parametrize(("first_nl", "second_nl"), [(False, True), (True, False)])
def test_resume_identity_rejects_enabling_or_disabling_nl_in_same_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    first_nl: bool,
    second_nl: bool,
) -> None:
    kwargs = _write_inputs(tmp_path)
    if first_nl:
        _enable_nl_receipt(kwargs)
    _install_fakes(monkeypatch, kwargs)
    driver.run_project_methods(**kwargs)

    if second_nl:
        _enable_nl_receipt(kwargs)
    else:
        kwargs["nl_target_source_manifest_path"] = None
    kwargs["force"] = True

    with pytest.raises(driver.CellStoreError, match="differs from the current result"):
        driver.run_project_methods(**kwargs)


def test_selection_seed_excludes_factor_and_token_seed_includes_it() -> None:
    selection = {driver.derive_selection_seed(42, 3, 7) for _factor in FULL_FACTORS}
    token = {
        driver.derive_generation_seed(42, "SelfConditioning", 3, 7, factor)
        for factor in FULL_FACTORS
    }
    assert len(selection) == 1
    assert len(token) == len(FULL_FACTORS)
    assert driver.derive_selection_seed(42, 3, 7) != driver.derive_selection_seed(42, 3, 8)
    assert driver.derive_selection_seed(42, 3, 7) != driver.derive_selection_seed(42, 4, 7)
    assert driver.derive_selection_seed(42, 3, 7) != driver.derive_selection_seed(43, 3, 7)


def test_existing_seed_domains_match_stability_goldens() -> None:
    assert (
        driver.derive_generation_seed(42, "SelfConditioning", 0, 0, 1.0) == _GENERATION_SEED_GOLDEN
    )
    assert driver.derive_selection_seed(42, 0, 0) == _SELECTION_SEED_GOLDEN


def test_encdec_generation_seed_methods_are_accepted_and_distinct() -> None:
    methods = (
        "SelfConditioning",
        "SelfConditioningMasked",
        "EncDecDiffMean",
        "EncDecLinearProbe",
        "EncDecPromptSteering",
        "EncDecLoRA",
    )

    seeds = {driver.derive_generation_seed(42, method, 0, 0, 1.0) for method in methods}

    # Every method (including both project methods) draws a distinct token-seed
    # stream at the same cell: the method name is a hash field, so a baseline
    # can never silently reuse SelfConditioning's stream.
    assert len(seeds) == len(methods)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (
            (42, "Wrong", 3, 7, 1.0),
            "method must be one of 'EncDecDiffMean', 'EncDecLinearProbe', "
            "'EncDecPromptSteering', 'EncDecLoRA', 'SelfConditioning', "
            "'SelfConditioningMasked', or 'NLSteering'",
        ),
        ((42, "SelfConditioning", 3, 7, True), "factor must be numeric"),
        ((42, "SelfConditioning", 3, 7, float("nan")), "finite and non-negative"),
        ((42, "SelfConditioning", 3, 7, -1.0), "finite and non-negative"),
        ((True, "SelfConditioning", 3, 7, 1.0), "base_seed must be an integer"),
        ((-1, "SelfConditioning", 3, 7, 1.0), "base_seed must be an integer"),
        (((1 << 63), "SelfConditioning", 3, 7, 1.0), "base_seed must be an integer"),
    ],
)
def test_generation_seed_rejects_malformed_identity(
    args: tuple[object, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        driver.derive_generation_seed(*cast(Any, args))


def test_policy_must_be_explicitly_closed(tmp_path: Path) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["self_conditioning_policy"] = "legacy-policy"
    with pytest.raises(ValueError, match=SELF_CONDITIONING_POLICY):
        driver.run_project_methods(**kwargs)


def test_default_hydra_config_uses_rank_policy_without_threshold_scale() -> None:
    config_dir = _REPO_ROOT / "configs" / "eval" / "axbench_upstream"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="methods")
    assert cfg.self_conditioning_policy == SELF_CONDITIONING_POLICY
    assert cfg.self_conditioning_masked_policy == SELF_CONDITIONING_MASKED_POLICY
    assert "masked_threshold_scale" not in cfg
    assert cfg.nl_target_source_manifest_path is None
    assert "project_v4" in cfg.output_path
    assert "project_v4" in cfg.manifest_path


def test_nl_hydra_preset_inherits_defaults_and_selects_explicit_receipt() -> None:
    config_dir = _REPO_ROOT / "configs" / "eval" / "axbench_upstream"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(config_name="methods_nl")

    assert cfg.self_conditioning_policy == SELF_CONDITIONING_POLICY
    assert cfg.self_conditioning_masked_policy == SELF_CONDITIONING_MASKED_POLICY
    assert cfg.nl_target_source_manifest_path == (
        "out/axbench/nl_target_source_manifest.json"
    )
    assert "project_nl_v1" in cfg.output_path
    assert "project_nl_v1" in cfg.manifest_path


def test_existing_output_fails_without_force_and_preserves_bytes(tmp_path: Path) -> None:
    kwargs = _write_inputs(tmp_path)
    output_path = Path(kwargs["output_path"])
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"keep")

    with pytest.raises(FileExistsError, match="pass force=true"):
        driver.run_project_methods(**kwargs)
    assert output_path.read_bytes() == b"keep"


def test_force_rejects_threshold_era_output_without_deleting_it(tmp_path: Path) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["force"] = True
    output_path = Path(kwargs["output_path"])
    manifest_path = Path(kwargs["manifest_path"])
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"threshold-era-parquet")
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "methods": {"SelfConditioningMasked": {"masked_threshold_scale": 0.1}},
            }
        ),
        encoding="utf-8",
    )
    original_manifest = manifest_path.read_bytes()

    with pytest.raises(ValueError, match="manually clear.*rank-era"):
        driver.run_project_methods(**kwargs)

    assert output_path.read_bytes() == b"threshold-era-parquet"
    assert manifest_path.read_bytes() == original_manifest


def test_force_rejects_stale_threshold_resume_identity_without_deleting_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    driver.run_project_methods(**kwargs)
    output_path = Path(kwargs["output_path"])
    manifest_path = Path(kwargs["manifest_path"])
    identity_path = manifest_path.parent / "_project_cell_resume_v1" / "identity.json"
    identity = json.loads(identity_path.read_text(encoding="utf-8"))
    identity["semantic_identity"]["configuration"]["self_conditioning_masked_policy"] = (
        "prompt_last_segment_max_pool_logit_rank_diffmean_sign_mask_v2"
    )
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    stale_identity = identity_path.read_bytes()
    canonical_output = output_path.read_bytes()
    canonical_manifest = manifest_path.read_bytes()
    kwargs["force"] = True

    with pytest.raises(RuntimeError, match="resume identity.*differs.*refusing to overwrite"):
        driver.run_project_methods(**kwargs)

    assert identity_path.read_bytes() == stale_identity
    assert output_path.read_bytes() == canonical_output
    assert manifest_path.read_bytes() == canonical_manifest


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("max_concepts", True, "max_concepts must be a positive integer"),
        ("device", "", "device must be a nonempty string"),
        ("seed", -1, "seed must be a nonnegative integer"),
        ("seed", 2**63, "seed must be a nonnegative integer in \\[0, 2\\^63 - 1\\]"),
        ("max_new_tokens", False, "max_new_tokens must be a positive integer"),
        ("extraction_batch_size", 0, "extraction_batch_size must be a positive integer"),
        ("force", 1, "force must be a boolean"),
    ],
)
def test_scalar_config_domains_fail_before_artifact_io(
    tmp_path: Path,
    key: str,
    value: object,
    error: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs[key] = value
    with pytest.raises(ValueError, match=error):
        driver.run_project_methods(**kwargs)


def test_multi_defect_config_reports_driver_specific_errors_before_force(
    tmp_path: Path,
) -> None:
    """Pins the ratified validation precedence: ``force`` is checked LAST.

    A config with both a bad driver-specific scalar and a non-bool ``force``
    must report the driver-specific defect first (the pre-refactor order),
    guarding the shared-helper extraction against precedence drift (final
    merge-gate R1 finding).
    """

    kwargs = _write_inputs(tmp_path)
    kwargs["extraction_batch_size"] = 0
    kwargs["force"] = 1
    with pytest.raises(ValueError, match="extraction_batch_size must be a positive integer"):
        driver.run_project_methods(**kwargs)


@pytest.mark.parametrize(
    ("manifest", "error"),
    [
        (
            {
                "schema_version": True,
                "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
                "benchmark": {"revision": "dataset-revision"},
            },
            "schema_version must equal 1",
        ),
        (
            {
                "schema_version": 1,
                "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
                "benchmark": {"revision": "   "},
            },
            "benchmark revision",
        ),
        (
            {
                "schema_version": 1,
                "axbench_upstream_commit": "wrong-commit",
                "benchmark": {"revision": "dataset-revision"},
            },
            "vendored snapshot",
        ),
        (
            {
                "schema_version": 1,
                "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
                "benchmark": {"revision": "dataset-revision"},
                "nonfinite": float("nan"),
            },
            "strict JSON",
        ),
    ],
)
def test_prepared_manifest_identity_fails_before_model_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manifest: dict[str, Any],
    error: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    prepared_manifest = Path(kwargs["prepared_dir"]) / "prepare_manifest.json"
    prepared_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    setup_calls = 0

    def forbidden_setup(*_args: object) -> object:
        nonlocal setup_calls
        setup_calls += 1
        raise AssertionError("invalid provenance must fail before setup")

    monkeypatch.setattr(driver, "load_method_setup", forbidden_setup)
    with pytest.raises(ValueError, match=error):
        driver.run_project_methods(**kwargs)
    assert setup_calls == 0
    assert not Path(kwargs["output_path"]).exists()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda paths: paths.update(output_path=paths["target_meta_path"]), "output_path"),
        (
            lambda paths: paths.update(manifest_path=paths["feature_statistics_path"]),
            "manifest_path",
        ),
        (
            lambda paths: paths.update(
                main_path=Path(paths["output_path"]).with_suffix(".rejected.parquet")
            ),
            "rejected",
        ),
        (
            lambda paths: paths.update(
                output_path=Path(paths["prepared_dir"]) / "generate" / "train_data.parquet"
            ),
            "output_path",
        ),
        (lambda paths: paths.update(manifest_path=paths["output_path"]), "manifest_path"),
    ],
)
def test_every_read_and_write_path_is_alias_safe_before_io(
    tmp_path: Path,
    mutate: Any,
    error: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    mutate(kwargs)
    with pytest.raises(ValueError, match=error):
        driver.run_project_methods(**kwargs)


def test_failed_final_validation_writes_only_complete_rejected_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs, generation=" ")
    monkeypatch.setattr(driver, "_assert_first_concept_not_collapsed", lambda *_a, **_k: None)
    output_path = Path(kwargs["output_path"])
    rejected_path = output_path.with_suffix(".rejected.parquet")

    with pytest.raises(ValueError, match="blank generations") as caught:
        driver.run_project_methods(**kwargs)

    assert not output_path.exists()
    assert not Path(kwargs["manifest_path"]).exists()
    rejected = pd.read_parquet(rejected_path)
    assert rejected.shape == (240, 18)
    assert rejected["SelfConditioning_steered_generation"].eq(" ").all()
    assert any(str(rejected_path) in note for note in caught.value.__notes__)


def test_force_preserves_valid_outputs_when_replacement_fails_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["force"] = True
    output_path = Path(kwargs["output_path"])
    manifest_path = Path(kwargs["manifest_path"])
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"old-canonical-parquet")
    manifest_path.write_bytes(b"old-canonical-manifest")
    _install_fakes(monkeypatch, kwargs, generation=" ")
    monkeypatch.setattr(driver, "_assert_first_concept_not_collapsed", lambda *_a, **_k: None)

    with pytest.raises(ValueError, match="blank generations"):
        driver.run_project_methods(**kwargs)

    assert output_path.read_bytes() == b"old-canonical-parquet"
    assert manifest_path.read_bytes() == b"old-canonical-manifest"
    assert output_path.with_suffix(".rejected.parquet").is_file()


def test_rejected_sidecar_failure_preserves_original_validation_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs, generation=" ")
    monkeypatch.setattr(driver, "_assert_first_concept_not_collapsed", lambda *_a, **_k: None)

    def failed_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated rejected-sidecar failure")

    monkeypatch.setattr(driver, "atomic_write_parquet", failed_write)
    with pytest.raises(ValueError, match="blank generations") as caught:
        driver.run_project_methods(**kwargs)
    assert isinstance(caught.value.__cause__, OSError)
    assert any("failed to write rejected project frame" in note for note in caught.value.__notes__)


def test_manifest_serialization_precedes_canonical_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    monkeypatch.setattr(driver, "_manifest", lambda **_kwargs: {"bad": float("nan")})

    with pytest.raises(ValueError, match="Out of range float values"):
        driver.run_project_methods(**kwargs)
    assert not Path(kwargs["output_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()


def test_forced_publication_removes_stale_manifest_before_parquet_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    driver.run_project_methods(**kwargs)
    kwargs["force"] = True

    def fail_publication(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("simulated forced project publication failure")

    monkeypatch.setattr(driver, "atomic_write_parquet", fail_publication)
    with pytest.raises(OSError, match="forced project publication failure"):
        driver.run_project_methods(**kwargs)

    assert not Path(kwargs["manifest_path"]).exists()


def test_force_replaces_both_outputs_after_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["force"] = True
    output_path = Path(kwargs["output_path"])
    manifest_path = Path(kwargs["manifest_path"])
    output_path.parent.mkdir(parents=True)
    output_path.write_bytes(b"old-parquet")
    manifest_path.write_bytes(b"old-manifest")
    _install_fakes(monkeypatch, kwargs)

    result = driver.run_project_methods(**kwargs)

    pd.testing.assert_frame_equal(pd.read_parquet(output_path), result, check_dtype=False)
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["schema_version"] == 3


def test_cuda_cleanup_branch_runs_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["device"] = "cuda:0"
    _install_fakes(monkeypatch, kwargs)
    cleanup_calls = 0
    monkeypatch.setattr(driver.t.cuda, "is_available", lambda: True)

    def fake_empty_cache() -> None:
        nonlocal cleanup_calls
        cleanup_calls += 1

    monkeypatch.setattr(driver.t.cuda, "empty_cache", fake_empty_cache)
    driver.run_project_methods(**kwargs)
    assert cleanup_calls == 1


def test_hydra_ingress_forwards_every_key_with_exact_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    raw = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    captured: list[dict[str, Any]] = []
    monkeypatch.setattr(
        driver,
        "run_project_methods",
        lambda **received: captured.append(received) or pd.DataFrame(),
    )

    driver.run_project_methods_from_dictconfig(OmegaConf.create(raw))
    raw["nl_target_source_manifest_path"] = None
    driver.run_project_methods_from_dictconfig(OmegaConf.create(raw))

    expected = {
        **kwargs,
        "project_input_path": None,
        "project_input_manifest_path": None,
        "nl_target_source_manifest_path": None,
        "concept_ids": None,
    }
    assert captured == [expected, expected]
    assert type(captured[0]["max_concepts"]) is int
    assert type(captured[0]["force"]) is bool
    for key in (
        "main_path",
        "prepared_dir",
        "output_path",
        "manifest_path",
        "target_meta_path",
        "feature_statistics_path",
    ):
        assert isinstance(captured[0][key], Path)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (lambda config: config.pop("force"), "missing=\\['force'\\]"),
        (lambda config: config.update(temperature=1.0), "unknown=\\['temperature'\\]"),
    ],
)
def test_hydra_ingress_rejects_missing_and_unknown_keys(
    tmp_path: Path,
    mutation: Any,
    expected: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    raw = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    mutation(raw)
    with pytest.raises(ValueError, match=expected):
        driver.run_project_methods_from_dictconfig(OmegaConf.create(raw))


def test_hydra_ingress_requires_mapping_with_string_keys() -> None:
    with pytest.raises(ValueError, match="config must be a mapping"):
        driver.run_project_methods_from_dictconfig(cast(Any, OmegaConf.create(["not", "mapping"])))
    with pytest.raises(ValueError, match="config keys must be strings"):
        driver.run_project_methods_from_dictconfig(cast(Any, OmegaConf.create({1: "value"})))


def _generations(count: int, blanks: int, *, healthy_length: int = 40) -> list[str]:
    return ([""] * blanks) + (["x" * healthy_length] * (count - blanks))


def test_early_collapse_threshold_boundaries_and_final_no_floor() -> None:
    driver._assert_first_concept_not_collapsed(
        {"SelfConditioning": _generations(120, 4)},
        concept_id=0,
    )
    with pytest.raises(ValueError, match=r"5 of 120 blank generations"):
        driver._assert_first_concept_not_collapsed(
            {"SelfConditioning": _generations(120, 5)},
            concept_id=3,
        )
    with pytest.raises(ValueError, match=r"median stripped length is 1"):
        driver._assert_first_concept_not_collapsed(
            {"SelfConditioning": [","] * 120},
            concept_id=4,
        )

    project = _main_frame(max_concepts=1).loc[:, list(INPUT_COLUMNS)].copy()
    project["SelfConditioning_steered_generation"] = "a healthy generation"
    project["SelfConditioning_strength"] = project["factor"].astype("float64")
    project["SelfConditioningMasked_steered_generation"] = "a healthy masked generation"
    project["SelfConditioningMasked_strength"] = project["factor"].astype("float64")
    project.loc[project.index[:2], "SelfConditioning_steered_generation"] = " "
    with pytest.raises(ValueError, match="blank generations"):
        validate_steering_frame(project, source="project", max_concepts=1)


def test_project_frame_carries_both_method_columns_in_contract_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)

    result = driver.run_project_methods(**kwargs)

    assert list(result.columns) == [
        *INPUT_COLUMNS,
        "SelfConditioning_steered_generation",
        "SelfConditioning_strength",
        "SelfConditioningMasked_steered_generation",
        "SelfConditioningMasked_strength",
    ]
    assert result["SelfConditioningMasked_steered_generation"].str.startswith("masked ").all()
    assert not result["SelfConditioning_steered_generation"].str.startswith("masked ").any()
    assert result["SelfConditioningMasked_strength"].equals(result["factor"].astype("float64"))
    assert result["SelfConditioning_strength"].equals(result["factor"].astype("float64"))


def test_masked_shares_selection_seeds_but_uses_distinct_token_seeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)

    driver.run_project_methods(**kwargs)

    assert observed["masked_selection_seeds"] == observed["selection_seeds"]
    for pair, masked_seeds in observed["masked_token_seeds"].items():
        assert len(masked_seeds) == len(FULL_FACTORS)
        assert masked_seeds.isdisjoint(observed["token_seeds"][pair])


def test_masked_policy_must_be_explicitly_closed(tmp_path: Path) -> None:
    kwargs = _write_inputs(tmp_path)
    kwargs["self_conditioning_masked_policy"] = "legacy-masked-policy"
    with pytest.raises(ValueError, match=SELF_CONDITIONING_MASKED_POLICY):
        driver.run_project_methods(**kwargs)


@pytest.mark.parametrize(
    "obsolete_key", ["masked_threshold_scale", "masked_threshold_scale_overrides"]
)
def test_obsolete_masked_config_keys_fail_before_artifact_io(
    tmp_path: Path,
    obsolete_key: str,
) -> None:
    kwargs = _write_inputs(tmp_path)
    raw = {key: str(value) if isinstance(value, Path) else value for key, value in kwargs.items()}
    raw[obsolete_key] = 0.1
    with pytest.raises(ValueError, match=rf"unknown=\['{obsolete_key}'\]"):
        driver.run_project_methods_from_dictconfig(OmegaConf.create(raw))
    assert not Path(kwargs["output_path"]).exists()


def test_first_concept_collapse_names_the_masked_method(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _install_fakes(monkeypatch, kwargs)
    healthy_batch = driver.generate_self_conditioning_batch

    def collapse_only_the_masked_policy(*args: Any, **generate_kwargs: Any) -> tuple[str, ...]:
        healthy = healthy_batch(*args, **generate_kwargs)
        requests = tuple(args[4])
        return tuple(
            "ab" if isinstance(request.policy, SelfCondScoreMasking) else generation
            for request, generation in zip(requests, healthy, strict=True)
        )

    monkeypatch.setattr(
        driver,
        "generate_self_conditioning_batch",
        collapse_only_the_masked_policy,
    )

    with pytest.raises(
        ValueError,
        match="'SelfConditioningMasked' concept_id=0 has collapsed output",
    ):
        driver.run_project_methods(**kwargs)
    assert not Path(kwargs["output_path"]).exists()


def test_masked_token_seed_stream_is_method_scoped() -> None:
    lerp = driver.derive_generation_seed(42, "SelfConditioning", 3, 7, 1.0)
    masked = driver.derive_generation_seed(42, "SelfConditioningMasked", 3, 7, 1.0)
    assert lerp != masked


def test_shuffled_source_keeps_both_methods_row_aligned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every generation cell must land on its own (concept, input, factor) row
    even when the source frame arrives concept-interleaved; a transposition in
    either method's by-position restore would swap cell contents silently."""
    kwargs = _write_inputs(tmp_path)
    frame = pd.read_parquet(kwargs["main_path"])
    frame.sample(frac=1.0, random_state=7).reset_index(drop=True).to_parquet(
        kwargs["main_path"],
        index=False,
    )
    _install_fakes(monkeypatch, kwargs)

    result = driver.run_project_methods(**kwargs)

    for raw_row in result.itertuples(index=False):
        row: Any = raw_row
        pair = (int(row.concept_id), int(row.input_id))
        cell = f"{pair} factor={float(row.factor)}"
        lerp_value = str(row.SelfConditioning_steered_generation)
        masked_value = str(row.SelfConditioningMasked_steered_generation)
        assert lerp_value.endswith(cell), (pair, row.factor, lerp_value)
        assert masked_value.endswith(cell), (pair, row.factor, masked_value)
        assert masked_value.startswith("masked ")
        assert not lerp_value.startswith("masked ")


def test_masked_pool_preflight_aborts_before_target_model_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An infeasible masked pool in ANY concept must abort after the dense
    fits but before the target checkpoint is loaded or any row generated."""
    kwargs = _write_inputs(tmp_path)
    observed = _install_fakes(monkeypatch, kwargs)

    def fit_with_infeasible_second_concept(*_args: Any, **_kwargs: Any) -> dict[int, t.Tensor]:
        observed["events"].append("fit")
        return {
            0: t.linspace(0.8, -0.8, 8),
            1: t.linspace(-0.1, -0.8, 8),
        }

    monkeypatch.setattr(driver, "fit_concept_dense_diffmeans", fit_with_infeasible_second_concept)

    with pytest.raises(
        ValueError,
        match=r"concept_id=1.*empty positive candidate pool",
    ):
        driver.run_project_methods(**kwargs)

    assert "target" not in observed["events"]
    assert not Path(kwargs["output_path"]).exists()


def test_turn_manifest_geometry_requires_raw_statistics_identity() -> None:
    setup: Any = SimpleNamespace(
        segment_mode=driver.SegmentMode.turn_delimited_fixed_tokens,
        prompt_selection_context=turn_prompt_selection_context(),
        prompt_statistics_raw_sha256=None,
    )
    with pytest.raises(ValueError, match="no raw statistics identity"):
        driver._turn_self_conditioning_segment_geometry(setup)


def test_global_subset_matches_full_cells_and_preparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    kwargs = _write_inputs(tmp_path)
    full_state = _install_fakes(monkeypatch, kwargs)
    full = driver.run_project_methods(**kwargs)
    subset_kwargs: dict[str, Any] = {
        **kwargs,
        "concept_ids": [1],
        "output_path": tmp_path / "subset/inference/steering_data.parquet",
        "manifest_path": tmp_path / "subset/run_manifest.json",
    }
    subset_state = _install_fakes(monkeypatch, subset_kwargs)
    subset = driver.run_project_methods(**subset_kwargs)
    pd.testing.assert_frame_equal(subset, full.loc[full.concept_id.eq(1)])
    assert full_state["fitted_concepts"] == [(0, 1)]
    assert subset_state["fitted_concepts"] == [(1,)]
    assert subset_state['prepared_tokens'] == full_state['prepared_tokens']
    assert set(subset_state['selection_seeds']) == {(1, i) for i in range(10)}
    for key in ('selection_seeds', 'token_seeds', 'masked_selection_seeds', 'masked_token_seeds'):
        assert subset_state[key] == {pair: seeds for pair, seeds in full_state[key].items() if pair[0] == 1}
    manifest = json.loads(subset_kwargs['manifest_path'].read_text())
    assert manifest['configuration']['concept_ids'] == [1]
    fit_identity = json.loads((tmp_path / 'subset/_project_fit_resume_v1/identity.json').read_text())
    assert fit_identity['concept_ids'] == [1]
    receipt = json.loads((tmp_path / 'subset/_project_cell_resume_v1/fit_artifacts.json').read_text())
    assert [item['concept_id'] for item in receipt['files']] == [1]
    with pytest.raises(FileExistsError):
        driver.run_project_methods(**subset_kwargs)
    with pytest.raises(ValueError, match='exact key product'):
        validate_steering_frame(subset.iloc[:-1], source='project', max_concepts=2, concept_ids=(1,))


def test_sparse_global_50_99_grid_keeps_native_cell_seeds() -> None:
    frame = _main_frame(100)
    full = driver.build_project_cell_grid(frame, seed=42)
    ids = (50, 61, 99)
    subset = driver.build_project_cell_grid(frame.loc[frame.concept_id.isin(ids)], seed=42)
    def semantic(cell):
        return (cell.method, cell.concept_id, cell.input_id, cell.factor_hex, cell.selection_seed, cell.token_seed)
    assert [semantic(cell) for cell in subset] == [semantic(cell) for cell in full if cell.concept_id in ids]
    assert len(subset) == 3 * 10 * 12 * 2


@pytest.mark.parametrize(
    ("concept_ids", "cell_count", "trace_concept"),
    [(None, 480, 0), ([1], 240, 1), ([1, 0], 480, 0)],
)
def test_global_subset_profile_matches_unprofiled_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    concept_ids: list[int] | None, cell_count: int, trace_concept: int,
) -> None:
    kwargs: dict[str, Any] = {**_write_inputs(tmp_path), "concept_ids": concept_ids}
    _install_fakes(monkeypatch, kwargs)
    driver.run_project_methods(**kwargs)
    reference_manifest = kwargs["manifest_path"]
    kwargs.update(
        output_path=tmp_path / "candidate/inference/steering_data.parquet",
        manifest_path=tmp_path / "candidate/run_manifest.json",
    )
    overrides = [
        f"{key}={value if isinstance(value, Path) else json.dumps(value)}"
        for key, value in kwargs.items()
    ]

    benchmark.run_profile(
        kind="project",
        output_dir=tmp_path / "profile",
        config_name="methods",
        overrides=overrides,
        torch_traces=True,
        synchronize_cuda=False,
    )

    comparison = compare_project_bundle(reference_manifest, kwargs["manifest_path"])
    assert comparison.cell_count == cell_count
    rank_dir = tmp_path / "profile/rank_00000"
    traces = list((rank_dir / "torch").glob("*/key_averages.json"))
    assert len(traces) == 1
    trace = json.loads(traces[0].read_text())
    assert trace["selector"]["concept_id"] == trace_concept
    assert trace["cell"]["concept_id"] == trace_concept
    summary = yaml.safe_load((rank_dir / "regions/summary.yaml").read_text())
    assert summary["metadata"]["representative_selectors"] == [trace["selector"]]
