"""Authenticate completed encoder-decoder baseline positives for latent Adv."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch as t

from scripts.eval.axbench_upstream.merge_methods import (
    INPUT_COLUMNS,
    METHOD_FACTORS,
    validate_steering_frame,
)
from scripts.eval.axbench_upstream.run_evaluation import validate_evaluation_outputs
from steering.eval.axbench_adv.best_point import (
    load_steering_entries,
    select_best_points,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import validate_rating_completion
from steering.eval.axbench_adv.native_mailbox import read_json
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from steering.eval.axbench_upstream.cell_store import CellResultStore
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    load_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.layer_resolution import (
    VENDORED_LORA_LAYERS,
    VENDORED_STEERING_LAYER,
    resolve_vendored_layer,
    resolve_vendored_layers,
)
from steering.eval.axbench_upstream.encdec_baselines.lora import load_lora
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    prompt_wrapper_for_frame,
    recover_concept_steering_prompts,
    source_manifest_path,
)
from steering.eval.axbench_upstream.encdec_baselines.run import (
    METHOD_ORDER,
    _artifact_inventory,
    _expected_artifacts,
    _steered_prompt_frame,
)
from steering.eval.axbench_upstream.methods import MethodSetup, load_method_setup
from steering.eval.axbench_upstream.run_methods import (
    _file_identity,
    _target_checkpoint_identity,
    _tokenizer_artifact_identity,
)
from steering.eval.axbench_upstream.work_grid import build_encdec_cell_grid

METHOD_FACTORS_ENCDEC = {method: METHOD_FACTORS[method] for method in ENCDEC_BASELINE_METHODS}
TARGET_MODEL_SHA256 = "9e7fb23e4f39b2094c175fb7e80b91e6df7a30be6e8f3ecfe4d90b6eaa8f1842"
TARGET_META_SHA256 = "81e0af5b64b9dfcf8d6aa4ec80aacbf086a8885e86b6d8a247e40789119eda31"
TARGET_MODEL_TAG = "encdec400m-l22-s32-mlp1-full-sft-20260921"


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise AxBenchAdvError(f"encoder-decoder source {label} differs")


@dataclass(frozen=True)
class EncDecSources:
    setup: MethodSetup
    source_main: pd.DataFrame
    best_factors: Mapping[str, float]
    steering_prompts: Mapping[int, str]
    identity: Mapping[str, Any]
    manifest: Mapping[str, Any]
    fitted: Mapping[int, Mapping[str, FittedDirection]]
    lora_states: Mapping[int, dict[str, t.Tensor]]
    source_digest: Mapping[str, str]


def validate_encdec_sources(
    *,
    positive_root: Path,
    source_main_path: Path,
    source_aux_path: Path,
    prepared_dir: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    judged_input_path: Path,
    judged_dir: Path,
    pretrain_meta_path: Path | None = None,
) -> EncDecSources:
    """Bind producer rows, fits, checkpoint, conditioning, and judged factors."""

    positive_root = positive_root.resolve()
    source_main_path = source_main_path.resolve()
    source_aux_path = source_aux_path.resolve()
    prepared_dir = prepared_dir.resolve()
    target_meta_path = target_meta_path.resolve()
    feature_statistics_path = feature_statistics_path.resolve()
    judged_input_path = judged_input_path.resolve()
    judged_dir = judged_dir.resolve()
    manifest_path = positive_root / "run_manifest.json"
    resume_root = positive_root / "_encdec_cell_resume_v1"
    train_dir = positive_root / "train"
    manifest = read_json(manifest_path)
    bound = read_json(resume_root / "identity.json")
    receipt = read_json(resume_root / "fit_artifacts.json")
    if manifest.get("schema_version") != 1 or bound.get("schema_version") != 1:
        raise AxBenchAdvError("encoder-decoder producer schema differs")
    semantic = bound.get("semantic_identity")
    if not isinstance(semantic, dict) or semantic.get("driver") != "encdec":
        raise AxBenchAdvError("encoder-decoder producer resume identity is invalid")
    # CellResultStore below authenticates the full identity, grid and fingerprint.
    for key in ("repository_git_commit", "repository_git_worktree", "package_lock_sha256"):
        _equal(manifest.get(key), semantic.get(key), key)
    config = manifest.get("configuration")
    identity_inputs = semantic.get("inputs")
    if not isinstance(config, dict) or not isinstance(identity_inputs, dict):
        raise AxBenchAdvError("encoder-decoder producer configuration or inputs are missing")
    max_concepts = config.get("max_concepts")
    if type(max_concepts) is not int or max_concepts != 50:
        raise AxBenchAdvError("encoder-decoder Adv requires the completed c50 producer")
    if config.get("seed") != 42 or config.get("max_new_tokens") != 128:
        raise AxBenchAdvError("encoder-decoder producer generation contract differs")
    if type(config.get("require_exact_statistics_identity")) is not bool:
        raise AxBenchAdvError("encoder-decoder statistics identity policy must be Boolean")
    for key in ("max_concepts", "seed", "max_new_tokens", "require_exact_statistics_identity"):
        _equal(semantic.get("configuration", {}).get(key), config[key], key)
    paths = {
        "main_sha256": source_main_path,
        "aux_sha256": source_aux_path,
        "main_source_manifest_sha256": source_manifest_path(source_main_path),
        "aux_source_manifest_sha256": source_manifest_path(source_aux_path),
        "prepared_manifest_sha256": prepared_dir / "prepare_manifest.json",
        "target_meta_sha256": target_meta_path,
        "feature_statistics_sha256": feature_statistics_path,
    }
    for key, path in paths.items():
        _equal(identity_inputs.get(key), _sha(path), key)
    for key, name in (
        ("prepared_metadata", "metadata.jsonl"),
        ("prepared_training_data", "train_data.parquet"),
    ):
        _equal(identity_inputs.get(key), _file_identity(prepared_dir / "generate" / name), key)
    _equal(
        identity_inputs.get("target_model"),
        _target_checkpoint_identity(target_meta_path),
        "target weights",
    )
    _equal(identity_inputs["target_model"]["sha256"], TARGET_MODEL_SHA256, "campaign target model")
    _equal(identity_inputs["target_meta_sha256"], TARGET_META_SHA256, "campaign target metadata")
    _equal(identity_inputs.get("tokenizer"), _tokenizer_artifact_identity(), "RustBPE tokenizer")
    _equal(
        identity_inputs.get("pretrain_meta_sha256"),
        None if pretrain_meta_path is None else _sha(pretrain_meta_path),
        "pretrain metadata",
    )
    for key, path in (
        ("main_steering_parquet_sha256", source_main_path),
        ("aux_steering_parquet_sha256", source_aux_path),
        ("feature_statistics_sha256", feature_statistics_path),
    ):
        _equal(manifest.get("inputs", {}).get(key), _sha(path), key)
    _equal(
        manifest.get("prepared_manifest"),
        read_json(prepared_dir / "prepare_manifest.json"),
        "prepared manifest",
    )
    setup = load_method_setup(
        target_meta_path,
        feature_statistics_path,
        pretrain_meta_path=pretrain_meta_path,
        require_exact_statistics_identity=config["require_exact_statistics_identity"],
    )
    _equal(manifest.get("target", {}).get("step"), setup.meta.get("step"), "target step")
    _equal(setup.meta.get("step"), 1013, "campaign target step")
    _equal(manifest.get("target", {}).get("model_tag"), TARGET_MODEL_TAG, "campaign model tag")
    _equal(manifest.get("generation"), {"temperature": 1.0, "top_k": 50}, "generation sampling")
    model_config = setup.meta.get("model_config")
    if not isinstance(model_config, dict):
        raise AxBenchAdvError("target model geometry is missing")
    n_layer = int(model_config["n_layer"])
    width = int(model_config["n_embd"])
    layer_resolution = manifest.get("layer_resolution")
    if not isinstance(layer_resolution, dict):
        raise AxBenchAdvError("positive layer resolution is missing")
    positive_layer = resolve_vendored_layer(VENDORED_STEERING_LAYER, target_n_layer=n_layer)
    lora_layers = resolve_vendored_layers(VENDORED_LORA_LAYERS, target_n_layer=n_layer)
    _equal(layer_resolution.get("target_n_layer"), n_layer, "target depth")
    _equal(layer_resolution.get("resolved_intervention_layer"), positive_layer, "positive layer")
    _equal(layer_resolution.get("resolved_lora_layers"), list(lora_layers), "positive LoRA layers")
    main_source = pd.read_parquet(source_main_path)
    aux_source = pd.read_parquet(source_aux_path)
    positive_main = pd.read_parquet(positive_root / "main/inference/steering_data.parquet")
    positive_aux = pd.read_parquet(positive_root / "aux/inference/steering_data.parquet")
    for frame, name in (
        (main_source, "main"),
        (aux_source, "aux"),
        (positive_main, "encdec_main"),
        (positive_aux, "encdec_aux"),
    ):
        validate_steering_frame(
            frame,
            source=name,
            max_concepts=50,
            check_method_values=name in {"encdec_main", "encdec_aux"},
        )
    if not positive_main.loc[:, list(INPUT_COLUMNS)].equals(
        main_source.loc[:, list(INPUT_COLUMNS)]
    ):
        raise AxBenchAdvError("positive main source rows differ")
    # The producer replaces steered_input in the auxiliary frame. Authenticate
    # that reconstructed prompt below; all other source columns must match.
    shared_columns = [column for column in INPUT_COLUMNS if column != "steered_input"]
    one = positive_main.loc[positive_main["factor"].eq(1.0), shared_columns].reset_index(
        drop=True
    )
    if not positive_aux.loc[:, shared_columns].reset_index(drop=True).equals(one):
        raise AxBenchAdvError("positive auxiliary source rows differ")
    wrapper = prompt_wrapper_for_frame(source_aux_path)
    _equal(wrapper, prompt_wrapper_for_frame(source_main_path), "source prompt wrapper")
    prompts = recover_concept_steering_prompts(aux_source, max_concepts=50, wrapper=wrapper)
    steered, _ = _steered_prompt_frame(positive_main, prompts)
    if positive_aux["steered_input"].tolist() != steered["original_prompt"].tolist():
        raise AxBenchAdvError("positive PromptSteering text differs from authenticated source")
    for method in ("EncDecPromptSteering", "EncDecLoRA"):
        if positive_aux[f"{method}_strength"].tolist() != [1.0] * len(positive_aux):
            raise AxBenchAdvError(f"positive {method} strengths differ")
    train_inventory = manifest.get("train_artifacts", {}).get("files")
    expected = _expected_artifacts(train_dir, max_concepts=50)
    _equal(
        train_inventory,
        _artifact_inventory(expected, train_dir=train_dir),
        "positive fit inventory",
    )
    if {path.name for path in train_dir.iterdir()} != {path.name for path, _, _ in expected}:
        raise AxBenchAdvError("positive fit directory contains unexpected files")
    method_manifest = manifest.get("methods")
    if not isinstance(method_manifest, dict) or set(method_manifest) != set(
        ENCDEC_BASELINE_METHODS
    ):
        raise AxBenchAdvError("positive method manifest coverage differs")
    _equal(manifest.get("method_order"), list(METHOD_ORDER), "positive method order")
    for method in ENCDEC_BASELINE_METHODS:
        _equal(
            method_manifest[method].get("factors"),
            list(METHOD_FACTORS_ENCDEC[method]),
            f"{method} factor grid",
        )
    for concept_id, prompt in prompts.items():
        _equal(
            method_manifest["EncDecPromptSteering"]["steering_prompt_sha256_by_concept"].get(
                str(concept_id)
            ),
            hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            f"concept {concept_id} steering prompt",
        )
    fitted: dict[int, dict[str, FittedDirection]] = {}
    lora_states: dict[int, dict[str, t.Tensor]] = {}
    for concept_id in range(50):
        fitted[concept_id] = {}
        for method, kind, fit_method in (
            ("EncDecDiffMean", "diffmean", "diffmean"),
            ("EncDecLinearProbe", "linear_probe", "probe"),
        ):
            fit = load_direction(train_dir / f"concept_{concept_id:05d}_{kind}.safetensors")
            fitted[concept_id][fit_method] = fit
            if (
                fit.metadata.get("method") != fit_method
                or fit.metadata.get("intervention_layer") != positive_layer
                or fit.direction.numel() != width
                or (fit.max_activation == 0 and bool(fit.direction.ne(0).any()))
            ):
                raise AxBenchAdvError(
                    f"positive {method} fit metadata differs for concept {concept_id}"
                )
            _equal(
                method_manifest[method]["max_activation_by_concept"].get(str(concept_id)),
                repr(float(fit.max_activation)),
                f"{method} calibration",
            )
            concept_rows = positive_main.loc[positive_main["concept_id"].eq(concept_id)]
            if not concept_rows[f"{method}_strength"].eq(
                concept_rows["factor"] * fit.max_activation
            ).all():
                raise AxBenchAdvError(
                    f"positive {method} strengths differ for concept {concept_id}"
                )
        state, metadata = load_lora(train_dir / f"concept_{concept_id:05d}_lora.safetensors")
        lora_states[concept_id] = state
        expected_metadata = method_manifest["EncDecLoRA"]["training_by_concept"].get(
            str(concept_id)
        )
        if not isinstance(expected_metadata, dict) or metadata != {
            key: str(value) for key, value in expected_metadata.items()
        }:
            raise AxBenchAdvError(f"positive LoRA metadata differs for concept {concept_id}")
        keys = {
            f"layer_{layer}_attn_c_proj_lora_{side}" for layer in lora_layers for side in ("A", "B")
        }
        if set(state) != keys:
            raise AxBenchAdvError(f"positive LoRA layer coverage differs for concept {concept_id}")
    cells = build_encdec_cell_grid(main_source, seed=42)
    first = tuple(cell for cell in cells if cell.concept_id == 0)
    with CellResultStore(
        root=resume_root,
        cells=cells,
        tripwire_cells=first,
        semantic_identity=semantic,
        rank=0,
        world_size=1,
    ) as store:
        _equal(
            receipt,
            {
                "schema_version": 1,
                "semantic_fingerprint": store.semantic_fingerprint,
                "files": train_inventory,
            },
            "fit receipt",
        )
        stored = store.complete_results()
    merged = pd.read_parquet(judged_input_path)
    if judged_input_path.name != "steering_data.parquet" or not merged.loc[
        :, list(INPUT_COLUMNS)
    ].equals(positive_main.loc[:, list(INPUT_COLUMNS)]):
        raise AxBenchAdvError("judged input base rows differ from the positive producer")
    validate_evaluation_outputs(merged, judged_dir)
    for positive in (positive_main, positive_aux):
        keys = ["concept_id", "input_id", "factor"]
        joined = merged.set_index(keys).loc[positive.set_index(keys).index]
        for method in ENCDEC_BASELINE_METHODS:
            column = f"{method}_steered_generation"
            if column in positive and not joined[column].reset_index(drop=True).equals(
                positive[column].reset_index(drop=True)
            ):
                raise AxBenchAdvError(f"judged positive {method} generations differ from producer")
    for cell, generation in stored.items():
        actual = merged.iloc[cell.source_row_position][f"{cell.method}_steered_generation"]
        if actual != generation:
            raise AxBenchAdvError(f"positive published generation differs from durable cell {cell}")
    entries = load_steering_entries(judged_dir / "steering.jsonl", expected_concept_ids=range(50))
    evaluated = pd.read_parquet(judged_dir / "steering_data.parquet")
    for entry in entries:
        blocks = entry["results"]["LMJudgeEvaluator"]
        concept_rows = evaluated.loc[evaluated["concept_id"].eq(entry["concept_id"])]
        for method in ENCDEC_BASELINE_METHODS:
            block = blocks[method]
            for summary_field in (
                "relevance_concept_ratings",
                "fluency_ratings",
                "relevance_instruction_ratings",
            ):
                raw_field = f"raw_{summary_field}"
                suffix = f"LMJudgeEvaluator_{summary_field}"
                raw_values = block[raw_field]
                parquet_values = concept_rows[f"{method}_{suffix}"].tolist()
                completion_field = summary_field.replace("ratings", "completions")
                for rating, evaluated_rating, completion, evaluated_completion in zip(
                    raw_values,
                    parquet_values,
                    block[completion_field],
                    concept_rows[f"{method}_LMJudgeEvaluator_{completion_field}"],
                    strict=True,
                ):
                    # The shared validator already checks types, finiteness and
                    # matching null masks for both representations.
                    if rating is None:
                        continue
                    if not 0 <= rating <= 2 or evaluated_rating != rating:
                        raise AxBenchAdvError(
                            f"judged {method} {raw_field} is invalid or differs from evaluated Parquet"
                        )
                    if completion != evaluated_completion:
                        raise AxBenchAdvError("judge completion differs from evaluated Parquet")
                    validate_rating_completion(rating, completion)
                for index, (factor, declared_summary) in enumerate(zip(
                    METHOD_FACTORS_ENCDEC[method], block[summary_field], strict=True
                )):
                    values = [
                        float(value)
                        for value, row_factor in zip(
                            raw_values, concept_rows["factor"], strict=True
                        )
                        if row_factor == factor
                    ]
                    summary = sum(values) / 10
                    if not math.isclose(summary, declared_summary, abs_tol=1e-12, rel_tol=1e-12):
                        raise AxBenchAdvError(f"judged {method} {summary_field} summary differs")
                    # Rank the checked raw grades; tolerated serialization
                    # roundoff in a declared summary must not break their ties.
                    block[summary_field][index] = summary
    best = select_best_points(
        entries, ENCDEC_BASELINE_METHODS, method_factors=METHOD_FACTORS_ENCDEC
    )
    return EncDecSources(
        setup=setup,
        source_main=merged,
        best_factors={method: point.factor for method, point in best.items()},
        steering_prompts=prompts,
        identity=semantic,
        manifest=manifest,
        fitted=fitted,
        lora_states=lora_states,
        source_digest={
            "manifest": _sha(manifest_path),
            "resume": _sha(resume_root / "identity.json"),
            "receipt": _sha(resume_root / "fit_artifacts.json"),
            "positive_main": _sha(positive_root / "main/inference/steering_data.parquet"),
            "positive_aux": _sha(positive_root / "aux/inference/steering_data.parquet"),
            "judged_input": _sha(judged_input_path),
            "judged_jsonl": _sha(judged_dir / "steering.jsonl"),
            "judged_parquet": _sha(judged_dir / "steering_data.parquet"),
        },
    )
