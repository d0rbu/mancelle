"""Run the four encoder-decoder analogs of the upstream AxBench baselines.

Direction and LoRA artifacts are atomically published per concept and may be
resumed only through the cell store's exact semantic identity and immutable fit
receipt. Canonical Parquets remain atomic and rank zero writes the manifest
last, after both Parquets have been published.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import torch as t
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_parquet, atomic_write_text
from nanochat.tokenizer import get_tokenizer
from scripts.eval.axbench_upstream.merge_methods import (
    INPUT_COLUMNS,
    METHOD_FACTORS,
    FrameSource,
    method_columns,
    methods_for_source,
    validate_steering_frame,
)
from steering._common.ddp import rank_synchronized_stage
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_upstream.cell_store import CellResultStore, CellStoreError
from steering.eval.axbench_upstream.data import ConceptTrainingPairs, select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines.capture import (
    build_fit_row_conditioning,
    capture_post_prefix_activations,
)
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    PROBE_C,
    PROBE_MAX_ITER,
    PROBE_SEED_DOMAIN,
    PROBE_SOLVER,
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
    load_direction,
    save_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.generation import (
    DirectionGenerationRequest,
    generate_encdec_baseline,
    generate_encdec_direction_batch,
    steering_delta,
)
from steering.eval.axbench_upstream.encdec_baselines.layer_resolution import (
    VENDORED_LORA_LAYERS,
    VENDORED_QWEN_N_LAYER,
    VENDORED_STEERING_LAYER,
    resolve_vendored_layer,
    resolve_vendored_layers,
)
from steering.eval.axbench_upstream.encdec_baselines.lora import (
    LORA_ALPHA,
    LORA_BATCH_ROWS,
    LORA_DROPOUT,
    LORA_EPOCHS,
    LORA_GRAD_ACCUM,
    LORA_LR,
    LORA_RANK,
    LORA_SEED_DOMAIN,
    LORA_WEIGHT_DECAY,
    LoRATrainingRow,
    _derive_lora_seed,
    load_lora,
    save_lora,
    train_encdec_lora,
    validate_lora_state,
)
from steering.eval.axbench_upstream.encdec_baselines.prompt_recovery import (
    QUESTION_MARKER,
    PromptWrapper,
    prompt_wrapper_for_frame,
    recover_concept_steering_prompts,
    source_manifest_path,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    render_fit_row,
    template_prefix_length,
)
from steering.eval.axbench_upstream.methods import (
    TEMPERATURE,
    TOP_K,
    MethodSetup,
    PreparedPromptContext,
    assert_prepared_prompt_matches,
    load_extraction_stack,
    load_method_setup,
    load_target_model,
    prepare_prompt_context,
)
from steering.eval.axbench_upstream.profiling import cell_trace_requested, trace_cell
from steering.eval.axbench_upstream.prompt_prep import (
    RenderedPromptPlan,
    build_rendered_prompt_plan,
)
from steering.eval.axbench_upstream.run_methods import (
    _assert_disjoint_paths,
    _assert_first_concept_not_collapsed,
    _distributed_coordinates,
    _file_identity,
    _json_object,
    _manifest_header,
    _rejected_output_path,
    _repository_source_sha256,
    _runtime_identity,
    _sha256,
    _target_checkpoint_identity,
    _tokenizer_artifact_identity,
    _turn_self_conditioning_segment_geometry,
    _validate_common_scalar_config,
    _validate_prepared_manifest,
)
from steering.eval.axbench_upstream.seeds import derive_generation_seed as derive_generation_seed
from steering.eval.axbench_upstream.seeds import derive_selection_seed as derive_selection_seed
from steering.eval.axbench_upstream.work_grid import (
    CellKey,
    build_encdec_cell_grid,
    owned_encdec_cells,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.target_conditioning import validate_feature_dict
from steering.immutable_io import ImmutableArtifactError, install_bytes_once
from steering.segmentation import SegmentMode

ENCDEC_MAIN_METHODS = methods_for_source(FrameSource.ENCDEC_MAIN)
ENCDEC_AUX_METHODS = methods_for_source(FrameSource.ENCDEC_AUX)
ENCDEC_MAIN_COLUMNS = method_columns(ENCDEC_MAIN_METHODS)
ENCDEC_AUX_COLUMNS = method_columns(ENCDEC_AUX_METHODS)
METHOD_ORDER = methods_for_source(FrameSource.MERGED)
_MAIN_METHOD_FIT_KEYS: tuple[tuple[str, str], ...] = (
    ("EncDecDiffMean", "diffmean"),
    ("EncDecLinearProbe", "probe"),
)
_MAIN_METHOD_FIT_BY_METHOD = dict(_MAIN_METHOD_FIT_KEYS)
CAPTURE_PREFIX_LENGTH = 2
_SELECTION_SEED_CONTRACT = {
    "domain": "axbench-upstream-selfconditioning-selection-v1",
    "fields": "domain,base_seed,concept_id,input_id",
    "join": "\\x1f",
    "digest": "sha256 first 8 bytes big-endian",
    "modulus": "2**31 - 1",
}
_TOKEN_SEED_CONTRACT = {
    "fields": "base_seed,method,concept_id,input_id,float_factor_hex",
    "join": "\\x1f",
    "digest": "sha256 first 8 bytes big-endian",
    "modulus": "2**31 - 1",
}
_PROBE_SEED_CONTRACT = {
    "domain": PROBE_SEED_DOMAIN,
    "fields": "domain,base_seed,concept_id",
    "join": "\\x1f",
    "digest": "sha256 first 8 bytes big-endian",
    "modulus": "2**31 - 1",
}
_LORA_TRAINING_SEED_CONTRACT = {
    "domain": LORA_SEED_DOMAIN,
    "fields": "domain,base_seed,concept_id",
    "join": "\\x1f",
    "digest": "sha256 first 8 bytes big-endian",
    "modulus": "2**31 - 1",
    "consumers": (
        "CPU default generator + active accelerator generator (dropout; both RNG states "
        "restored) + isolated CPU shuffle generator"
    ),
}
_CONFIG_KEYS = frozenset(
    {
        "main_path",
        "aux_path",
        "prepared_dir",
        "output_main_path",
        "output_aux_path",
        "manifest_path",
        "train_dir",
        "target_meta_path",
        "feature_statistics_path",
        "max_concepts",
        "device",
        "seed",
        "max_new_tokens",
        "force",
        "pretrain_meta_path",
        "require_exact_statistics_identity",
    }
)


def _validate_artifact_paths(
    *,
    main_path: Path,
    aux_path: Path,
    prepared_dir: Path,
    output_main_path: Path,
    output_aux_path: Path,
    manifest_path: Path,
    train_dir: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    pretrain_meta_path: Path | None = None,
) -> tuple[Path, Path]:
    rejected_main_path = _rejected_output_path(output_main_path)
    rejected_aux_path = _rejected_output_path(output_aux_path)
    read_paths = {
        "main_path": main_path,
        "main_source_manifest": source_manifest_path(main_path),
        "aux_path": aux_path,
        "aux_source_manifest": source_manifest_path(aux_path),
        "prepared_dir": prepared_dir,
        "prepared_manifest": prepared_dir / "prepare_manifest.json",
        "prepared_metadata": prepared_dir / "generate" / "metadata.jsonl",
        "prepared_training_data": prepared_dir / "generate" / "train_data.parquet",
        "target_meta_path": target_meta_path,
        "feature_statistics_path": feature_statistics_path,
    }
    if pretrain_meta_path is not None:
        read_paths["pretrain_meta_path"] = pretrain_meta_path
    write_paths = {
        "output_main_path": output_main_path,
        "output_aux_path": output_aux_path,
        "rejected main output path": rejected_main_path,
        "rejected aux output path": rejected_aux_path,
        "manifest_path": manifest_path,
        "train_dir": train_dir,
    }
    _assert_disjoint_paths(read_paths, write_paths)
    for other_name, other_path in {**read_paths, **write_paths}.items():
        if other_name == "train_dir":
            continue
        if other_path.is_relative_to(train_dir) or train_dir.is_relative_to(other_path):
            raise ValueError(
                "train_dir must not contain or be contained by "
                f"{other_name}: {train_dir}, {other_path}"
            )
    return rejected_main_path, rejected_aux_path


def _target_sequence_len(setup: MethodSetup) -> int:
    model_config = setup.meta.get("model_config")
    sequence_len = model_config.get("sequence_len") if isinstance(model_config, dict) else None
    if type(sequence_len) is not int or sequence_len <= 0:
        raise ValueError("target metadata model_config.sequence_len must be a positive integer")
    return sequence_len


def _target_n_layer(setup: MethodSetup) -> int:
    model_config = setup.meta.get("model_config")
    n_layer = model_config.get("n_layer") if isinstance(model_config, dict) else None
    if type(n_layer) is not int or n_layer <= 0:
        raise ValueError("target metadata model_config.n_layer must be a positive integer")
    return n_layer


def _concept_names_match(
    frame: pd.DataFrame,
    training: Mapping[int, ConceptTrainingPairs],
) -> None:
    main_concepts = frame.groupby("concept_id", sort=False)["input_concept"].first().to_dict()
    for concept_id, examples in training.items():
        if main_concepts.get(concept_id) != examples.concept:
            raise ValueError(
                "prepared metadata concept does not match the main frame: "
                f"concept_id={concept_id}, prepared={examples.concept!r}, "
                f"main={main_concepts.get(concept_id)!r}"
            )


def _steered_prompt_frame(
    main_frame: pd.DataFrame,
    steering_prompts: Mapping[int, str],
) -> tuple[pd.DataFrame, dict[tuple[int, int], str]]:
    factor_one = main_frame.loc[main_frame["factor"].eq(1.0), list(INPUT_COLUMNS)].copy()
    pairs = list(
        zip(
            factor_one["concept_id"].astype(int),
            factor_one["input_id"].astype(int),
            strict=True,
        )
    )
    expected_pairs = {
        (int(concept_id), int(input_id))
        for concept_id in sorted(main_frame["concept_id"].unique())
        for input_id in sorted(
            main_frame.loc[main_frame["concept_id"].eq(concept_id), "input_id"].unique()
        )
    }
    if len(pairs) != len(set(pairs)) or set(pairs) != expected_pairs:
        raise ValueError("factor-1.0 main rows must cover every concept/input pair exactly once")

    raw_by_pair: dict[tuple[int, int], str] = {}
    raw_prompts: list[str] = []
    for row in factor_one.itertuples(index=False):
        concept_id = int(row.concept_id)
        input_id = int(row.input_id)
        raw = steering_prompts[concept_id] + QUESTION_MARKER + str(row.original_prompt)
        raw_by_pair[(concept_id, input_id)] = raw
        raw_prompts.append(raw)
    factor_one["original_prompt"] = pd.Series(
        raw_prompts,
        index=factor_one.index,
        dtype="object",
    )
    return factor_one, raw_by_pair


def _prepare_fit_rows(
    setup: MethodSetup,
    extraction_model: Any,
    sae: Any,
    tokenizer: Any,
    training: Mapping[int, ConceptTrainingPairs],
    *,
    sequence_len: int,
) -> dict[int, dict[str, tuple[LoRATrainingRow, ...]]]:
    result: dict[int, dict[str, tuple[LoRATrainingRow, ...]]] = {}
    for concept_id, examples in training.items():
        classes: dict[str, tuple[LoRATrainingRow, ...]] = {}
        for class_name, pairs in (("positive", examples.positive), ("negative", examples.negative)):
            rows: list[LoRATrainingRow] = []
            for row_index, (input_text, output_text) in enumerate(pairs):
                surface = f"concept {concept_id} {class_name} row {row_index}"
                rendered = render_fit_row(
                    tokenizer,
                    input_text=input_text,
                    output_text=output_text,
                    sequence_len=sequence_len,
                    surface=surface,
                )
                conditioning = build_fit_row_conditioning(
                    setup,
                    extraction_model,
                    sae,
                    tokenizer,
                    token_ids=rendered.ids,
                    surface=surface,
                )
                rows.append(LoRATrainingRow(rendered=rendered, conditioning=conditioning))
            classes[class_name] = tuple(rows)
        result[concept_id] = classes
    return result


def _validate_prepared_context(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    prepared: PreparedPromptContext,
    prompt: str,
    *,
    max_new_tokens: int,
) -> None:
    assert_prepared_prompt_matches(tokenizer, prompt, prepared)
    if len(prepared.token_ids) + max_new_tokens > int(model.dec_config.sequence_len):
        raise ValueError("prepared prompt plus generation budget exceeds decoder sequence length")
    if not prepared.segments:
        raise ValueError("prepared prompt has no real feature segments")
    for segment_index, segment in enumerate(prepared.segments):
        selected = {
            int(feature_id): int(bucket)
            for feature_id, bucket in zip(
                segment.feature_ids,
                segment.buckets,
                strict=True,
            )
        }
        if len(selected) != len(segment.feature_ids):
            raise ValueError(
                f"prepared prompt segment {segment_index} contains duplicate feature IDs"
            )
        validate_feature_dict(
            selected,
            setup.n_sae_features,
            2,
            name=f"prepared prompt segment {segment_index}",
            feature_vocab=setup.feature_vocab,
        )


def _prepare_all(
    setup: MethodSetup,
    extraction_model: Any,
    sae: Any,
    tokenizer: Any,
    plan: RenderedPromptPlan,
) -> dict[tuple[int, ...], PreparedPromptContext]:
    return {
        exact_tokens: prepare_prompt_context(
            setup,
            extraction_model,
            sae,
            tokenizer,
            token_ids=exact_tokens,
            assistant_end_id=assistant_end_id,
        )
        for exact_tokens, (_prompt, assistant_end_id) in plan.prompt_by_tokens.items()
    }


def _validate_all(
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    plan: RenderedPromptPlan,
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    *,
    max_new_tokens: int,
) -> None:
    for exact_tokens, prepared in prepared_by_tokens.items():
        prompt, _assistant_end_id = plan.prompt_by_tokens[exact_tokens]
        _validate_prepared_context(
            setup,
            model,
            tokenizer,
            prepared,
            prompt,
            max_new_tokens=max_new_tokens,
        )


# Rejected-frame bytes differ intentionally from run_methods.py; keep this writer local.
def _write_rejected_frame(frame: pd.DataFrame, rejected_path: Path, *, label: str) -> None:
    atomic_write_parquet(
        pa.Table.from_pandas(frame, preserve_index=False),
        rejected_path,
    )
    logger.error("Complete rejected {} frame written to {}", label, rejected_path)


def _validate_with_rejected_fallback(
    frame: pd.DataFrame,
    *,
    source: str,
    max_concepts: int,
    rejected_path: Path,
) -> None:
    try:
        validate_steering_frame(frame, source=source, max_concepts=max_concepts)
    except ValueError as exc:
        try:
            _write_rejected_frame(frame, rejected_path, label=source)
        except BaseException as rejected_exc:
            exc.add_note(
                f"failed to write rejected {source} frame to {rejected_path}: "
                f"{type(rejected_exc).__name__}: {rejected_exc}"
            )
            raise exc from rejected_exc
        exc.add_note(f"complete rejected {source} frame written to {rejected_path}")
        raise


def _artifact_path(train_dir: Path, concept_id: int, kind: str) -> Path:
    return train_dir / f"concept_{concept_id:05d}_{kind}.safetensors"


def _expected_artifacts(
    train_dir: Path,
    *,
    max_concepts: int,
) -> list[tuple[Path, str, int]]:
    artifacts: list[tuple[Path, str, int]] = []
    for concept_id in range(max_concepts):
        artifacts.extend(
            (
                (
                    _artifact_path(train_dir, concept_id, "diffmean"),
                    "EncDecDiffMean",
                    concept_id,
                ),
                (
                    _artifact_path(train_dir, concept_id, "linear_probe"),
                    "EncDecLinearProbe",
                    concept_id,
                ),
                (
                    _artifact_path(train_dir, concept_id, "lora"),
                    "EncDecLoRA",
                    concept_id,
                ),
            )
        )
    return artifacts


def _validate_train_resume_dir(train_dir: Path, *, max_concepts: int) -> tuple[Path, ...]:
    if not train_dir.exists():
        return ()
    if not train_dir.is_dir():
        raise FileExistsError(
            f"encoder-decoder baseline train output directory must be empty or a "
            f"validated resume directory: {train_dir}"
        )
    expected_names = {
        path.name
        for path, _method, _concept_id in _expected_artifacts(train_dir, max_concepts=max_concepts)
    }
    existing: list[Path] = []
    for path in train_dir.iterdir():
        if path.is_symlink() or not path.is_file() or path.name not in expected_names:
            raise FileExistsError(
                "encoder-decoder baseline train output directory must be empty or contain "
                f"only exact resumable artifacts; found {path}"
            )
        existing.append(path)
    return tuple(sorted(existing))


def _lora_training_metadata(
    *,
    layers: Sequence[int],
    base_seed: int,
    concept_id: int,
    n_rows: int,
) -> dict[str, Any]:
    n_batches = math.ceil(n_rows / LORA_BATCH_ROWS)
    optimizer_steps = LORA_EPOCHS * math.ceil(n_batches / LORA_GRAD_ACCUM)
    scheduler_horizon = LORA_EPOCHS * max(1, math.ceil(n_batches / LORA_GRAD_ACCUM))
    return {
        "rank": LORA_RANK,
        "alpha": LORA_ALPHA,
        "scaling": LORA_ALPHA / math.sqrt(LORA_RANK),
        "dropout": LORA_DROPOUT,
        "layers": list(layers),
        "seed": _derive_lora_seed(base_seed, concept_id),
        "n_rows": n_rows,
        "epochs": LORA_EPOCHS,
        "batch_rows": LORA_BATCH_ROWS,
        "gradient_accumulation_batches": LORA_GRAD_ACCUM,
        "learning_rate": LORA_LR,
        "weight_decay": LORA_WEIGHT_DECAY,
        "optimizer_steps": optimizer_steps,
        "scheduler_horizon": scheduler_horizon,
    }


def _artifact_inventory(
    artifacts: Sequence[tuple[Path, str, int]],
    *,
    train_dir: Path,
) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(train_dir).as_posix(),
            "method": method,
            "concept_id": concept_id,
            "sha256": _sha256(path),
        }
        for path, method, concept_id in artifacts
    ]


def _encdec_resume_identity(
    *,
    main_path: Path,
    aux_path: Path,
    prepared_dir: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    pretrain_meta_path: Path | None,
    tokenizer_identity: Mapping[str, int | str],
    max_concepts: int,
    device: str,
    seed: int,
    max_new_tokens: int,
    require_exact_statistics_identity: bool,
) -> dict[str, Any]:
    header = _manifest_header()
    return {
        "driver": "encdec",
        "schema_version": 1,
        **header,
        "repository_source_sha256": _repository_source_sha256(),
        "inputs": {
            "main_sha256": _sha256(main_path),
            "main_source_manifest_sha256": _sha256(source_manifest_path(main_path)),
            "aux_sha256": _sha256(aux_path),
            "aux_source_manifest_sha256": _sha256(source_manifest_path(aux_path)),
            "prepared_manifest_sha256": _sha256(prepared_dir / "prepare_manifest.json"),
            "prepared_metadata": _file_identity(prepared_dir / "generate" / "metadata.jsonl"),
            "prepared_training_data": _file_identity(
                prepared_dir / "generate" / "train_data.parquet"
            ),
            "target_meta_sha256": _sha256(target_meta_path),
            "target_model": _target_checkpoint_identity(target_meta_path),
            "pretrain_meta_sha256": (
                None if pretrain_meta_path is None else _sha256(pretrain_meta_path)
            ),
            "feature_statistics_sha256": _sha256(feature_statistics_path),
            "tokenizer": dict(tokenizer_identity),
        },
        "configuration": {
            "max_concepts": max_concepts,
            "device_type": device.split(":", 1)[0],
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "temperature_hex": float(TEMPERATURE).hex(),
            "top_k": TOP_K,
            "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
            "require_exact_statistics_identity": require_exact_statistics_identity,
            "layer_resolution": {
                "vendored_qwen_n_layer": VENDORED_QWEN_N_LAYER,
                "vendored_steering_layer": VENDORED_STEERING_LAYER,
                "vendored_lora_layers": list(VENDORED_LORA_LAYERS),
                "formula": "round(target_n_layer * vendored_layer / vendored_qwen_n_layer)",
            },
            "probe": {
                "C_hex": float(PROBE_C).hex(),
                "max_iter": PROBE_MAX_ITER,
                "solver": PROBE_SOLVER,
                "seed_domain": PROBE_SEED_DOMAIN,
            },
            "lora": {
                "rank": LORA_RANK,
                "alpha_hex": float(LORA_ALPHA).hex(),
                "dropout_hex": float(LORA_DROPOUT).hex(),
                "epochs": LORA_EPOCHS,
                "batch_rows": LORA_BATCH_ROWS,
                "gradient_accumulation_batches": LORA_GRAD_ACCUM,
                "learning_rate_hex": float(LORA_LR).hex(),
                "weight_decay_hex": float(LORA_WEIGHT_DECAY).hex(),
                "seed_domain": LORA_SEED_DOMAIN,
            },
        },
        "runtime": _runtime_identity(device),
    }


def _install_fit_artifact_receipt(
    *,
    resume_root: Path,
    semantic_fingerprint: str,
    artifacts: Sequence[tuple[Path, str, int]],
    train_dir: Path,
) -> None:
    payload = {
        "schema_version": 1,
        "semantic_fingerprint": semantic_fingerprint,
        "files": _artifact_inventory(artifacts, train_dir=train_dir),
    }
    encoded = (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    try:
        install_bytes_once(
            encoded,
            resume_root / "fit_artifacts.json",
            surface="AxBench-Upstream encoder-decoder fit-artifact receipt",
        )
    except ImmutableArtifactError as exc:
        raise CellStoreError(str(exc)) from exc


def _load_fit_artifacts(
    *,
    train_dir: Path,
    max_concepts: int,
    seed: int,
    intervention_layer: int,
    lora_layers: Sequence[int],
    fit_rows: Mapping[int, Mapping[str, Sequence[LoRATrainingRow]]],
    model: Any,
) -> tuple[
    dict[int, dict[str, FittedDirection]],
    dict[int, dict[str, t.Tensor]],
    dict[int, dict[str, Any]],
    list[tuple[Path, str, int]],
]:
    artifacts = _expected_artifacts(train_dir, max_concepts=max_concepts)
    expected_names = {path.name for path, _method, _concept_id in artifacts}
    actual_names = {path.name for path in train_dir.iterdir() if path.is_file()}
    if actual_names != expected_names:
        raise CellStoreError(
            "encoder-decoder fit artifact set is incomplete or unexpected: "
            f"missing={sorted(expected_names - actual_names)}, "
            f"unexpected={sorted(actual_names - expected_names)}"
        )
    fitted: dict[int, dict[str, FittedDirection]] = {}
    lora_states: dict[int, dict[str, t.Tensor]] = {}
    lora_metadata: dict[int, dict[str, Any]] = {}
    for concept_id in range(max_concepts):
        try:
            fitted[concept_id] = {
                "diffmean": load_direction(_artifact_path(train_dir, concept_id, "diffmean")),
                "probe": load_direction(_artifact_path(train_dir, concept_id, "linear_probe")),
            }
        except Exception as exc:
            raise CellStoreError(
                f"could not load encoder-decoder direction artifacts for concept "
                f"{concept_id}: {exc}"
            ) from exc
        _validate_fitted_direction_layers(
            fitted[concept_id],
            concept_id=concept_id,
            intervention_layer=intervention_layer,
            expected_width=int(model.dec_config.n_embd),
        )
        lora_path = _artifact_path(train_dir, concept_id, "lora")
        try:
            lora_state, stored_metadata = load_lora(lora_path)
            validate_lora_state(model, lora_state, layers=lora_layers)
        except Exception as exc:
            raise CellStoreError(f"could not load valid LoRA artifact {lora_path}: {exc}") from exc
        expected_metadata = _lora_training_metadata(
            layers=lora_layers,
            base_seed=seed,
            concept_id=concept_id,
            n_rows=len(fit_rows[concept_id]["positive"]),
        )
        serialized_expected = {key: str(value) for key, value in expected_metadata.items()}
        if stored_metadata != serialized_expected:
            raise CellStoreError(
                f"{lora_path}: LoRA metadata differs from the current fit contract: "
                f"expected={serialized_expected!r}, actual={stored_metadata!r}"
            )
        lora_states[concept_id] = lora_state
        lora_metadata[concept_id] = expected_metadata
    return fitted, lora_states, lora_metadata, artifacts


def _reset_compiler_state_before_generation() -> None:
    """Discard fit-specialized compile state before inference begins.

    Fit capture and LoRA training exercise the module-global compiled packed
    cross-attention callable with variable training shapes and gradient modes.
    Those specializations can exhaust Dynamo's recompile limit, making the
    later autoregressive path fall back to unfused eager FlexAttention.  The
    public reset clears only process-local compiler state; Inductor's
    filesystem cache remains available when inference specializes afresh.
    """

    with profile_region("axbench_upstream.encdec.compiler_phase_reset"):
        t.compiler.reset()


def _validate_fitted_direction_layers(
    fitted: Mapping[str, FittedDirection],
    *,
    concept_id: int,
    intervention_layer: int,
    expected_width: int,
) -> None:
    expected_methods = {"diffmean": "diffmean", "probe": "probe"}
    if set(fitted) != set(expected_methods):
        raise CellStoreError(
            f"concept {concept_id} direction set differs: "
            f"missing={sorted(set(expected_methods) - set(fitted))}, "
            f"unexpected={sorted(set(fitted) - set(expected_methods))}"
        )
    for fit_key, direction in fitted.items():
        if direction.metadata.get("method") != expected_methods[fit_key]:
            raise CellStoreError(
                f"concept {concept_id} {fit_key} direction metadata has method="
                f"{direction.metadata.get('method')!r}, expected {expected_methods[fit_key]!r}"
            )
        if direction.metadata.get("intervention_layer") != intervention_layer:
            raise CellStoreError(
                f"concept {concept_id} {fit_key} direction metadata has intervention_layer="
                f"{direction.metadata.get('intervention_layer')!r}, expected "
                f"{intervention_layer}"
            )
        if (
            direction.direction.dtype is not t.float32
            or direction.direction.device.type != "cpu"
            or direction.direction.shape != (expected_width,)
            or not direction.direction.is_contiguous()
            or not bool(t.isfinite(direction.direction).all())
            or not math.isfinite(direction.max_activation)
            or direction.max_activation < 0.0
        ):
            raise CellStoreError(
                f"concept {concept_id} {fit_key} direction violates the target-model contract: "
                f"expected_shape={(expected_width,)}, "
                f"actual_shape={tuple(direction.direction.shape)}, "
                f"dtype={direction.direction.dtype}, device={direction.direction.device}, "
                f"contiguous={direction.direction.is_contiguous()}, "
                f"max_activation={direction.max_activation!r}"
            )


def _generate_encdec_cell(
    cell: CellKey,
    *,
    rows: Sequence[Any],
    prompt_plan: RenderedPromptPlan,
    steered_plan: RenderedPromptPlan,
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    steered_prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    fitted: Mapping[int, Mapping[str, FittedDirection]],
    lora_states: Mapping[int, dict[str, t.Tensor]],
    intervention_layer: int,
    lora_layers: Sequence[int],
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> str:
    concept_id, input_id, factor = cell.validated_source_values(rows)
    pair = (concept_id, input_id)
    exact_tokens = prompt_plan.rendered_tokens_by_pair[pair]
    prepared = prepared_by_tokens[exact_tokens]
    if cell.method in _MAIN_METHOD_FIT_BY_METHOD:
        fit_key = _MAIN_METHOD_FIT_BY_METHOD[cell.method]
        fit = fitted[concept_id][fit_key]
        prepared_for_cell = prepared
        delta = steering_delta(
            model,
            direction=fit.direction,
            coefficient=factor * fit.max_activation,
            layer=intervention_layer,
        )
    elif cell.method == "EncDecPromptSteering":
        steered_tokens = steered_plan.rendered_tokens_by_pair[pair]
        prepared_for_cell = steered_prepared_by_tokens[steered_tokens]
        delta = nullcontext()
    elif cell.method == "EncDecLoRA":
        prepared_for_cell = prepared
        delta = steering_delta(
            model,
            lora_state=lora_states[concept_id],
            lora_layers=lora_layers,
        )
    else:
        raise AssertionError(f"unsupported encoder-decoder method in cell grid: {cell.method}")
    with trace_cell(cell):
        with profile_region(
            "axbench_upstream.cell",
            driver="encdec",
            method=cell.method,
            concept_id=concept_id,
            input_id=input_id,
            factor_hex=cell.factor_hex,
            source_row_position=cell.source_row_position,
        ):
            with delta:
                return generate_encdec_baseline(
                    setup,
                    model,
                    tokenizer,
                    prepared_for_cell,
                    selection_seed=cell.selection_seed,
                    seed=cell.token_seed,
                    max_new_tokens=max_new_tokens,
                )


def _generate_encdec_direction_cells(
    cells: Sequence[CellKey],
    *,
    rows: Sequence[Any],
    prompt_plan: RenderedPromptPlan,
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    fitted: Mapping[int, Mapping[str, FittedDirection]],
    intervention_layer: int,
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[str, ...]:
    """Generate one same-prompt group of untraced direction cells together."""
    grouped = tuple(cells)
    if len(grouped) < 2:
        raise ValueError("direction-cell batching requires at least two cells")
    pair = (grouped[0].concept_id, grouped[0].input_id)
    exact_tokens = prompt_plan.rendered_tokens_by_pair[pair]
    prepared = prepared_by_tokens[exact_tokens]
    requests: list[DirectionGenerationRequest] = []
    for cell in grouped:
        if cell.method not in _MAIN_METHOD_FIT_BY_METHOD:
            raise ValueError(f"non-direction cell reached direction batch: {cell}")
        if (cell.concept_id, cell.input_id) != pair:
            raise ValueError(
                "direction-cell batch must share one concept/input pair; "
                f"expected {pair}, got {(cell.concept_id, cell.input_id)}"
            )
        concept_id, _input_id, factor = cell.validated_source_values(rows)
        fit = fitted[concept_id][_MAIN_METHOD_FIT_BY_METHOD[cell.method]]
        requests.append(
            DirectionGenerationRequest(
                direction=fit.direction,
                coefficient=factor * fit.max_activation,
                layer=intervention_layer,
                selection_seed=cell.selection_seed,
                token_seed=cell.token_seed,
            )
        )

    with profile_region(
        "axbench_upstream.cell_batch",
        driver="encdec",
        concept_id=pair[0],
        input_id=pair[1],
        physical_batch_size=len(grouped),
    ):
        return generate_encdec_direction_batch(
            setup,
            model,
            tokenizer,
            prepared,
            requests,
            max_new_tokens=max_new_tokens,
        )


def _append_missing_encdec_cells(
    store: CellResultStore,
    cells: Sequence[CellKey],
    *,
    rows: Sequence[Any],
    prompt_plan: RenderedPromptPlan,
    steered_plan: RenderedPromptPlan,
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    steered_prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    fitted: Mapping[int, Mapping[str, FittedDirection]],
    lora_states: Mapping[int, dict[str, t.Tensor]],
    intervention_layer: int,
    lora_layers: Sequence[int],
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> None:
    """Append missing cells, keeping unsafe or traced prompt groups scalar."""
    missing = store.missing(cells)
    boundary_only_batch_safe = setup.segment_mode is SegmentMode.fixed_tokens
    traced_pairs = {
        (cell.concept_id, cell.input_id) for cell in missing if cell_trace_requested(cell)
    }
    # Turn mode stays on _generate_encdec_cell: its steering_delta spans the
    # complete scalar generate_stream call, which also owns structural stops.
    batchable = {
        cell
        for cell in missing
        if boundary_only_batch_safe
        and cell.method in _MAIN_METHOD_FIT_BY_METHOD
        and (cell.concept_id, cell.input_id) not in traced_pairs
    }
    direction_groups: dict[tuple[int, int], list[CellKey]] = {}
    for cell in missing:
        if cell in batchable:
            direction_groups.setdefault((cell.concept_id, cell.input_id), []).append(cell)

    completed: set[CellKey] = set()
    for cell in missing:
        if cell in completed:
            continue
        group = direction_groups.get((cell.concept_id, cell.input_id), [])
        if cell in batchable and len(group) >= 2:
            generations = _generate_encdec_direction_cells(
                group,
                rows=rows,
                prompt_plan=prompt_plan,
                prepared_by_tokens=prepared_by_tokens,
                fitted=fitted,
                intervention_layer=intervention_layer,
                setup=setup,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            if len(generations) != len(group):
                raise RuntimeError(
                    "direction batch returned the wrong number of generations: "
                    f"expected {len(group)}, got {len(generations)}"
                )
            for grouped_cell, generation in zip(group, generations, strict=True):
                store.append(grouped_cell, generation)
            completed.update(group)
            continue

        store.append(
            cell,
            _generate_encdec_cell(
                cell,
                rows=rows,
                prompt_plan=prompt_plan,
                steered_plan=steered_plan,
                prepared_by_tokens=prepared_by_tokens,
                steered_prepared_by_tokens=steered_prepared_by_tokens,
                fitted=fitted,
                lora_states=lora_states,
                intervention_layer=intervention_layer,
                lora_layers=lora_layers,
                setup=setup,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            ),
        )
        completed.add(cell)


def _manifest(
    *,
    setup: MethodSetup,
    prepared_manifest: dict[str, Any],
    config: dict[str, Any],
    main_path: Path,
    aux_path: Path,
    feature_statistics_path: Path,
    prompt_wrapper: PromptWrapper,
    target_n_layer: int,
    intervention_layer: int,
    lora_layers: Sequence[int],
    fitted: Mapping[int, Mapping[str, FittedDirection]],
    lora_metadata: Mapping[int, dict[str, Any]],
    steering_prompts: Mapping[int, str],
    train_dir: Path,
    artifacts: Sequence[tuple[Path, str, int]],
) -> dict[str, Any]:
    dataset_revision = _validate_prepared_manifest(prepared_manifest)
    user_config = setup.meta.get("user_config")
    model_tag = user_config.get("model_tag") if isinstance(user_config, dict) else None
    layer_resolution = {
        "vendored_qwen_n_layer": VENDORED_QWEN_N_LAYER,
        "vendored_steering_layer": VENDORED_STEERING_LAYER,
        "vendored_lora_layers": list(VENDORED_LORA_LAYERS),
        "target_n_layer": target_n_layer,
        "formula": "round(target_n_layer * vendored_layer / vendored_qwen_n_layer)",
        "resolved_intervention_layer": intervention_layer,
        "resolved_lora_layers": list(lora_layers),
    }
    shared_direction = {
        "intervention_layer": intervention_layer,
        "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
        "pooling_population": (
            "all rendered fit-row token positions at or after prefix_length, "
            "including template boundary and assistant-end tokens"
        ),
        "selection_seed": _SELECTION_SEED_CONTRACT,
        "token_seed": _TOKEN_SEED_CONTRACT,
    }
    turn_geometry = _turn_self_conditioning_segment_geometry(setup)
    batched_decode_note = (
        "independent B=1 prefills; compatible direction cells use bounded B>1 "
        "decode; PromptSteering, LoRA, and singleton resume remainders stay scalar; "
        "top_k=50 with per-cell SHA-256 seeds"
        if turn_geometry is None
        else "independent B=1 prefills; every cell stays scalar because turn-delimited "
        "conditioning can refresh mid-completion, so the bounded B>1 decode never runs; "
        "top_k=50 with per-cell SHA-256 seeds"
    )
    return {
        "schema_version": 1,
        **_manifest_header(),
        "target": {
            "checkpoint_meta": str(setup.target_meta_path),
            "model_tag": model_tag,
            "step": setup.meta.get("step"),
        },
        "dataset_revision": dataset_revision,
        "prepared_manifest": prepared_manifest,
        "configuration": config,
        "layer_resolution": layer_resolution,
        "generation": {"temperature": TEMPERATURE, "top_k": TOP_K},
        **({} if turn_geometry is None else {"self_conditioning_segment_geometry": turn_geometry}),
        "inputs": {
            "main_steering_parquet_sha256": _sha256(main_path),
            "main_source_manifest_sha256": _sha256(source_manifest_path(main_path)),
            "aux_steering_parquet_sha256": _sha256(aux_path),
            "aux_source_manifest_sha256": _sha256(source_manifest_path(aux_path)),
            "feature_statistics_sha256": _sha256(feature_statistics_path),
        },
        "method_order": list(METHOD_ORDER),
        "methods": {
            "EncDecDiffMean": {
                "factors": list(METHOD_FACTORS["EncDecDiffMean"]),
                "strength_semantics": "factor_times_fit_row_max_activation",
                **shared_direction,
                "formula": (
                    "direction = (token_mean_pos - token_mean_neg) / (norm + f32_eps); "
                    "steer h += factor*max_act*direction at the resolved intervention_layer"
                ),
                "max_activation_by_concept": {
                    str(concept_id): repr(float(values["diffmean"].max_activation))
                    for concept_id, values in fitted.items()
                },
            },
            "EncDecLinearProbe": {
                "factors": list(METHOD_FACTORS["EncDecLinearProbe"]),
                "strength_semantics": "factor_times_fit_row_max_activation",
                **shared_direction,
                "formula": (
                    "direction = F.normalize(L1 liblinear_logreg_coef(row_mean_activations)); "
                    "steer identical to diffmean"
                ),
                "probe_seed": _PROBE_SEED_CONTRACT,
                "estimator": {
                    "solver": PROBE_SOLVER,
                    "l1_ratio": "1.0 (L1)",
                    "C": PROBE_C,
                    "max_iter": PROBE_MAX_ITER,
                    "random_state_domain": PROBE_SEED_DOMAIN,
                    "class_weight": None,
                },
                "max_activation_by_concept": {
                    str(concept_id): repr(float(values["probe"].max_activation))
                    for concept_id, values in fitted.items()
                },
            },
            "EncDecPromptSteering": {
                "factors": list(METHOD_FACTORS["EncDecPromptSteering"]),
                "strength_semantics": "declared_factor_1.0",
                "selection_seed": _SELECTION_SEED_CONTRACT,
                "token_seed": _TOKEN_SEED_CONTRACT,
                "question_marker": QUESTION_MARKER,
                "source_prompt_wrapper": {
                    "model": prompt_wrapper.model.value,
                    "revision": prompt_wrapper.revision,
                },
                "recovery_rule": (
                    "select the exact vendored wrapper from source-manifest model/revision "
                    "provenance and recover one shared nonblank steering prompt per concept; "
                    "compose prompt + QUESTION_MARKER + original_prompt"
                ),
                "steering_prompt_sha256_by_concept": {
                    str(concept_id): hashlib.sha256(prompt.encode("utf-8")).hexdigest()
                    for concept_id, prompt in steering_prompts.items()
                },
            },
            "EncDecLoRA": {
                "factors": list(METHOD_FACTORS["EncDecLoRA"]),
                "strength_semantics": "declared_factor_1.0",
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "scaling": LORA_ALPHA / math.sqrt(LORA_RANK),
                "scaling_formula": "scaling = alpha / sqrt(rank)",
                "forward_formula": (
                    "output = base(x) + (B @ A @ dropout(x_fp32)) * scaling, "
                    "cast back to the activation dtype (PEFT 0.13.2 dtype boundary)"
                ),
                "training_seed": _LORA_TRAINING_SEED_CONTRACT,
                "dropout": LORA_DROPOUT,
                "layers": list(lora_layers),
                "epochs": LORA_EPOCHS,
                "batch_rows": LORA_BATCH_ROWS,
                "gradient_accumulation_batches": LORA_GRAD_ACCUM,
                "learning_rate": LORA_LR,
                "weight_decay": LORA_WEIGHT_DECAY,
                "selection_seed": _SELECTION_SEED_CONTRACT,
                "token_seed": _TOKEN_SEED_CONTRACT,
                "training_by_concept": {
                    str(concept_id): metadata for concept_id, metadata in lora_metadata.items()
                },
            },
        },
        "divergence_notes": {
            "1_target_model_template_engine_environment": "forced by encoder-decoder architecture",
            "2_conditioned_fits_self_conditioned_evaluation": "owner-ratified operating point",
            "3_per_cell_generation_and_sha256_seeds": batched_decode_note,
            "4_fit_row_max_activation": "calibrated on fit rows rather than a latent stage",
            "5_nonpositive_max_activation_raises": "no silent fallback to 50",
            "6_fp32_accumulation": "mean and loss accumulation use fp32",
            "7_cached_probe_activations": "frozen-base equivalent cached fit",
            "8_probe_intercept_semantics": (
                "the L1 estimator fits an intercept used only during training; "
                "steering projects onto the normalized coefficient vector alone"
            ),
            "9_strength_only_schema": "no perplexity columns",
            "10_lora_b1_token_mean": "six B=1 sum-loss forwards divided by supervised tokens",
            "11_seeded_shuffles": "deterministic seeded row order",
            "12_probe_estimator": (
                "probe estimator = project row-mean L1 logistic regression "
                "(vendored GD probe unlearnable at this activation scale; AUC evidence above)"
            ),
        },
        "train_artifacts": {
            "directory": str(train_dir),
            "files": _artifact_inventory(artifacts, train_dir=train_dir),
        },
    }


def run_encdec_baselines(
    *,
    main_path: Path,
    aux_path: Path,
    prepared_dir: Path,
    output_main_path: Path,
    output_aux_path: Path,
    manifest_path: Path,
    train_dir: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    max_concepts: int,
    device: str,
    seed: int,
    max_new_tokens: int,
    force: bool = False,
    pretrain_meta_path: Path | None = None,
    require_exact_statistics_identity: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit, train, and generate all four encoder-decoder baseline analogs."""

    _validate_common_scalar_config(
        max_concepts=max_concepts,
        device=device,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    if type(force) is not bool:
        raise ValueError("force must be a boolean")
    if type(require_exact_statistics_identity) is not bool:
        raise ValueError("require_exact_statistics_identity must be a boolean")
    rank, world_size = _distributed_coordinates()
    stage_device = device if device.startswith("cuda") else "cpu"
    with rank_synchronized_stage("encoder-decoder input preflight", device=stage_device):
        main_path = Path(main_path).expanduser().resolve()
        aux_path = Path(aux_path).expanduser().resolve()
        prepared_dir = Path(prepared_dir).expanduser().resolve()
        output_main_path = Path(output_main_path).expanduser().resolve()
        output_aux_path = Path(output_aux_path).expanduser().resolve()
        manifest_path = Path(manifest_path).expanduser().resolve()
        train_dir = Path(train_dir).expanduser().resolve()
        target_meta_path = Path(target_meta_path).expanduser().resolve()
        feature_statistics_path = Path(feature_statistics_path).expanduser().resolve()
        pretrain_meta_path = (
            None if pretrain_meta_path is None else Path(pretrain_meta_path).expanduser().resolve()
        )
        rejected_main_path, rejected_aux_path = _validate_artifact_paths(
            main_path=main_path,
            aux_path=aux_path,
            prepared_dir=prepared_dir,
            output_main_path=output_main_path,
            output_aux_path=output_aux_path,
            manifest_path=manifest_path,
            train_dir=train_dir,
            target_meta_path=target_meta_path,
            feature_statistics_path=feature_statistics_path,
            pretrain_meta_path=pretrain_meta_path,
        )
        existing = [
            path for path in (output_main_path, output_aux_path, manifest_path) if path.exists()
        ]
        if existing and not force:
            raise FileExistsError(
                f"encoder-decoder baseline output already exists: {existing}; "
                "pass force=true to rebuild"
            )
        existing_train_artifacts = _validate_train_resume_dir(
            train_dir,
            max_concepts=max_concepts,
        )

        prepared_manifest = _json_object(prepared_dir / "prepare_manifest.json")
        _validate_prepared_manifest(prepared_manifest)
        main_frame = pd.read_parquet(main_path)
        validate_steering_frame(
            main_frame,
            source="main",
            max_concepts=max_concepts,
            check_method_values=False,
        )
        aux_frame = pd.read_parquet(aux_path)
        validate_steering_frame(
            aux_frame,
            source="aux",
            max_concepts=max_concepts,
            check_method_values=False,
        )
        encdec_main = main_frame.loc[:, list(INPUT_COLUMNS)].copy()
        training = select_training_pairs(prepared_dir, max_concepts=max_concepts)
        _concept_names_match(encdec_main, training)
        main_prompt_wrapper = prompt_wrapper_for_frame(main_path)
        aux_prompt_wrapper = prompt_wrapper_for_frame(aux_path)
        if main_prompt_wrapper != aux_prompt_wrapper:
            raise ValueError(
                "main and auxiliary AxBench frames have different prompt-wrapper provenance: "
                f"main={main_prompt_wrapper.detected_model}, "
                f"aux={aux_prompt_wrapper.detected_model}"
            )
        steering_prompts = recover_concept_steering_prompts(
            aux_frame,
            max_concepts=max_concepts,
            wrapper=aux_prompt_wrapper,
        )

        nanochat_tokenizer = get_tokenizer()
        tokenizer_identity = _tokenizer_artifact_identity()
        prefix_length = template_prefix_length(nanochat_tokenizer)
        if prefix_length != CAPTURE_PREFIX_LENGTH:
            raise ValueError(
                "AxBench encoder-decoder baseline template prefix length must equal "
                f"{CAPTURE_PREFIX_LENGTH}, got {prefix_length}"
            )
        prompt_plan = build_rendered_prompt_plan(
            encdec_main,
            nanochat_tokenizer,
            declared_factors=METHOD_FACTORS[ENCDEC_MAIN_METHODS[0]],
            render=render_prompt_for_chat_completion,
        )
        steered_frame, raw_steered_by_pair = _steered_prompt_frame(
            encdec_main,
            steering_prompts,
        )
        steered_plan = build_rendered_prompt_plan(
            steered_frame,
            nanochat_tokenizer,
            declared_factors=(1.0,),
            render=render_prompt_for_chat_completion,
        )

        rows = prompt_plan.rows
        first_concept_id = int(rows[0].concept_id)
        cells = build_encdec_cell_grid(encdec_main, seed=seed)
        first_cells = tuple(cell for cell in cells if cell.concept_id == first_concept_id)
        resume_root = manifest_path.parent / "_encdec_cell_resume_v1"
        if existing_train_artifacts and not (resume_root / "identity.json").is_file():
            raise FileExistsError(
                "encoder-decoder baseline train output directory contains artifacts without "
                f"a bound resume identity: {train_dir}"
            )
        semantic_identity = _encdec_resume_identity(
            main_path=main_path,
            aux_path=aux_path,
            prepared_dir=prepared_dir,
            target_meta_path=target_meta_path,
            feature_statistics_path=feature_statistics_path,
            pretrain_meta_path=pretrain_meta_path,
            tokenizer_identity=tokenizer_identity,
            max_concepts=max_concepts,
            device=device,
            seed=seed,
            max_new_tokens=max_new_tokens,
            require_exact_statistics_identity=require_exact_statistics_identity,
        )
    with rank_synchronized_stage(
        "encoder-decoder checkpoint geometry validation",
        device=stage_device,
    ):
        # Resolve every checkpoint witness, including turn cadence, before the
        # resume store creates directories or publishes identity.json.
        setup = load_method_setup(
            target_meta_path,
            feature_statistics_path,
            pretrain_meta_path=pretrain_meta_path,
            require_exact_statistics_identity=require_exact_statistics_identity,
        )

    store: CellResultStore | None = None
    with rank_synchronized_stage(
        "encoder-decoder resume initialization",
        device=stage_device,
    ):
        if rank == 0:
            store = CellResultStore(
                root=resume_root,
                cells=cells,
                tripwire_cells=first_cells,
                semantic_identity=semantic_identity,
                rank=rank,
                world_size=world_size,
            )
    with rank_synchronized_stage("encoder-decoder resume validation", device=stage_device):
        if rank != 0:
            store = CellResultStore(
                root=resume_root,
                cells=cells,
                tripwire_cells=first_cells,
                semantic_identity=semantic_identity,
                rank=rank,
                world_size=world_size,
            )
    assert store is not None

    fit_rows: dict[int, dict[str, tuple[LoRATrainingRow, ...]]] = {}
    extraction_model: Any | None = None
    sae: Any | None = None
    try:
        with rank_synchronized_stage(
            "encoder-decoder conditioning preparation",
            device=stage_device,
        ):
            sequence_len = _target_sequence_len(setup)
            target_n_layer = _target_n_layer(setup)
            intervention_layer = resolve_vendored_layer(
                VENDORED_STEERING_LAYER,
                target_n_layer=target_n_layer,
            )
            lora_layers = resolve_vendored_layers(
                VENDORED_LORA_LAYERS,
                target_n_layer=target_n_layer,
            )
            logger.info(
                "Preparing {} original prompts, {} steered prompts, and fit rows for {} "
                "concepts (target_n_layer={}, intervention_layer={}, lora_layers={})",
                len(prompt_plan.prompt_by_tokens),
                len(steered_plan.prompt_by_tokens),
                max_concepts,
                target_n_layer,
                intervention_layer,
                lora_layers,
            )
            extraction_model, sae = load_extraction_stack(setup, device)
            prepared_by_label = {
                label: _prepare_all(setup, extraction_model, sae, nanochat_tokenizer, plan)
                for label, plan in (("original", prompt_plan), ("steered", steered_plan))
            }
            fit_rows = _prepare_fit_rows(
                setup,
                extraction_model,
                sae,
                nanochat_tokenizer,
                training,
                sequence_len=sequence_len,
            )
    finally:
        extraction_model = None
        sae = None
        gc.collect()
        if device.startswith("cuda") and t.cuda.is_available():
            t.cuda.empty_cache()

    with rank_synchronized_stage(
        "encoder-decoder target-model validation",
        device=stage_device,
    ):
        logger.info("Loading target checkpoint after releasing the extraction stack")
        model, tokenizer = load_target_model(setup, device)
        if int(model.dec_config.sequence_len) != sequence_len:
            raise ValueError(
                "loaded target sequence length differs from target metadata: "
                f"{int(model.dec_config.sequence_len)} != {sequence_len}"
            )
        for label, plan in (("original", prompt_plan), ("steered", steered_plan)):
            _validate_all(
                setup,
                model,
                tokenizer,
                plan,
                prepared_by_label[label],
                max_new_tokens=max_new_tokens,
            )
        prepared_by_tokens = prepared_by_label["original"]
        steered_prepared_by_tokens = prepared_by_label["steered"]

    with rank_synchronized_stage("encoder-decoder concept fits", device=stage_device):
        train_dir.mkdir(parents=True, exist_ok=True)
        owned_concept_ids = tuple(
            concept_id for concept_id in range(max_concepts) if concept_id % world_size == rank
        )
        with profile_region(
            "axbench_upstream.encdec.concept_fits",
            owned_concepts=len(owned_concept_ids),
        ):
            for concept_id in owned_concept_ids:
                concept_rows = fit_rows[concept_id]
                diffmean_path = _artifact_path(train_dir, concept_id, "diffmean")
                probe_path = _artifact_path(train_dir, concept_id, "linear_probe")
                if not diffmean_path.is_file() or not probe_path.is_file():
                    positive_activations = [
                        capture_post_prefix_activations(
                            model,
                            row.conditioning,
                            layer=intervention_layer,
                            prefix_length=prefix_length,
                        )
                        for row in concept_rows["positive"]
                    ]
                    negative_activations = [
                        capture_post_prefix_activations(
                            model,
                            row.conditioning,
                            layer=intervention_layer,
                            prefix_length=prefix_length,
                        )
                        for row in concept_rows["negative"]
                    ]
                    if not diffmean_path.is_file():
                        diffmean = fit_diffmean_direction(
                            positive_activations,
                            negative_activations,
                            intervention_layer=intervention_layer,
                            surface=f"concept {concept_id} EncDecDiffMean",
                        )
                        save_direction(
                            diffmean_path,
                            diffmean,
                        )
                    if not probe_path.is_file():
                        probe = fit_probe_direction(
                            positive_activations,
                            negative_activations,
                            intervention_layer=intervention_layer,
                            seed=seed,
                            concept_id=concept_id,
                            surface=f"concept {concept_id} EncDecLinearProbe",
                        )
                        save_direction(
                            probe_path,
                            probe,
                        )
                lora_path = _artifact_path(train_dir, concept_id, "lora")
                metadata = _lora_training_metadata(
                    layers=lora_layers,
                    base_seed=seed,
                    concept_id=concept_id,
                    n_rows=len(concept_rows["positive"]),
                )
                if not lora_path.is_file():
                    positive_rows = concept_rows["positive"]
                    lora_state = train_encdec_lora(
                        model,
                        positive_rows,
                        layers=lora_layers,
                        base_seed=seed,
                        concept_id=concept_id,
                        surface=f"concept {concept_id} EncDecLoRA",
                    )
                    save_lora(
                        lora_path,
                        lora_state,
                        metadata,
                    )

    fitted: dict[int, dict[str, FittedDirection]] = {}
    lora_states: dict[int, dict[str, t.Tensor]] = {}
    lora_metadata: dict[int, dict[str, Any]] = {}
    artifacts: list[tuple[Path, str, int]] = []
    with rank_synchronized_stage("encoder-decoder fit validation", device=stage_device):
        fitted, lora_states, lora_metadata, artifacts = _load_fit_artifacts(
            train_dir=train_dir,
            max_concepts=max_concepts,
            seed=seed,
            intervention_layer=intervention_layer,
            lora_layers=lora_layers,
            fit_rows=fit_rows,
            model=model,
        )
        if rank == 0:
            _install_fit_artifact_receipt(
                resume_root=resume_root,
                semantic_fingerprint=store.semantic_fingerprint,
                artifacts=artifacts,
                train_dir=train_dir,
            )

    # Keep this boundary unconditional so generation starts from the same
    # process-local compiler-state contract whether this rank fitted, resumed,
    # or owned no concepts.  Synchronize failures so no peer enters generation
    # after another rank failed to reset its compiler state.
    with rank_synchronized_stage(
        "encoder-decoder compiler phase reset",
        device=stage_device,
    ):
        _reset_compiler_state_before_generation()

    rank_cells = owned_encdec_cells(cells, rank=rank, world_size=world_size)
    owned_first_cells = tuple(cell for cell in rank_cells if cell.concept_id == first_concept_id)
    owned_bulk_cells = tuple(cell for cell in rank_cells if cell.concept_id != first_concept_id)

    with rank_synchronized_stage(
        "encoder-decoder first-concept generation",
        device=stage_device,
    ):
        _append_missing_encdec_cells(
            store,
            owned_first_cells,
            rows=rows,
            prompt_plan=prompt_plan,
            steered_plan=steered_plan,
            prepared_by_tokens=prepared_by_tokens,
            steered_prepared_by_tokens=steered_prepared_by_tokens,
            fitted=fitted,
            lora_states=lora_states,
            intervention_layer=intervention_layer,
            lora_layers=lora_layers,
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        store.flush()

    with rank_synchronized_stage("encoder-decoder first-concept tripwire", device=stage_device):
        if rank == 0:
            with CellResultStore(
                root=resume_root,
                cells=cells,
                tripwire_cells=first_cells,
                semantic_identity=semantic_identity,
                rank=rank,
                world_size=world_size,
            ) as tripwire_store:
                first_results = tripwire_store.results_for(first_cells)
                _assert_first_concept_not_collapsed(
                    {
                        method: [
                            first_results[cell] for cell in first_cells if cell.method == method
                        ]
                        for method in (*ENCDEC_MAIN_METHODS, *ENCDEC_AUX_METHODS)
                    },
                    first_concept_id,
                    label="encoder-decoder baseline method",
                )
                tripwire_store.mark_tripwire_passed()
    with rank_synchronized_stage("encoder-decoder tripwire receipt", device=stage_device):
        store.require_tripwire_passed()

    with rank_synchronized_stage("encoder-decoder bulk generation", device=stage_device):
        _append_missing_encdec_cells(
            store,
            owned_bulk_cells,
            rows=rows,
            prompt_plan=prompt_plan,
            steered_plan=steered_plan,
            prepared_by_tokens=prepared_by_tokens,
            steered_prepared_by_tokens=steered_prepared_by_tokens,
            fitted=fitted,
            lora_states=lora_states,
            intervention_layer=intervention_layer,
            lora_layers=lora_layers,
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        store.flush()
    store.close()

    with rank_synchronized_stage("encoder-decoder result reconstruction", device=stage_device):
        with CellResultStore(
            root=resume_root,
            cells=cells,
            tripwire_cells=first_cells,
            semantic_identity=semantic_identity,
            rank=rank,
            world_size=world_size,
        ) as completed_store:
            cell_generations = completed_store.complete_results()

    generations: dict[str, dict[int, str]] = {
        method: {} for method in (*ENCDEC_MAIN_METHODS, *ENCDEC_AUX_METHODS)
    }
    for cell, generation in cell_generations.items():
        generations[cell.method][cell.source_row_position] = generation

    for method, fit_key in _MAIN_METHOD_FIT_KEYS:
        encdec_main[f"{method}_steered_generation"] = pd.Series(
            [generations[method][position] for position in range(len(rows))],
            index=encdec_main.index,
            dtype="object",
        )
        encdec_main[f"{method}_strength"] = pd.Series(
            [
                float(row.factor) * fitted[int(row.concept_id)][fit_key].max_activation
                for row in rows
            ],
            index=encdec_main.index,
            dtype="float64",
        )
    encdec_main = encdec_main.loc[:, [*INPUT_COLUMNS, *ENCDEC_MAIN_COLUMNS]]

    aux_positions = [position for position, row in enumerate(rows) if float(row.factor) == 1.0]
    encdec_aux = encdec_main.iloc[aux_positions].loc[:, list(INPUT_COLUMNS)].copy()
    encdec_aux["steered_input"] = pd.Series(
        [
            raw_steered_by_pair[(int(rows[position].concept_id), int(rows[position].input_id))]
            for position in aux_positions
        ],
        index=encdec_aux.index,
        dtype="object",
    )
    for method in ENCDEC_AUX_METHODS:
        encdec_aux[f"{method}_steered_generation"] = pd.Series(
            [generations[method][position] for position in aux_positions],
            index=encdec_aux.index,
            dtype="object",
        )
        encdec_aux[f"{method}_strength"] = pd.Series(
            [1.0] * len(aux_positions),
            index=encdec_aux.index,
            dtype="float64",
        )
    encdec_aux = encdec_aux.loc[:, [*INPUT_COLUMNS, *ENCDEC_AUX_COLUMNS]]

    with rank_synchronized_stage("encoder-decoder final validation", device=stage_device):
        if rank == 0:
            _validate_with_rejected_fallback(
                encdec_main,
                source="encdec_main",
                max_concepts=max_concepts,
                rejected_path=rejected_main_path,
            )
            _validate_with_rejected_fallback(
                encdec_aux,
                source="encdec_aux",
                max_concepts=max_concepts,
                rejected_path=rejected_aux_path,
            )
        else:
            validate_steering_frame(
                encdec_main,
                source="encdec_main",
                max_concepts=max_concepts,
            )
            validate_steering_frame(
                encdec_aux,
                source="encdec_aux",
                max_concepts=max_concepts,
            )

    config = {
        "main_path": str(main_path),
        "aux_path": str(aux_path),
        "prepared_dir": str(prepared_dir),
        "output_main_path": str(output_main_path),
        "output_aux_path": str(output_aux_path),
        "manifest_path": str(manifest_path),
        "train_dir": str(train_dir),
        "target_meta_path": str(target_meta_path),
        "feature_statistics_path": str(feature_statistics_path),
        "max_concepts": max_concepts,
        "device": device,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "force": force,
        "pretrain_meta_path": None if pretrain_meta_path is None else str(pretrain_meta_path),
        "require_exact_statistics_identity": require_exact_statistics_identity,
    }
    with rank_synchronized_stage(
        "encoder-decoder canonical publication",
        device=stage_device,
    ):
        if rank == 0:
            manifest = _manifest(
                setup=setup,
                prepared_manifest=prepared_manifest,
                config=config,
                main_path=main_path,
                aux_path=aux_path,
                feature_statistics_path=feature_statistics_path,
                prompt_wrapper=aux_prompt_wrapper,
                target_n_layer=target_n_layer,
                intervention_layer=intervention_layer,
                lora_layers=lora_layers,
                fitted=fitted,
                lora_metadata=lora_metadata,
                steering_prompts=steering_prompts,
                train_dir=train_dir,
                artifacts=artifacts,
            )
            manifest_payload = (
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            if force and manifest_path.exists():
                manifest_path.unlink()
            atomic_write_parquet(
                pa.Table.from_pandas(encdec_main, preserve_index=False),
                output_main_path,
            )
            atomic_write_parquet(
                pa.Table.from_pandas(encdec_aux, preserve_index=False),
                output_aux_path,
            )
            atomic_write_text(manifest_payload, manifest_path)
            logger.info(
                "Wrote {} encoder-decoder main rows and {} auxiliary rows",
                len(encdec_main),
                len(encdec_aux),
            )
    return encdec_main, encdec_aux


def run_encdec_baselines_from_dictconfig(
    cfg: DictConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the thin Hydra config and call :func:`run_encdec_baselines`."""

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError("AxBench encoder-decoder baseline config must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("AxBench encoder-decoder baseline config keys must be strings")
    config = cast(Mapping[str, Any], raw)
    missing = sorted(_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(
            "AxBench encoder-decoder baseline config keys do not match; "
            f"missing={missing}, unknown={unknown}"
        )
    return run_encdec_baselines(
        main_path=Path(str(config["main_path"])),
        aux_path=Path(str(config["aux_path"])),
        prepared_dir=Path(str(config["prepared_dir"])),
        output_main_path=Path(str(config["output_main_path"])),
        output_aux_path=Path(str(config["output_aux_path"])),
        manifest_path=Path(str(config["manifest_path"])),
        train_dir=Path(str(config["train_dir"])),
        target_meta_path=Path(str(config["target_meta_path"])),
        feature_statistics_path=Path(str(config["feature_statistics_path"])),
        max_concepts=config["max_concepts"],
        device=config["device"],
        seed=config["seed"],
        max_new_tokens=config["max_new_tokens"],
        force=config["force"],
        pretrain_meta_path=(
            None
            if config["pretrain_meta_path"] is None
            else Path(str(config["pretrain_meta_path"]))
        ),
        require_exact_statistics_identity=config["require_exact_statistics_identity"],
    )


__all__ = (
    "CAPTURE_PREFIX_LENGTH",
    "ENCDEC_AUX_COLUMNS",
    "ENCDEC_AUX_METHODS",
    "ENCDEC_MAIN_COLUMNS",
    "ENCDEC_MAIN_METHODS",
    "METHOD_ORDER",
    "run_encdec_baselines",
    "run_encdec_baselines_from_dictconfig",
)
