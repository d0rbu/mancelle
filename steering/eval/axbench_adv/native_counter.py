"""Native decoder DiffMean/LinearProbe fits and additive counter generation.

The zero control is the authenticated native PromptSteering producer. Nonzero
points preserve that producer's exact user prompts and sampling seeds while
subtracting an own-model, own-checkpoint residual direction at one explicit
decoder block output.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import pyarrow as pa
import torch as t
import torch.nn as nn
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_bytes, atomic_write_parquet
from scripts.eval.axbench_adv.pressures import (
    magnitude_label,
    validate_magnitudes,
)
from steering.eval.axbench_adv.direction_fitting import (
    SIGNED_CALIBRATION_KEY,
    fit_direction_pair,
    require_counter_calibration,
    require_fit_direction_normalization,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.native_decoder import validate_native_source
from steering.eval.axbench_adv.native_mailbox import digest_json, read_json, write_json
from steering.eval.axbench_base.baselines.hooks import register_residual_hook
from steering.eval.axbench_upstream.data import ConceptTrainingPairs, select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines.directions import FittedDirection
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    render_fit_row,
    template_prefix_length,
)
from steering.eval.axbench_upstream.native_baselines import (
    capture_native_post_prefix_activations,
    resolve_native_model_contract,
)
from steering.eval.axbench_upstream.native_contract import METHOD as NATIVE_METHOD
from steering.eval.axbench_upstream.native_contract import load_native_output
from steering.eval.axbench_upstream.native_decoder import (
    NativeDecoderAdapter,
    canonical_bytes,
    cell_seed,
    checkpoint_identity,
)
from steering.immutable_io import install_bytes_once, stream_hash_file_descriptor_bound
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    load_safetensors_file_snapshot,
)

FORMAT = "axbench-native-decoder-counter-v1"
CAPTURE_SITE = "decoder_block_output"
CAPTURE_PREFIX_LENGTH = 2
COUNTER_METHODS = ("DiffMean", "LinearProbe")
COUNTER_MAGNITUDES = (0.25, 0.5, 1.0, 2.0)
_FIT_KIND = {"DiffMean": "diffmean", "LinearProbe": "linear_probe"}
_DIRECTION_FORMAT = "axbench-encdec-direction-v1"


def file_sha256(path: Path) -> str:
    """Hash one stable file without changing the authenticated source runtime."""

    return stream_hash_file_descriptor_bound(path).sha256


class NativeCounterAdapter(Protocol):
    model: Any
    tokenizer: Any

    def generate_user_text(
        self,
        user_text: str,
        *,
        seed: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str: ...


def _strict_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AxBenchAdvError(f"{label} must be a string-keyed mapping")
    return dict(value)


def _metadata_geometry(meta_path: Path) -> dict[str, int]:
    try:
        document = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AxBenchAdvError(f"invalid native checkpoint metadata: {meta_path}") from exc
    model_config = document.get("model_config") if isinstance(document, dict) else None
    if not isinstance(model_config, dict):
        raise AxBenchAdvError("native counter checkpoint metadata has no model_config")
    geometry: dict[str, int] = {}
    for key in ("n_layer", "n_embd", "sequence_len"):
        value = model_config.get(key)
        if type(value) is not int or value <= 0:
            raise AxBenchAdvError(f"native counter checkpoint model_config.{key} is invalid")
        geometry[key] = value
    return geometry


def validate_runtime_identity(
    *,
    cfg: DictConfig,
    source_manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    """Bind relocated local files to the expected and source checkpoint hashes."""

    if type(cfg.step) is not int:
        raise AxBenchAdvError("native counter step must be an integer")
    runtime = checkpoint_identity(
        Path(str(cfg.checkpoint_dir)),
        int(cfg.step),
        Path(str(cfg.tokenizer_path)),
    )
    geometry = _metadata_geometry(Path(str(cfg.checkpoint_dir)) / f"meta_{int(cfg.step):06d}.json")
    raw_expected = OmegaConf.to_container(cfg.expected, resolve=True)
    expected = _strict_mapping(raw_expected, label="expected native identity")
    if set(expected) != {
        "step", "model_sha256", "meta_sha256", "tokenizer_sha256", "n_layer", "n_embd"
    } or any(type(expected[key]) is not int for key in ("step", "n_layer", "n_embd")):
        raise AxBenchAdvError("expected native identity has invalid fields or types")
    observed = {
        "step": runtime["step"],
        "model_sha256": runtime["model_sha256"],
        "meta_sha256": runtime["meta_sha256"],
        "tokenizer_sha256": runtime["tokenizer_sha256"],
        "n_layer": geometry["n_layer"],
        "n_embd": geometry["n_embd"],
    }
    if observed != expected:
        raise AxBenchAdvError(f"native runtime checkpoint/tokenizer identity differs: {observed!r}")
    model_contract = resolve_native_model_contract(geometry)
    if type(cfg.layer) is not int or cfg.layer != model_contract.intervention_layer:
        raise AxBenchAdvError("native counter layer differs from the reviewed geometry mapping")
    source_checkpoint = source_manifest.get("checkpoint")
    if not isinstance(source_checkpoint, dict) or any(
        source_checkpoint.get(key) != expected[key]
        for key in ("step", "model_sha256", "meta_sha256", "tokenizer_sha256")
    ):
        raise AxBenchAdvError("native standard source used a different checkpoint/tokenizer")
    return runtime, geometry


def _validate_loaded_model(model: nn.Module, geometry: Mapping[str, int]) -> None:
    """Validate loaded geometry and the exact count for contracts that pin it."""

    model_contract = resolve_native_model_contract(geometry)
    model_config = cast(Any, model).config
    if (
        int(model_config.n_layer) != model_contract.n_layer
        or int(model_config.n_embd) != model_contract.n_embd
        or int(model_config.sequence_len) != model_contract.sequence_len
    ):
        raise AxBenchAdvError("loaded native model geometry differs from metadata")
    if (
        model_contract.parameter_count is not None
        and sum(parameter.numel() for parameter in model.parameters())
        != model_contract.parameter_count
    ):
        raise AxBenchAdvError("loaded native model parameter count differs")


def _concept_scope(
    *,
    cfg: DictConfig,
    source_manifest: Mapping[str, Any],
    source_frame: pd.DataFrame,
) -> tuple[tuple[int, ...], pd.DataFrame, dict[str, Any] | None]:
    """Select the sole supported derivative scope: canonical 0..49 of standard100."""

    source_max_concepts = int(source_manifest["source"]["max_concepts"])
    if source_max_concepts != int(cfg.max_concepts):
        raise AxBenchAdvError("native counter concept count differs from standard source")
    concept_count = cfg.concept_count
    if concept_count is None:
        concept_ids = tuple(range(source_max_concepts))
        return concept_ids, source_frame, None
    if type(concept_count) is not int or concept_count != 50 or source_max_concepts != 100:
        raise AxBenchAdvError(
            "concept_count must be null or exactly 50 for a canonical standard100 prefix"
        )
    concept_ids = tuple(range(concept_count))
    selected = source_frame.loc[source_frame.concept_id.isin(concept_ids)].reset_index(drop=True)
    if len(selected) != concept_count * 10:
        raise AxBenchAdvError("native counter canonical prefix coverage differs")
    scope = {
        "kind": "canonical_prefix",
        "concept_count": concept_count,
        "concept_ids": list(concept_ids),
        "authenticated_source_max_concepts": source_max_concepts,
    }
    return concept_ids, selected, scope


@contextmanager
def scoped_native_counter(
    model: nn.Module,
    *,
    layer: int,
    direction: t.Tensor,
    magnitude: float,
) -> Iterator[None]:
    """Subtract one saved fp32 unit direction on every prefill/decode forward."""

    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise AxBenchAdvError("native counter magnitude must be numeric")
    parsed_magnitude = float(magnitude)
    if not math.isfinite(parsed_magnitude) or parsed_magnitude < 0:
        raise AxBenchAdvError("native counter magnitude must be finite and nonnegative")
    model_config = cast(Any, model).config
    if type(layer) is not int or not 0 <= layer < int(model_config.n_layer):
        raise AxBenchAdvError("native counter layer is outside the decoder")
    if (
        direction.dtype is not t.float32
        or direction.device.type != "cpu"
        or direction.ndim != 1
        or direction.numel() != int(model_config.n_embd)
        or not direction.is_contiguous()
        or not bool(t.isfinite(direction).all())
    ):
        raise AxBenchAdvError(
            "native counter direction must be contiguous CPU float32 with decoder width"
        )
    if parsed_magnitude == 0.0:
        yield
        return
    handle = register_residual_hook(model, layer, direction, -parsed_magnitude)
    try:
        yield
    finally:
        handle.remove()


def _direction_path(output: Path, concept_id: int, method: str) -> Path:
    return output / "fits" / f"c{concept_id:05d}" / f"{_FIT_KIND[method]}.safetensors"


def _magnitude_label(value: float) -> str:
    labels = {0.25: "m0p25", 0.5: "m0p5", 1.0: "m1", 2.0: "m2"}
    return labels[value] if value in labels else magnitude_label(value)


def _point_selection(
    cfg: DictConfig,
    configured_magnitudes: Sequence[float] = COUNTER_MAGNITUDES,
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Resolve placement-only point filters without changing the full grid identity."""

    point_methods = COUNTER_METHODS if cfg.point_methods is None else tuple(cfg.point_methods)
    if (
        not point_methods
        or len(set(point_methods)) != len(point_methods)
        or any(method not in COUNTER_METHODS for method in point_methods)
    ):
        raise AxBenchAdvError(
            f"point_methods must be null or a unique subset of {list(COUNTER_METHODS)}"
        )
    point_magnitudes = (
        tuple(configured_magnitudes)
        if cfg.point_magnitudes is None
        else cfg.point_magnitudes
    )
    try:
        point_magnitudes = validate_magnitudes(point_magnitudes)
    except (TypeError, ValueError) as exc:
        raise AxBenchAdvError(
            "point_magnitudes must be null or a strictly positive finite ascending "
            f"unique subset of {list(configured_magnitudes)}"
        ) from exc
    if any(magnitude not in configured_magnitudes for magnitude in point_magnitudes):
        raise AxBenchAdvError(
            f"point_magnitudes must be null or a subset of {list(configured_magnitudes)}"
        )
    return point_methods, point_magnitudes


def _fit_contract(
    *,
    identity_sha256: str,
    concept_id: int,
    examples: ConceptTrainingPairs,
    layer: int,
    geometry: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "identity_sha256": identity_sha256,
        "concept_id": concept_id,
        "concept": examples.concept,
        "genre": examples.genre,
        "capture_site": CAPTURE_SITE,
        "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
        "intervention_layer": layer,
        "n_layer": geometry["n_layer"],
        "n_embd": geometry["n_embd"],
        "n_positive_rows": len(examples.positive),
        "n_negative_rows": len(examples.negative),
        "rendering": "trained RustBPE full user/assistant conversation",
        "population": "all rendered tokens at or after the shared template prefix",
    }


def _validate_direction(
    fitted: FittedDirection,
    *,
    method: str,
    contract: Mapping[str, Any],
    seed: int,
) -> None:
    direction = fitted.direction
    if (
        direction.dtype is not t.float32
        or direction.device.type != "cpu"
        or direction.ndim != 1
        or direction.numel() != contract["n_embd"]
        or not direction.is_contiguous()
        or not bool(t.isfinite(direction).all())
    ):
        raise AxBenchAdvError("native fitted direction geometry/dtype is invalid")
    family = _FIT_KIND[method]
    require_fit_direction_normalization(
        direction,
        family=family,
        surface=f"native {method} fit",
        error_cls=AxBenchAdvError,
    )
    calibration_key = (
        SIGNED_CALIBRATION_KEY
        if fitted.max_activation <= 0.0 and bool((direction != 0.0).any().item())
        else "max_activation"
    )
    require_counter_calibration(
        {calibration_key: repr(float(fitted.max_activation))},
        direction=direction,
        surface=f"native {method} fit",
        error_cls=AxBenchAdvError,
    )
    metadata = fitted.metadata
    expected_method = "diffmean" if method == "DiffMean" else "probe"
    required = {
        "method": expected_method,
        "intervention_layer": contract["intervention_layer"],
        "n_positive_rows": contract["n_positive_rows"],
        "n_negative_rows": contract["n_negative_rows"],
        "native_contract": dict(contract),
    }
    if any(metadata.get(key) != value for key, value in required.items()):
        raise AxBenchAdvError(f"native {method} fit provenance differs")
    if method == "LinearProbe" and (
        metadata.get("seed") != seed
        or metadata.get("concept_id") != contract["concept_id"]
        or not isinstance(metadata.get("estimator"), dict)
    ):
        raise AxBenchAdvError("native LinearProbe estimator provenance differs")


def _save_native_direction(path: Path, fitted: FittedDirection) -> None:
    """Write a raw-counter fit without exposing signed calibration as a scale."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    direction = fitted.direction.detach().to(dtype=t.float32, device="cpu").contiguous()
    calibration_key = (
        SIGNED_CALIBRATION_KEY
        if fitted.max_activation <= 0.0 and bool((direction != 0.0).any().item())
        else "max_activation"
    )
    metadata = {
        "format": _DIRECTION_FORMAT,
        calibration_key: repr(float(fitted.max_activation)),
        "metadata_json": json.dumps(
            fitted.metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    require_counter_calibration(
        metadata,
        direction=direction,
        surface=f"native direction artifact {path}",
    )
    atomic_write_bytes(
        deterministic_safetensors_bytes({"direction": direction}, metadata=metadata),
        path,
    )


def _load_native_direction(path: Path) -> FittedDirection:
    """Load one native raw-counter fit, including signed calibration metadata."""

    path = Path(path)
    tensors, loaded_metadata = load_safetensors_file_snapshot(path)
    metadata = loaded_metadata or {}
    if metadata.get("format") != _DIRECTION_FORMAT:
        raise ValueError(f"{path}: invalid or missing direction artifact format")
    calibration_keys = {"max_activation", SIGNED_CALIBRATION_KEY} & set(metadata)
    if len(calibration_keys) != 1:
        raise ValueError(f"{path}: direction artifact must contain exactly one calibration field")
    expected_metadata_keys = {
        "format",
        "metadata_json",
        next(iter(calibration_keys)),
    }
    unexpected_metadata = sorted(set(metadata) - expected_metadata_keys)
    if unexpected_metadata:
        raise ValueError(f"{path}: unexpected direction metadata keys: {unexpected_metadata}")
    if "metadata_json" not in metadata:
        raise ValueError(f"{path}: missing metadata_json metadata")
    if set(tensors) != {"direction"}:
        raise ValueError(f"{path}: direction artifact must contain only 'direction'")
    direction = tensors["direction"]
    if (
        direction.dtype is not t.float32
        or direction.device.type != "cpu"
        or direction.ndim != 1
        or direction.numel() == 0
        or not direction.is_contiguous()
        or not bool(t.isfinite(direction).all().item())
    ):
        raise ValueError(
            f"{path}: direction must be a finite, nonempty, contiguous CPU float32 vector"
        )
    calibration = require_counter_calibration(
        metadata,
        direction=direction,
        surface=f"native direction artifact {path}",
    )

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        fit_metadata = json.loads(
            metadata["metadata_json"],
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: metadata_json is not strict JSON: {exc}") from exc
    if not isinstance(fit_metadata, dict):
        raise ValueError(f"{path}: metadata_json must contain an object")
    return FittedDirection(
        direction=direction,
        max_activation=calibration,
        metadata=fit_metadata,
    )


def _load_validated_direction(
    path: Path,
    *,
    method: str,
    contract: Mapping[str, Any],
    seed: int,
) -> FittedDirection:
    try:
        fitted = _load_native_direction(path)
    except (OSError, ValueError) as exc:
        raise AxBenchAdvError(f"invalid native direction artifact: {path}") from exc
    _validate_direction(fitted, method=method, contract=contract, seed=seed)
    return fitted


def _capture_class(
    adapter: NativeCounterAdapter,
    pairs: Sequence[tuple[str, str]],
    *,
    concept_id: int,
    class_name: str,
    layer: int,
) -> list[t.Tensor]:
    rows: list[t.Tensor] = []
    for row_index, (input_text, output_text) in enumerate(pairs):
        surface = f"native concept {concept_id} {class_name} row {row_index}"
        rendered = render_fit_row(
            adapter.tokenizer,
            input_text=input_text,
            output_text=output_text,
            sequence_len=int(adapter.model.config.sequence_len),
            surface=surface,
        )
        rows.append(
            capture_native_post_prefix_activations(
                adapter.model,
                rendered,
                layer=layer,
                prefix_length=CAPTURE_PREFIX_LENGTH,
            )
        )
    return rows


def _fit_all(
    *,
    get_adapter: Callable[[], NativeCounterAdapter],
    training: Mapping[int, ConceptTrainingPairs],
    source_frame: pd.DataFrame,
    output: Path,
    identity: Mapping[str, Any],
    geometry: Mapping[str, int],
    layer: int,
    seed: int,
) -> dict[int, dict[str, FittedDirection]]:
    identity_sha = digest_json(identity)
    fitted: dict[int, dict[str, FittedDirection]] = {}
    source_concepts = source_frame.groupby("concept_id", sort=False)["input_concept"].first()
    for concept_id in range(len(training)):
        examples = training[concept_id]
        if source_concepts.get(concept_id) != examples.concept:
            raise AxBenchAdvError(f"native fit/source concept differs for concept_id={concept_id}")
        contract = _fit_contract(
            identity_sha256=identity_sha,
            concept_id=concept_id,
            examples=examples,
            layer=layer,
            geometry=geometry,
        )
        paths = {method: _direction_path(output, concept_id, method) for method in COUNTER_METHODS}
        missing = [method for method, path in paths.items() if not path.is_file()]
        if missing:
            adapter = get_adapter()
            positive = _capture_class(
                adapter,
                examples.positive,
                concept_id=concept_id,
                class_name="positive",
                layer=layer,
            )
            negative = _capture_class(
                adapter,
                examples.negative,
                concept_id=concept_id,
                class_name="negative",
                layer=layer,
            )
            raw_fits = fit_direction_pair(
                positive,
                negative,
                layer=layer,
                seed=seed,
                concept_id=concept_id,
                surface=f"native concept {concept_id}",
            )
            for method in missing:
                raw_fit = raw_fits[_FIT_KIND[method]]
                enriched = replace(
                    raw_fit,
                    metadata={**raw_fit.metadata, "native_contract": contract},
                )
                _validate_direction(enriched, method=method, contract=contract, seed=seed)
                _save_native_direction(paths[method], enriched)
        fitted[concept_id] = {
            method: _load_validated_direction(
                path,
                method=method,
                contract=contract,
                seed=seed,
            )
            for method, path in paths.items()
        }
        receipt = {
            "format": FORMAT,
            "identity_sha256": identity_sha,
            "concept_id": concept_id,
            "artifacts": {
                method: {"path": str(path), "sha256": file_sha256(path)}
                for method, path in paths.items()
            },
        }
        install_bytes_once(
            canonical_bytes(receipt),
            output / "fits" / f"c{concept_id:05d}" / "receipt.json",
            surface="native counter fit receipt",
        )
        logger.info("Native counter fit complete concept={}", concept_id)
    return fitted


def _fit_manifest(
    *,
    output: Path,
    identity: Mapping[str, Any],
    training: Mapping[int, ConceptTrainingPairs],
) -> dict[str, Any]:
    inventory = []
    expected_paths: set[Path] = set()
    for concept_id in range(len(training)):
        for method in COUNTER_METHODS:
            path = _direction_path(output, concept_id, method)
            expected_paths.add(path)
            if not path.is_file():
                raise AxBenchAdvError(f"native fit artifact is absent: {path}")
            inventory.append(
                {
                    "concept_id": concept_id,
                    "method": method,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
    fit_root = output / "fits"
    observed_paths = set(fit_root.glob("c*/*.safetensors")) if fit_root.exists() else set()
    if observed_paths != expected_paths:
        raise AxBenchAdvError(
            "native fit artifact coverage differs; "
            f"missing={sorted(map(str, expected_paths - observed_paths))[:5]}, "
            f"extra={sorted(map(str, observed_paths - expected_paths))[:5]}"
        )
    manifest = {
        "format": FORMAT,
        "complete": True,
        "identity_sha256": digest_json(identity),
        "concept_ids": list(range(len(training))),
        "methods": list(COUNTER_METHODS),
        "artifacts": inventory,
    }
    install_bytes_once(
        canonical_bytes(manifest),
        output / "fit_manifest.json",
        surface="native counter fit manifest",
    )
    return manifest


def _validate_existing_cells(
    cells_dir: Path,
    *,
    expected_names: set[str],
) -> None:
    if not cells_dir.exists():
        return
    # An interrupted immutable install can leave an uncommitted scratch file.
    observed = {path.name for path in cells_dir.glob("*.json") if path.is_file()}
    extra = observed - expected_names
    if extra:
        raise AxBenchAdvError(f"native counter point contains unexpected cells: {sorted(extra)}")


def _generate_point(
    *,
    get_adapter: Callable[[], NativeCounterAdapter],
    revalidate_inputs: Callable[[], None],
    source_manifest: Mapping[str, Any],
    source_frame: pd.DataFrame,
    output: Path,
    identity: Mapping[str, Any],
    runtime_checkpoint: Mapping[str, Any],
    fitted: Mapping[int, Mapping[str, FittedDirection]],
    method: str,
    magnitude: float,
    layer: int,
    max_new_cells: int | None,
    concept_ids: Sequence[int] | None = None,
) -> tuple[bool, int]:
    label = _magnitude_label(magnitude)
    point = output / "points" / _FIT_KIND[method] / label
    cells_dir = point / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    effective_concept_ids = (
        tuple(range(int(source_manifest["source"]["max_concepts"])))
        if concept_ids is None
        else tuple(concept_ids)
    )
    expected_names = {
        f"c{concept_id:05d}_i{input_id:05d}.json"
        for concept_id in effective_concept_ids
        for input_id in range(10)
    }
    _validate_existing_cells(cells_dir, expected_names=expected_names)
    point_identity = {
        "run_identity_sha256": digest_json(identity),
        "method": method,
        "magnitude_hex": float(magnitude).hex(),
        "operation": "residual -= magnitude * saved fp32 normalized direction",
    }
    point_sha = digest_json(point_identity)
    generation_cfg = dict(source_manifest["generation"])
    master_seed = int(generation_cfg.pop("seed"))
    results: list[str] = []
    generated = 0
    started = time.monotonic()
    for row in source_frame.to_dict(orient="records"):
        concept_id = int(row["concept_id"])
        input_id = int(row["input_id"])
        seed = cell_seed(master_seed, concept_id, input_id)
        direction_path = _direction_path(output, concept_id, method)
        key = {
            "concept_id": concept_id,
            "input_id": input_id,
            "seed": seed,
            "method": method,
            "magnitude_hex": float(magnitude).hex(),
            "direction_sha256": file_sha256(direction_path),
        }
        path = cells_dir / f"c{concept_id:05d}_i{input_id:05d}.json"
        if path.exists():
            record = read_json(path)
            if (
                set(record) != {"point_identity_sha256", "cell", "generation"}
                or record.get("point_identity_sha256") != point_sha
                or record.get("cell") != key
                or not isinstance(record.get("generation"), str)
            ):
                raise AxBenchAdvError(f"invalid/conflicting native counter cell: {path}")
            results.append(record["generation"])
            continue
        if max_new_cells is not None and generated >= max_new_cells:
            write_json(
                point / "progress.json",
                {
                    "complete": False,
                    "completed_cells": len(results),
                    "total_cells": len(source_frame),
                    "new_cells": generated,
                },
            )
            return False, generated
        adapter = get_adapter()
        direction = fitted[concept_id][method].direction
        with scoped_native_counter(
            adapter.model,
            layer=layer,
            direction=direction,
            magnitude=magnitude,
        ):
            generation = adapter.generate_user_text(
                str(row["native_user_prompt"]),
                seed=seed,
                **generation_cfg,
            )
        record = {
            "point_identity_sha256": point_sha,
            "cell": key,
            "generation": generation,
        }
        install_bytes_once(
            canonical_bytes(record),
            path,
            surface="native counter generation cell",
        )
        results.append(generation)
        generated += 1
        write_json(
            point / "progress.json",
            {
                "complete": False,
                "completed_cells": len(results),
                "total_cells": len(source_frame),
                "new_cells": generated,
                "elapsed_seconds": time.monotonic() - started,
            },
        )
    _validate_existing_cells(cells_dir, expected_names=expected_names)
    if len(results) != len(source_frame):
        raise AxBenchAdvError("native counter point coverage is incomplete")
    frame = source_frame.copy()
    frame[f"{NATIVE_METHOD}_steered_generation"] = results
    output_path = point / "steering_data.parquet"
    atomic_write_parquet(pa.Table.from_pandas(frame, preserve_index=False), output_path)
    direction_inventory = {
        str(concept_id): file_sha256(_direction_path(output, concept_id, method))
        for concept_id in effective_concept_ids
    }
    point_source = dict(source_manifest["source"])
    point_source["max_concepts"] = len(effective_concept_ids)
    counter_extension = {
        "format": FORMAT,
        "identity_sha256": digest_json(identity),
        "point_identity": point_identity,
        "capture_site": CAPTURE_SITE,
        "intervention_site": CAPTURE_SITE,
        "layer": layer,
        "directions": direction_inventory,
        "zero_control": {
            "source_manifest": identity["inputs"]["source_manifest"],
            "source_parquet": identity["inputs"]["source_parquet"],
            "generation_reused": True,
        },
    }
    identity_configuration = identity.get("configuration", {})
    concept_scope = (
        identity_configuration.get("concept_scope")
        if isinstance(identity_configuration, Mapping)
        else None
    )
    if concept_scope is not None:
        counter_extension["source_scope"] = {
            **concept_scope,
            "authenticated_source_manifest": identity["inputs"]["source_manifest"],
            "authenticated_source_parquet": identity["inputs"]["source_parquet"],
        }
    manifest = {
        **dict(source_manifest),
        "source": point_source,
        "checkpoint": dict(runtime_checkpoint),
        "complete": True,
        "completed_cells": len(frame),
        "total_cells": len(frame),
        "output": {
            "path": str(output_path.resolve()),
            "sha256": file_sha256(output_path),
            "rows": len(frame),
        },
        "counter_extension": counter_extension,
    }
    manifest_path = point / "run_manifest.json"
    revalidate_inputs()
    install_bytes_once(
        canonical_bytes(manifest),
        manifest_path,
        surface="native counter point manifest",
    )
    loaded_manifest, loaded_frame = load_native_output(manifest_path)
    if loaded_manifest.get("counter_extension") != manifest["counter_extension"]:
        raise AxBenchAdvError("native counter point extension changed after publication")
    if loaded_frame[f"{NATIVE_METHOD}_steered_generation"].tolist() != results:
        raise AxBenchAdvError("native counter point generations changed after publication")
    write_json(
        point / "progress.json",
        {
            "complete": True,
            "completed_cells": len(results),
            "total_cells": len(results),
            "new_cells": generated,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    logger.info("Native counter point complete method={} magnitude={}", method, magnitude)
    return True, generated


def run_native_counter(cfg: DictConfig) -> dict[str, Any]:
    """Fit own-model directions and generate the configured durable counter points."""

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, dict):
        raise AxBenchAdvError("native counter config must be a mapping")
    phase = cfg.phase
    if phase not in {"fit", "generate", "all"}:
        raise AxBenchAdvError("phase must be fit, generate, or all")
    if list(cfg.methods) != list(COUNTER_METHODS):
        raise AxBenchAdvError(f"methods must be exactly {list(COUNTER_METHODS)}")
    try:
        magnitudes = validate_magnitudes(cast(Sequence[float], raw.get("magnitudes")))
    except (TypeError, ValueError) as exc:
        raise AxBenchAdvError(str(exc)) from exc
    point_methods, point_magnitudes = _point_selection(cfg, magnitudes)
    if type(cfg.seed) is not int:
        raise AxBenchAdvError("native counter seed must be an integer")
    max_new_cells = cfg.max_cells
    if max_new_cells is not None and (type(max_new_cells) is not int or max_new_cells < 1):
        raise AxBenchAdvError("max_cells must be null or a positive integer")
    source_manifest_path = Path(str(cfg.source_manifest)).resolve()
    source_parquet_path = Path(str(cfg.source_parquet)).resolve()
    prepared_dir = Path(str(cfg.prepared_dir)).resolve()
    pinned_inputs = {
        key: {"path": str(path), "sha256": file_sha256(path)}
        for key, path in (
            ("source_manifest", source_manifest_path),
            ("source_parquet", source_parquet_path),
            ("prepared_metadata", prepared_dir / "generate" / "metadata.jsonl"),
            ("prepared_training_data", prepared_dir / "generate" / "train_data.parquet"),
        )
    }

    def check_pinned_inputs() -> None:
        for key, record in pinned_inputs.items():
            if file_sha256(Path(record["path"])) != record["sha256"]:
                raise AxBenchAdvError(f"native counter input changed during execution: {key}")

    source_manifest, source_frame = validate_native_source(
        source_manifest_path,
        source_parquet_path,
        checkpoint_dir=Path(str(cfg.checkpoint_dir)),
        tokenizer_path=Path(str(cfg.tokenizer_path)),
    )
    source_max_concepts = int(source_manifest["source"]["max_concepts"])
    concept_ids, source_frame, concept_scope = _concept_scope(
        cfg=cfg,
        source_manifest=source_manifest,
        source_frame=source_frame,
    )
    effective_concept_count = len(concept_ids)
    runtime_checkpoint, geometry = validate_runtime_identity(
        cfg=cfg,
        source_manifest=source_manifest,
    )
    output = Path(str(cfg.output_dir)).resolve()
    read_paths = (
        source_manifest_path,
        source_parquet_path,
        prepared_dir,
        Path(str(cfg.checkpoint_dir)).resolve(),
        Path(str(cfg.tokenizer_path)).resolve(),
    )
    if any(
        output == path or output.is_relative_to(path) or path.is_relative_to(output)
        for path in read_paths
    ):
        raise AxBenchAdvError("native counter output must be disjoint from every input")
    training = select_training_pairs(prepared_dir, max_concepts=effective_concept_count)
    check_pinned_inputs()
    root = Path(__file__).resolve().parents[3]
    implementation_files = (
        "steering/eval/axbench_adv/native_counter.py",
        "steering/eval/axbench_adv/native_decoder.py",
        "steering/eval/axbench_upstream/native_decoder.py",
        "steering/eval/axbench_upstream/native_contract.py",
        "steering/eval/axbench_upstream/data.py",
        "steering/eval/axbench_upstream/encdec_baselines/directions.py",
        "steering/eval/axbench_upstream/encdec_baselines/rendering.py",
        "steering/eval/axbench_base/baselines/hooks.py",
        "steering/eval/inference_engine.py",
        "nanochat/gpt.py",
        "nanochat/engine.py",
        "nanochat/tokenizer.py",
        "nanochat/sampling.py",
        "uv.lock",
    )
    configuration = {
        "max_concepts": source_max_concepts,
        "methods": list(COUNTER_METHODS),
        "magnitudes": [float(value).hex() for value in magnitudes],
        "layer": int(cfg.layer),
        "capture_site": CAPTURE_SITE,
        "intervention_site": CAPTURE_SITE,
        "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
        "seed": int(cfg.seed),
        "generation": dict(source_manifest["generation"]),
        "operation": "residual -= magnitude * saved fp32 normalized direction",
        "zero_control": "reuse authenticated standard source emissions; never regenerate",
    }
    if concept_scope is not None:
        configuration["concept_scope"] = concept_scope
    identity = {
        "format": FORMAT,
        "inputs": {**pinned_inputs, "runtime_checkpoint": runtime_checkpoint},
        "configuration": configuration,
        "model_geometry": geometry,
        "implementation_sha256": {name: file_sha256(root / name) for name in implementation_files},
    }
    identity_path = output / "identity.json"
    if output.exists() and not identity_path.is_file() and any(
        not (path.is_file() and path.name.startswith(".identity.json.tmp."))
        for path in output.iterdir()
    ):
        raise AxBenchAdvError("native counter refuses a nonempty output without identity.json")
    output.mkdir(parents=True, exist_ok=True)
    install_bytes_once(
        canonical_bytes(identity),
        identity_path,
        surface="native counter run identity",
    )
    control = {
        "format": FORMAT,
        "identity_sha256": digest_json(identity),
        "magnitude_hex": (0.0).hex(),
        "source_manifest": identity["inputs"]["source_manifest"],
        "source_parquet": identity["inputs"]["source_parquet"],
        "generation_reused": True,
        "judge_cache_reuse_required": True,
    }
    if concept_scope is not None:
        control["source_scope"] = concept_scope
    install_bytes_once(
        canonical_bytes(control),
        output / "control.json",
        surface="native counter zero control",
    )

    adapter: NativeDecoderAdapter | None = None

    def revalidate_inputs() -> None:
        check_pinned_inputs()
        validate_native_source(
            source_manifest_path,
            source_parquet_path,
            checkpoint_dir=Path(str(cfg.checkpoint_dir)),
            tokenizer_path=Path(str(cfg.tokenizer_path)),
        )
        current_runtime, current_geometry = validate_runtime_identity(
            cfg=cfg,
            source_manifest=source_manifest,
        )
        if current_runtime != runtime_checkpoint or current_geometry != geometry:
            raise AxBenchAdvError("native runtime identity changed during execution")

    def get_adapter() -> NativeDecoderAdapter:
        nonlocal adapter
        if adapter is None:
            adapter = NativeDecoderAdapter(
                checkpoint_dir=Path(str(cfg.checkpoint_dir)),
                step=int(cfg.step),
                tokenizer_path=Path(str(cfg.tokenizer_path)),
                device=str(cfg.device),
            )
            loaded = adapter.identity
            if any(
                loaded[key] != runtime_checkpoint[key]
                for key in ("step", "model_sha256", "meta_sha256", "tokenizer_sha256")
            ):
                raise AxBenchAdvError("native runtime changed while loading the model")
            if template_prefix_length(adapter.tokenizer) != CAPTURE_PREFIX_LENGTH:
                raise AxBenchAdvError(
                    f"trained RustBPE template prefix must be {CAPTURE_PREFIX_LENGTH} tokens"
                )
            _validate_loaded_model(adapter.model, geometry)
        return adapter

    fitted: dict[int, dict[str, FittedDirection]] = {}
    if phase in {"fit", "all"}:
        fitted = _fit_all(
            get_adapter=get_adapter,
            training=training,
            source_frame=source_frame,
            output=output,
            identity=identity,
            geometry=geometry,
            layer=int(cfg.layer),
            seed=int(cfg.seed),
        )
        revalidate_inputs()
        _fit_manifest(output=output, identity=identity, training=training)
    if phase == "fit":
        revalidate_inputs()
        status = {
            "phase": "fit_complete",
            "identity_sha256": digest_json(identity),
            "fit_manifest": str(output / "fit_manifest.json"),
        }
        write_json(output / "status.json", status)
        return status
    if not (output / "fit_manifest.json").is_file():
        raise AxBenchAdvError("generate phase requires a complete native fit manifest")
    fit_manifest = _fit_manifest(output=output, identity=identity, training=training)
    if fit_manifest.get("complete") is not True or fit_manifest.get(
        "identity_sha256"
    ) != digest_json(identity):
        raise AxBenchAdvError("native fit manifest identity differs")
    if not fitted:
        for concept_id, examples in training.items():
            contract = _fit_contract(
                identity_sha256=digest_json(identity),
                concept_id=concept_id,
                examples=examples,
                layer=int(cfg.layer),
                geometry=geometry,
            )
            fitted[concept_id] = {
                method: _load_validated_direction(
                    _direction_path(output, concept_id, method),
                    method=method,
                    contract=contract,
                    seed=int(cfg.seed),
                )
                for method in COUNTER_METHODS
            }
    remaining = max_new_cells
    completed_points: list[str] = []
    for method in point_methods:
        for magnitude in point_magnitudes:
            complete, generated = _generate_point(
                get_adapter=get_adapter,
                revalidate_inputs=revalidate_inputs,
                source_manifest=source_manifest,
                source_frame=source_frame,
                output=output,
                identity=identity,
                runtime_checkpoint=runtime_checkpoint,
                fitted=fitted,
                method=method,
                magnitude=magnitude,
                layer=int(cfg.layer),
                max_new_cells=remaining,
                concept_ids=concept_ids,
            )
            if complete:
                completed_points.append(f"{_FIT_KIND[method]}/{_magnitude_label(magnitude)}")
            if remaining is not None:
                remaining -= generated
            if not complete:
                revalidate_inputs()
                status = {
                    "phase": "generation_incomplete",
                    "identity_sha256": digest_json(identity),
                    "completed_points": completed_points,
                }
                write_json(output / "status.json", status)
                return status
    revalidate_inputs()
    status = {
        "phase": "generation_selection_complete_pending_native_judges",
        "identity_sha256": digest_json(identity),
        "control": str(output / "control.json"),
        "selected_methods": list(point_methods),
        "selected_magnitudes": [float(value).hex() for value in point_magnitudes],
        "completed_points": completed_points,
    }
    write_json(output / "status.json", status)
    return status


__all__ = [
    "CAPTURE_PREFIX_LENGTH",
    "CAPTURE_SITE",
    "COUNTER_MAGNITUDES",
    "COUNTER_METHODS",
    "capture_native_post_prefix_activations",
    "run_native_counter",
    "scoped_native_counter",
    "validate_runtime_identity",
]
