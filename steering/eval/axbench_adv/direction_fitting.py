"""Real-conditioned AxBench-training direction fitting and publication.

This module intentionally orchestrates existing rendering, conditioning,
capture, and pure fitting primitives. It adds only the campaign-specific
post-cross-attention site and the checkpoint-bound direction publication
adapter required by AxBench-Adv.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
from loguru import logger
from sklearn.exceptions import ConvergenceWarning

from nanochat.atomic_io import atomic_write_bytes, atomic_write_text
from nanochat.tokenizer import get_tokenizer
from steering.eval.axbench_base.direction_artifacts import (
    expected_artifact_kind,
    load_direction_artifact,
)
from steering.eval.axbench_base.direction_scale import (
    encode_max_activation,
    parse_max_activation,
)
from steering.eval.axbench_upstream.data import ConceptTrainingPairs, select_training_pairs
from steering.eval.axbench_upstream.encdec_baselines.capture import (
    FitRowConditioning,
    build_fit_row_conditioning,
    capture_post_prefix_activations,
)
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    PROBE_C,
    PROBE_MAX_ITER,
    PROBE_SEED_DOMAIN,
    PROBE_SOLVER,
    UNIT_NORM_EPS,
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
)
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    render_fit_row,
    template_prefix_length,
)
from steering.eval.axbench_upstream.methods import (
    MethodSetup,
    load_extraction_stack,
    load_method_setup,
    load_target_model,
)
from steering.eval.axbench_upstream.run_methods import _tokenizer_artifact_identity
from steering.safetensors_io import deterministic_safetensors_bytes
from steering.stats_io import content_sha256

FIT_MANIFEST_NAME = "fit_manifest.json"
FIT_MANIFEST_SCHEMA_VERSION = 1
CAPTURE_SITE = "post_cross_attention"
CAPTURE_PREFIX_LENGTH = 2
_FAMILIES = ("diffmean", "linear_probe")
SIGNED_CALIBRATION_KEY = "signed_calibration_max_activation"


@dataclass(frozen=True)
class FitPublication:
    """Completed manifest path and exact output inventory."""

    manifest_path: Path
    artifacts: tuple[Path, ...]


def _strict_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical strict JSON: {exc}") from exc


def _file_identity(path: Path) -> dict[str, int | str]:
    raw_path = Path(path).expanduser().resolve(strict=True)
    with raw_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {
        "name": raw_path.name,
        "byte_length": raw_path.stat().st_size,
        "sha256": digest,
    }


def tokenizer_artifact_identity() -> dict[str, Any]:
    """Return tokenizer provenance recorded once in the fit manifest."""

    identity = _tokenizer_artifact_identity()
    _strict_json_bytes(identity)
    return dict(identity)


def fit_recipe_identity(*, layer: int, decoder_width: int, seed: int) -> dict[str, Any]:
    """Return the complete fixed scientific recipe for one project fit."""

    return {
        "layer": layer,
        "site": CAPTURE_SITE,
        "decoder_width": decoder_width,
        "rendering": "nanochat_full_user_assistant_turn_no_truncation",
        "capture_prefix_length": CAPTURE_PREFIX_LENGTH,
        "retained_tokens": "all_tokens_after_first_two_rendered_control_tokens",
        "conditioning": "full_rendered_row_real_gemma_sae_turn_delimited",
        "diffmean_pooling": "token_weighted_positive_mean_minus_negative_mean",
        "probe_pooling": "per_row_token_mean",
        "normalization": (
            f"raw / (l2_norm + {UNIT_NORM_EPS!r}) for diffmean; F.normalize(coef) for probe"
        ),
        "max_activation": ("signed_fp32_max_over_same_retained_positive_and_negative_tokens"),
        "seed": seed,
        "probe": {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "solver": PROBE_SOLVER,
            "penalty": "L1 via l1_ratio=1.0",
            "C": PROBE_C,
            "max_iter": PROBE_MAX_ITER,
            "class_weight": None,
            "seed_domain": PROBE_SEED_DOMAIN,
        },
    }


def require_fit_direction_normalization(
    direction: t.Tensor,
    *,
    family: str,
    surface: str,
    error_cls: type[Exception] = ValueError,
) -> None:
    """Require the exact normalization family promised by the fit protocol."""

    if family not in _FAMILIES:
        raise error_cls(f"{surface}: unsupported direction family {family!r}")
    norm = float(direction.float().norm().item())
    if not math.isfinite(norm):
        raise error_cls(f"{surface}: direction norm must be finite")
    if norm == 0.0:
        return
    if family == "diffmean":
        # Additive-epsilon normalization is strictly at most one and may be
        # materially below one for tiny raw differences.
        if norm > 1.0 + 1e-6:
            raise error_cls(
                f"{surface}: DiffMean direction norm {norm} exceeds the "
                "additive-epsilon normalization bound"
            )
        return
    if not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6):
        raise error_cls(f"{surface}: linear-probe direction norm {norm} is not unit normalized")


def build_shared_fit_identity(
    *,
    setup: MethodSetup,
    prepared_dir: Path,
    feature_statistics_path: Path,
    layer: int,
    seed: int,
) -> dict[str, Any]:
    """Record the scientific coordinates and compact reproduction metadata."""

    if type(layer) is not int or layer < 0:
        raise ValueError("fit layer must be a nonnegative integer")
    model_config = setup.meta.get("model_config")
    n_layer = model_config.get("n_layer") if isinstance(model_config, Mapping) else None
    decoder_width = model_config.get("n_embd") if isinstance(model_config, Mapping) else None
    if type(n_layer) is not int or layer >= n_layer:
        raise ValueError(f"fit layer {layer} is outside checkpoint n_layer={n_layer!r}")
    if type(decoder_width) is not int or decoder_width <= 0:
        raise ValueError("target metadata model_config.n_embd must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("fit seed must be a nonnegative integer")
    step = setup.meta.get("step")
    if type(step) is not int or step < 0:
        raise ValueError("target metadata step must be a nonnegative integer")
    target_model_path = setup.target_meta_path.parent / f"model_{step:06d}.pt"
    prepared_dir = Path(prepared_dir).resolve(strict=True)
    prepared_train_path = prepared_dir / "generate" / "train_data.parquet"
    statistics_path = Path(feature_statistics_path).resolve(strict=True)
    identity = {
        "checkpoint": {
            "target_model": _file_identity(target_model_path),
            "companion_metadata": _file_identity(setup.target_meta_path),
        },
        "fit": fit_recipe_identity(layer=layer, decoder_width=decoder_width, seed=seed),
        "reproducibility": {
            "prepared_train": _file_identity(prepared_train_path),
            "statistics_content_sha256": content_sha256(statistics_path),
            "tokenizer": tokenizer_artifact_identity(),
        },
    }
    _strict_json_bytes(identity)
    return identity


def _artifact_path(output_root: Path, family: str, concept_id: int) -> Path:
    if family not in _FAMILIES:
        raise ValueError(f"unsupported project direction family {family!r}")
    if type(concept_id) is not int or concept_id < 0:
        raise ValueError("concept_id must be a nonnegative integer")
    return (
        Path(output_root) / family / "encdec" / "positive" / f"concept_{concept_id:04d}.safetensors"
    )


def require_counter_calibration(
    metadata: Mapping[str, str],
    *,
    direction: t.Tensor,
    surface: str,
    error_cls: type[Exception] = ValueError,
) -> float:
    """Validate calibration only; counter coefficients never use this value.

    Nonpositive calibration for a nonzero vector has a separate field, leaving
    scaled consumers' required ``max_activation`` absent so they fail closed.
    Positive fits and zero-vector no-ops retain their original artifact bytes.
    Callers separately validate the artifact coordinates and normalized vector.
    """
    signed = SIGNED_CALIBRATION_KEY in metadata
    if signed and "max_activation" in metadata:
        raise error_cls(f"{surface}: ambiguous signed and scaled calibration metadata")
    value = parse_max_activation(
        {"max_activation": metadata[SIGNED_CALIBRATION_KEY]} if signed else metadata,
        surface=surface,
        error_cls=error_cls,
    )
    if not bool(t.isfinite(direction).all().item()):
        raise error_cls(f"{surface}: direction contains non-finite values")
    zero = bool((direction == 0.0).all().item())
    if signed:
        if zero or value > 0.0:
            raise error_cls(
                f"{surface}: signed calibration requires a nonzero direction and value <= 0"
            )
    elif value < 0.0 or ((value == 0.0) != zero):
        raise error_cls(
            f"{surface}: invalid max_activation for direction; nonpositive counters require {SIGNED_CALIBRATION_KEY}"
        )
    return value


def artifact_metadata(
    *,
    family: str,
    concept_id: int,
    concept: str,
    fitted: FittedDirection,
    shared_identity: Mapping[str, Any],
) -> dict[str, str]:
    """Encode fit provenance and either a usable scale or signed calibration."""

    if not isinstance(concept, str) or not concept:
        raise ValueError("direction concept text must be nonempty")
    fit = shared_identity.get("fit")
    if not isinstance(fit, Mapping):
        raise ValueError("shared fit identity is missing fit coordinates")
    provenance = {
        "concept_id": concept_id,
        "concept": concept,
        "family": family,
        "fit_metadata": fitted.metadata,
    }
    calibration_key = (
        SIGNED_CALIBRATION_KEY
        if fitted.max_activation <= 0.0 and bool((fitted.direction != 0.0).any().item())
        else "max_activation"
    )
    return {
        "concept_id": str(concept_id),
        "layer": str(fit["layer"]),
        "mode": "positive",
        "target_model": "encdec",
        "artifact_kind": expected_artifact_kind(family),
        calibration_key: encode_max_activation(fitted.max_activation),
        "capture_site": str(fit["site"]),
        "provenance_json": _strict_json_bytes(provenance).decode("ascii"),
    }


def write_direction_artifact(
    path: Path,
    *,
    family: str,
    concept_id: int,
    concept: str,
    fitted: FittedDirection,
    shared_identity: Mapping[str, Any],
) -> Path:
    """Atomically write one deterministic project counter direction."""

    path = Path(path)
    expected_path = _artifact_path(path.parents[3], family, concept_id)
    if path != expected_path:
        raise ValueError(f"direction artifact path must be canonical: {path} != {expected_path}")
    direction = fitted.direction.detach().to(dtype=t.float32, device="cpu").contiguous()
    if (
        direction.ndim != 1
        or direction.numel() == 0
        or not bool(t.isfinite(direction).all().item())
    ):
        raise ValueError("fitted direction must be a finite nonempty float32 vector")
    fit = shared_identity.get("fit")
    width = fit.get("decoder_width") if isinstance(fit, Mapping) else None
    if direction.numel() != width:
        raise ValueError(
            f"fitted direction width {direction.numel()} does not match decoder width {width!r}"
        )
    require_fit_direction_normalization(
        direction,
        family=family,
        surface=f"project fit {family}/c{concept_id}",
    )
    metadata = artifact_metadata(
        family=family,
        concept_id=concept_id,
        concept=concept,
        fitted=fitted,
        shared_identity=shared_identity,
    )
    require_counter_calibration(
        metadata,
        direction=direction,
        surface=f"project fit {family}/c{concept_id}",
    )
    atomic_write_bytes(
        deterministic_safetensors_bytes({"direction": direction}, metadata=metadata),
        path,
    )
    return path


def fit_direction_pair(
    positive_activations: Sequence[t.Tensor],
    negative_activations: Sequence[t.Tensor],
    *,
    layer: int,
    seed: int,
    concept_id: int,
    surface: str,
) -> dict[str, FittedDirection]:
    """Fit both families from the same retained per-concept activations."""

    diffmean = fit_diffmean_direction(
        positive_activations,
        negative_activations,
        intervention_layer=layer,
        surface=f"{surface} DiffMean",
        calibration_only=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        probe = fit_probe_direction(
            positive_activations,
            negative_activations,
            intervention_layer=layer,
            seed=seed,
            concept_id=concept_id,
            surface=f"{surface} LinearProbe",
            calibration_only=True,
        )
    convergence = [
        warning for warning in caught if issubclass(warning.category, ConvergenceWarning)
    ]
    if convergence:
        raise RuntimeError(
            f"{surface}: linear probe emitted a convergence warning: {convergence[0].message}"
        )
    return {"diffmean": diffmean, "linear_probe": probe}


def publish_fit_manifest(
    output_root: Path,
    *,
    shared_identity: Mapping[str, Any],
    artifacts: Sequence[Path],
) -> Path:
    """Validate the complete tree and atomically publish its manifest last."""

    output_root = Path(output_root).resolve()
    inventory: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_path in sorted(artifacts):
        path = Path(raw_path).resolve(strict=True)
        if not path.is_relative_to(output_root):
            raise ValueError(f"fit artifact is outside output root: {path}")
        relative = path.relative_to(output_root).as_posix()
        if relative in seen:
            raise ValueError(f"duplicate fit artifact path {relative}")
        seen.add(relative)
        parts = Path(relative).parts
        if len(parts) != 4:
            raise ValueError(f"malformed fit artifact path {relative}")
        family = parts[0]
        concept_id = int(path.stem.removeprefix("concept_"))
        loaded = load_direction_artifact(path, error_type=ValueError)
        if loaded.identity.family != family or loaded.identity.concept_id != concept_id:
            raise ValueError(f"fit artifact coordinates differ from its path: {path}")
        inventory.append(
            {
                "path": relative,
                "family": family,
                "concept_id": concept_id,
            }
        )
    manifest = {
        "schema_version": FIT_MANIFEST_SCHEMA_VERSION,
        "status": "complete",
        "shared_identity": shared_identity,
        "artifacts": inventory,
    }
    path = output_root / FIT_MANIFEST_NAME
    if path.exists():
        raise FileExistsError(f"fit manifest already exists: {path}")
    atomic_write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        path,
    )
    return path


def _prepare_rows(
    setup: MethodSetup,
    extraction_model: Any,
    sae: Any,
    tokenizer: Any,
    training: Mapping[int, ConceptTrainingPairs],
    *,
    sequence_len: int,
) -> dict[int, dict[str, tuple[FitRowConditioning, ...]]]:
    prepared: dict[int, dict[str, tuple[FitRowConditioning, ...]]] = {}
    for concept_id, examples in training.items():
        classes: dict[str, tuple[FitRowConditioning, ...]] = {}
        for class_name, pairs in (
            ("positive", examples.positive),
            ("negative", examples.negative),
        ):
            rows: list[FitRowConditioning] = []
            for row_index, (input_text, output_text) in enumerate(pairs):
                surface = f"concept {concept_id} {class_name} row {row_index}"
                rendered = render_fit_row(
                    tokenizer,
                    input_text=input_text,
                    output_text=output_text,
                    sequence_len=sequence_len,
                    surface=surface,
                )
                rows.append(
                    build_fit_row_conditioning(
                        setup,
                        extraction_model,
                        sae,
                        tokenizer,
                        token_ids=rendered.ids,
                        surface=surface,
                    )
                )
            classes[class_name] = tuple(rows)
        prepared[concept_id] = classes
    return prepared


def fit_project_directions(
    *,
    prepared_dir: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    output_root: Path,
    max_concepts: int,
    layer: int,
    seed: int,
    device: str,
) -> FitPublication:
    """Fit and publish both project direction families for all concepts."""

    if type(max_concepts) is not int or max_concepts <= 0:
        raise ValueError("max_concepts must be a positive integer")
    prepared_dir = Path(prepared_dir).expanduser().resolve(strict=True)
    target_meta_path = Path(target_meta_path).expanduser().resolve(strict=True)
    feature_statistics_path = Path(feature_statistics_path).expanduser().resolve(strict=True)
    output_root = Path(output_root).expanduser().resolve()
    inputs = [
        prepared_dir,
        target_meta_path,
        feature_statistics_path,
    ]
    if any(
        output_root == path or output_root.is_relative_to(path) or path.is_relative_to(output_root)
        for path in inputs
    ):
        raise ValueError("fit output root must be path-disjoint from every input")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"fit output root must be new or empty; partial publications are not resumed: {output_root}"
        )

    setup = load_method_setup(
        target_meta_path,
        feature_statistics_path,
        require_exact_statistics_identity=True,
    )
    shared_identity = build_shared_fit_identity(
        setup=setup,
        prepared_dir=prepared_dir,
        feature_statistics_path=feature_statistics_path,
        layer=layer,
        seed=seed,
    )
    model_config = setup.meta.get("model_config")
    sequence_len = model_config.get("sequence_len") if isinstance(model_config, Mapping) else None
    if type(sequence_len) is not int or sequence_len <= 0:
        raise ValueError("target metadata model_config.sequence_len must be positive")
    training = select_training_pairs(prepared_dir, max_concepts=max_concepts)
    tokenizer = get_tokenizer()
    if template_prefix_length(tokenizer) != CAPTURE_PREFIX_LENGTH:
        raise ValueError(
            f"fit template prefix must contain exactly {CAPTURE_PREFIX_LENGTH} control tokens"
        )

    extraction_model: Any | None = None
    sae: Any | None = None
    try:
        logger.info("Preparing real SAE conditioning for {} fit concepts", max_concepts)
        extraction_model, sae = load_extraction_stack(setup, device)
        prepared_rows = _prepare_rows(
            setup,
            extraction_model,
            sae,
            tokenizer,
            training,
            sequence_len=sequence_len,
        )
    finally:
        extraction_model = None
        sae = None
        gc.collect()
        if device.startswith("cuda") and t.cuda.is_available():
            t.cuda.empty_cache()

    logger.info("Loading target checkpoint for post-cross-attention direction fitting")
    model, loaded_tokenizer = load_target_model(setup, device)
    del loaded_tokenizer
    model.eval()
    if int(model.dec_config.n_embd) != shared_identity["fit"]["decoder_width"]:
        raise ValueError("loaded target decoder width differs from fit identity")

    output_root.mkdir(parents=True, exist_ok=True)
    artifact_paths: list[Path] = []
    for concept_id in range(max_concepts):
        examples = training[concept_id]
        rows = prepared_rows[concept_id]
        positive = [
            capture_post_prefix_activations(
                model,
                row,
                layer=layer,
                prefix_length=CAPTURE_PREFIX_LENGTH,
                capture_site=CAPTURE_SITE,
            )
            for row in rows["positive"]
        ]
        negative = [
            capture_post_prefix_activations(
                model,
                row,
                layer=layer,
                prefix_length=CAPTURE_PREFIX_LENGTH,
                capture_site=CAPTURE_SITE,
            )
            for row in rows["negative"]
        ]
        fitted = fit_direction_pair(
            positive,
            negative,
            layer=layer,
            seed=seed,
            concept_id=concept_id,
            surface=f"concept {concept_id}",
        )
        for family in _FAMILIES:
            path = _artifact_path(output_root, family, concept_id)
            artifact_paths.append(
                write_direction_artifact(
                    path,
                    family=family,
                    concept_id=concept_id,
                    concept=examples.concept,
                    fitted=fitted[family],
                    shared_identity=shared_identity,
                )
            )
        # Activations are deliberately concept-bounded and released here.
        del positive, negative, fitted
        logger.info("Fitted project directions for concept {}/{}", concept_id + 1, max_concepts)

    manifest_path = publish_fit_manifest(
        output_root,
        shared_identity=shared_identity,
        artifacts=artifact_paths,
    )
    return FitPublication(manifest_path=manifest_path, artifacts=tuple(artifact_paths))


__all__ = (
    "CAPTURE_PREFIX_LENGTH",
    "CAPTURE_SITE",
    "FIT_MANIFEST_NAME",
    "FIT_MANIFEST_SCHEMA_VERSION",
    "FitPublication",
    "artifact_metadata",
    "build_shared_fit_identity",
    "fit_recipe_identity",
    "fit_direction_pair",
    "fit_project_directions",
    "publish_fit_manifest",
    "require_fit_direction_normalization",
    "require_counter_calibration",
    "SIGNED_CALIBRATION_KEY",
    "tokenizer_artifact_identity",
    "write_direction_artifact",
)
