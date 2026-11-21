"""Native decoder standalone methods under AxBench-Adv pressures.

The standard producer remains authoritative for positive DiffMean,
LinearProbe, and LoRA fits.  This module authenticates the verified union and
its original per-shard native artifacts, selects factors on Adv concepts
0..49, then applies GEPA or raw additive counter pressures without refitting.
"""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, Protocol, cast

import pandas as pd
import torch as t
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

from scripts.eval.axbench_adv.pressures import (
    COUNTER_PRESSURES,
    DEFAULT_MAGNITUDES,
    DIRECTION_METHOD,
    Pressure,
    magnitude_label,
    parse_pressures,
    validate_magnitudes,
)
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, INPUT_COLUMNS
from steering.eval.axbench_adv.best_point import (
    BestPoint,
    load_steering_entries,
    select_best_points,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.execution_scope import validate_concept_ids
from steering.eval.axbench_adv.final_judge import build_point_frame, write_point_frame
from steering.eval.axbench_adv.fitness import evaluate_fitness
from steering.eval.axbench_adv.gepa_optimization import (
    OptimizationExample,
    RawEvaluation,
    optimize_suffix,
)
from steering.eval.axbench_adv.native_counter import scoped_native_counter
from steering.eval.axbench_adv.native_decoder import attacked_user_text
from steering.eval.axbench_adv.native_mailbox import (
    ArtifactMailbox,
    MailboxReflection,
    checked_raw,
    digest_json,
    read_json,
    write_json,
)
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix
from steering.eval.axbench_adv.types import PROPOSAL_COUNTS
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    load_direction,
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
    _derive_lora_seed,
    load_lora,
)
from steering.eval.axbench_upstream.native_baseline_judge import (
    SUMMARY,
    validate_native_baseline_judged,
)
from steering.eval.axbench_upstream.native_baselines import (
    CAPTURE_PREFIX_LENGTH,
    CAPTURE_SITE,
    D14_MODEL_CONTRACT,
    DIRECTION_METHODS,
    LEGACY_MODEL_CONTRACT,
    METHOD_FACTORS,
    METHOD_KIND,
    NativeModelContract,
    derive_cell_seed,
    native_baseline_cell_filename,
    resolve_native_model_contract,
    scoped_native_direction,
    scoped_native_lora,
)
from steering.eval.axbench_upstream.native_baselines import (
    FORMAT as BASELINE_FORMAT,
)
from steering.eval.axbench_upstream.native_baselines import (
    METHODS as NATIVE_METHODS,
)
from steering.eval.axbench_upstream.native_contract import sha256
from steering.eval.axbench_upstream.native_decoder import (
    NativeDecoderAdapter,
    canonical_bytes,
    checkpoint_identity,
    native_runtime_settings,
)
from steering.immutable_io import install_bytes_once

FORMAT = "axbench-native-decoder-method-adv-v1"
UNION_FORMAT = "verified-native-decoder-method-union-v1"
ZERO_CONTROL_ARCHIVED = "archived"
ZERO_CONTROL_RUNTIME = "runtime"
ADV_SELECTION_CONCEPT_IDS = tuple(range(50))
METHODS = NATIVE_METHODS
PRESSURES = (
    Pressure.GEPA,
    Pressure.LINEAR_PROBE_COUNTER,
    Pressure.DIFFMEAN_COUNTER,
)
COUNTER_MAGNITUDES = DEFAULT_MAGNITUDES
_METHOD_SLUG = dict(METHOD_KIND)
# Public legacy aliases retained for callers/tests that imported the original
# D12 constants. Production paths resolve the authenticated geometry contract.
INTERVENTION_LAYER = LEGACY_MODEL_CONTRACT.intervention_layer
LORA_LAYERS = LEGACY_MODEL_CONTRACT.lora_layers


def _validate_zero_control_source(value: Any) -> str:
    source = str(value)
    if source not in {ZERO_CONTROL_ARCHIVED, ZERO_CONTROL_RUNTIME}:
        raise AxBenchAdvError("zero_control_source must be archived or runtime")
    return source


def _identity_zero_control_source(identity: Mapping[str, Any]) -> str:
    source = identity.get("zero_control_source", ZERO_CONTROL_ARCHIVED)
    if source == ZERO_CONTROL_ARCHIVED and "zero_control_source" in identity:
        raise AxBenchAdvError("archived zero-control identity must omit its default policy")
    return _validate_zero_control_source(source)


def _validate_pressures(configured: Sequence[Any]) -> tuple[Pressure, ...]:
    try:
        pressures = parse_pressures(tuple(str(value) for value in configured), environ={})
    except (TypeError, ValueError) as exc:
        raise AxBenchAdvError("native method Adv pressures are invalid") from exc
    if any(pressure not in PRESSURES for pressure in pressures):
        raise AxBenchAdvError(
            f"native method Adv pressures must be a nonempty subset of "
            f"{[str(pressure) for pressure in PRESSURES]}"
        )
    return pressures


class NativeMethodAdapter(Protocol):
    model: nn.Module

    def prompt_tokens(self, user_text: str) -> list[int]: ...

    def generate_user_text(
        self,
        user_text: str,
        *,
        seed: int,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> str: ...


@dataclass(frozen=True)
class NativeFitRef:
    """One authenticated standalone fit, relocatable from its producer host."""

    method: str
    concept_id: int
    path: Path
    sha256: str
    direction: FittedDirection | None
    lora_state: Mapping[str, t.Tensor] | None
    lora_metadata: Mapping[str, Any] | None


@dataclass(frozen=True)
class NativeMethodSources:
    """Complete source authentication needed before model or provider work."""

    source_frame: pd.DataFrame
    generation_frames: Mapping[str, pd.DataFrame]
    judged_entries: Mapping[str, tuple[dict[str, Any], ...]]
    best_points: Mapping[str, BestPoint]
    fits: Mapping[tuple[str, int], NativeFitRef]
    checkpoint: Mapping[str, Any]
    geometry: Mapping[str, int]
    generation: Mapping[str, Any]
    identity: Mapping[str, Any]


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise AxBenchAdvError(f"{label} must be a string-keyed mapping")
    return dict(value)


def _lower_sha(value: Any, *, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AxBenchAdvError(f"{label} must be a lowercase SHA-256")
    return value


def _checkpoint_scientific_identity(value: Any) -> dict[str, Any]:
    checkpoint = _mapping(value, label="native baseline checkpoint")
    step = checkpoint.get("step")
    if type(step) is not int or step < 0:
        raise AxBenchAdvError("native baseline checkpoint step differs")
    result = {"step": step}
    for field in ("model_sha256", "meta_sha256", "tokenizer_sha256"):
        result[field] = _lower_sha(
            checkpoint.get(field), label=f"native baseline checkpoint {field}"
        )
    return result


def _json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        return read_json(path)
    except (OSError, ValueError, TypeError) as exc:
        raise AxBenchAdvError(f"invalid {label}: {path}") from exc


def _validate_union(final_root: Path) -> tuple[dict[str, Any], dict[str, str]]:
    provenance_path = final_root / "union-provenance.json"
    document = _json(provenance_path, label="native method union provenance")
    required = {
        "format",
        "native_run_manifest",
        "concept_ids",
        "methods",
        "parents",
        "scores",
        "provider_calls",
        "files",
    }
    if set(document) != required:
        raise AxBenchAdvError("native method union provenance fields differ")
    if (
        document["format"] != UNION_FORMAT
        or document["native_run_manifest"] is not False
        or document["concept_ids"] != list(range(100))
        or document["methods"] != list(METHODS)
        or document["provider_calls"] != 0
    ):
        raise AxBenchAdvError("native method union identity/population differs")
    parents = document["parents"]
    if not isinstance(parents, list) or not parents:
        raise AxBenchAdvError("native method union must retain parent archives")
    parent_coordinates: set[tuple[str, str]] = set()
    for parent in parents:
        if (
            not isinstance(parent, dict)
            or set(parent) != {"remote", "sha256"}
            or not isinstance(parent["remote"], str)
            or not parent["remote"].strip()
        ):
            raise AxBenchAdvError("native method union parent record differs")
        coordinate = (
            parent["remote"],
            _lower_sha(parent["sha256"], label="native method union parent sha256"),
        )
        if coordinate in parent_coordinates:
            raise AxBenchAdvError("native method union parent archives are duplicated")
        parent_coordinates.add(coordinate)
    raw_files = _mapping(document["files"], label="native method union files")
    files: dict[str, str] = {}
    for relative, expected in raw_files.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.name == "union-provenance.json":
            raise AxBenchAdvError("native method union file path escapes its root")
        files[relative] = _lower_sha(expected, label=f"union file {relative}")
    observed = {
        str(path.relative_to(final_root))
        for path in final_root.rglob("*")
        if path.is_file() and path != provenance_path
    }
    if observed != set(files):
        raise AxBenchAdvError("native method union file inventory differs")
    for relative, expected in files.items():
        if sha256(final_root / relative) != expected:
            raise AxBenchAdvError(f"native method union file hash differs: {relative}")
    return document, files


def _validate_archive_parents(union: Mapping[str, Any], archive_roots: Sequence[Path]) -> None:
    """Bind extracted native files to the union-declared archive bytes."""

    expected = sorted(str(parent["sha256"]) for parent in union["parents"])
    observed: list[str] = []
    for root in archive_roots:
        archives = [
            path
            for path in (
                root.parent / "judged-archive.tar.gz",
                root.parent.parent / "judged.tar.gz",
            )
            if path.is_file()
        ]
        if len(archives) != 1:
            raise AxBenchAdvError(f"native archive parent evidence differs: {root}")
        archive_path = archives[0]
        observed.append(sha256(archive_path))
        archived_native: dict[Path, str] = {}
        try:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                for member in archive:
                    member_path = PurePosixPath(member.name)
                    if not member_path.parts or member_path.parts[0] != "native":
                        continue
                    if member_path.is_absolute() or ".." in member_path.parts:
                        raise AxBenchAdvError("native archive member path escapes its root")
                    if member.isdir():
                        continue
                    if not member.isfile() or len(member_path.parts) == 1:
                        raise AxBenchAdvError("native archive contains a non-file native member")
                    relative = Path(*member_path.parts[1:])
                    if relative in archived_native:
                        raise AxBenchAdvError("native archive contains duplicate native members")
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise AxBenchAdvError("native archive file cannot be read")
                    digest = hashlib.sha256()
                    while chunk := stream.read(1024 * 1024):
                        digest.update(chunk)
                    archived_native[relative] = digest.hexdigest()
        except (OSError, tarfile.TarError) as exc:
            raise AxBenchAdvError(
                f"native archive cannot be authenticated: {archive_path}"
            ) from exc
        if not archived_native:
            raise AxBenchAdvError(f"native archive has no native files: {root}")
        observed_native = {path.relative_to(root) for path in root.rglob("*") if path.is_file()}
        if set(archived_native) != observed_native:
            raise AxBenchAdvError(f"native archive extracted inventory differs: {root}")
        for relative, expected_hash in archived_native.items():
            if sha256(root / relative) != expected_hash:
                raise AxBenchAdvError(f"native archive extracted file differs: {root / relative}")
    if sorted(observed) != expected:
        raise AxBenchAdvError("native archive roots differ from union parent archives")


def _aggregate_rows(path: Path, *, method: str) -> list[dict[str, Any]]:
    frame = pd.read_csv(path)
    required = {"method", "factor", "n_concepts", *SUMMARY}
    if set(frame.columns) != required or len(frame) != len(METHOD_FACTORS[method]):
        raise AxBenchAdvError(f"native {method} aggregate columns/rows differ")
    rows = frame.to_dict(orient="records")
    if [float(row["factor"]) for row in rows] != list(METHOD_FACTORS[method]):
        raise AxBenchAdvError(f"native {method} aggregate factor grid differs")
    for row in rows:
        if row["method"] != method or row["n_concepts"] != 100:
            raise AxBenchAdvError(f"native {method} aggregate population differs")
        for field in SUMMARY:
            value = row[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AxBenchAdvError(f"native {method} aggregate rating is not numeric")
            if not math.isfinite(float(value)) or not 0 <= float(value) <= 2:
                raise AxBenchAdvError(f"native {method} aggregate rating is outside [0,2]")
    return rows


def _score_row_matches(actual: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    if actual.keys() != expected.keys():
        return False
    for key, value in expected.items():
        if key in SUMMARY:
            candidate = actual[key]
            if (
                isinstance(candidate, bool)
                or not isinstance(candidate, (int, float))
                or not math.isclose(float(candidate), float(value), rel_tol=0.0, abs_tol=1e-12)
            ):
                return False
        elif actual[key] != value:
            return False
    return True


def _validate_union_method(
    final_root: Path,
    union: Mapping[str, Any],
    *,
    method: str,
) -> tuple[pd.DataFrame, tuple[dict[str, Any], ...]]:
    directory = final_root / _METHOD_SLUG[method]
    generation_path = directory / "generation.parquet"
    generation = pd.read_parquet(generation_path)
    calculated = validate_native_baseline_judged(generation, directory, method=method)
    entries = tuple(
        load_steering_entries(directory / "steering.jsonl", expected_concept_ids=tuple(range(100)))
    )
    rows = _aggregate_rows(directory / "aggregate.csv", method=method)
    expected_scores = [
        {
            "method": method,
            "factor": factor,
            "n_concepts": 100,
            **{field: calculated[field][float(factor).hex()] for field in SUMMARY},
        }
        for factor in METHOD_FACTORS[method]
    ]
    for actual, expected in zip(rows, expected_scores, strict=True):
        if not _score_row_matches(actual, expected):
            raise AxBenchAdvError(f"native {method} aggregate arithmetic differs")
    union_scores = union.get("scores")
    method_scores = (
        [row for row in union_scores if isinstance(row, dict) and row.get("method") == method]
        if isinstance(union_scores, list)
        else []
    )
    if (
        not isinstance(union_scores, list)
        or len(method_scores) != len(expected_scores)
        or any(
            not _score_row_matches(actual, expected)
            for actual, expected in zip(method_scores, expected_scores, strict=True)
        )
    ):
        raise AxBenchAdvError(f"native {method} union scores differ from judged data")
    return generation, entries


def _common_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    configuration = _mapping(identity.get("configuration"), label="native baseline configuration")
    inputs = _mapping(identity.get("inputs"), label="native baseline inputs")
    geometry = _mapping(identity.get("geometry"), label="native baseline geometry")
    if set(geometry) != {"n_layer", "n_embd", "sequence_len"} or any(
        type(value) is not int for value in geometry.values()
    ):
        raise AxBenchAdvError("native baseline geometry fields differ")
    model_contract = resolve_native_model_contract(cast(dict[str, int], geometry))
    if (
        identity.get("schema_version") != 1
        or identity.get("format") != BASELINE_FORMAT
        or configuration.get("max_concepts") != 100
        or configuration.get("direction_factors") != list(FULL_FACTORS)
        or configuration.get("lora_factors") != [1.0]
        or configuration.get("layer") != model_contract.intervention_layer
        or configuration.get("capture_site") != CAPTURE_SITE
        or configuration.get("capture_prefix_length") != CAPTURE_PREFIX_LENGTH
        or configuration.get("lora_layers") != list(model_contract.lora_layers)
        or configuration.get("seed") != 42
        or configuration.get("prompt") != "original_prompt only"
        or configuration.get("training_split") != "TRAIN only"
        or configuration.get("direction_scale")
        != "factor * fit-row max_activation * unit direction"
    ):
        raise AxBenchAdvError("native baseline semantic identity differs")
    methods = configuration.get("methods")
    concepts = configuration.get("concept_ids")
    if (
        not isinstance(methods, list)
        or not methods
        or any(method not in METHODS for method in methods)
        or len(set(methods)) != len(methods)
        or not isinstance(concepts, list)
        or not concepts
        or any(type(concept) is not int for concept in concepts)
        or concepts != list(range(concepts[0], concepts[-1] + 1))
    ):
        raise AxBenchAdvError("native baseline methods/concepts are invalid")
    generation = configuration.get("generation")
    if generation != {"max_new_tokens": 128, "temperature": 1.0, "top_k": 50}:
        raise AxBenchAdvError("native baseline generation policy differs")
    checkpoint = _checkpoint_scientific_identity(inputs.get("checkpoint"))
    renderer = configuration.get("renderer")
    if renderer not in {None, "RustBPE.render_conversation(max_tokens=None) + assistant_start"}:
        raise AxBenchAdvError("native baseline renderer identity differs")
    scientific_inputs = {
        "checkpoint": checkpoint,
        **{
            field: {
                "sha256": _lower_sha(
                    _mapping(inputs.get(field), label=f"native baseline {field}").get("sha256"),
                    label=f"native baseline {field} sha256",
                )
            }
            for field in ("source", "prepared_metadata", "prepared_training_data")
        },
    }
    return {
        "inputs": scientific_inputs,
        "geometry": geometry,
        "generation": generation,
        "configuration": {
            key: value
            for key, value in configuration.items()
            if key not in {"methods", "concept_ids", "renderer"}
        },
    }


def _validate_direction_fit(
    fitted: FittedDirection,
    *,
    method: str,
    concept_id: int,
    concept: str,
    identity_sha256: str,
    model_contract: NativeModelContract = LEGACY_MODEL_CONTRACT,
) -> None:
    contract = fitted.metadata.get("native_contract")
    expected_method = "diffmean" if method == "DiffMean" else "probe"
    if (
        not isinstance(contract, dict)
        or fitted.direction.shape != (model_contract.n_embd,)
        or fitted.direction.dtype is not t.float32
        or fitted.direction.device.type != "cpu"
        or not fitted.direction.is_contiguous()
        or not bool(t.isfinite(fitted.direction).all())
        or not math.isfinite(fitted.max_activation)
        or fitted.max_activation < 0
        or fitted.metadata.get("method") != expected_method
        or contract.get("format") != BASELINE_FORMAT
        or contract.get("identity_sha256") != identity_sha256
        or contract.get("method") != method
        or contract.get("concept_id") != concept_id
        or contract.get("concept") != concept
        or contract.get("training_split") != "TRAIN"
        or contract.get("capture_site") != CAPTURE_SITE
        or contract.get("intervention_layer") != model_contract.intervention_layer
        or contract.get("n_layer") != model_contract.n_layer
        or contract.get("n_embd") != model_contract.n_embd
    ):
        raise AxBenchAdvError(f"native standalone {method} fit contract differs for c{concept_id}")
    norm = float(fitted.direction.norm().item())
    if not math.isfinite(norm) or norm <= 0 or norm > 1.0 + 1e-5:
        raise AxBenchAdvError(f"native standalone {method} direction norm is invalid")
    if method == "LinearProbe" and (
        not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
        or fitted.metadata.get("seed") != 42
        or fitted.metadata.get("concept_id") != concept_id
    ):
        raise AxBenchAdvError("native standalone LinearProbe normalization/provenance differs")


def _validate_lora_fit(
    path: Path,
    *,
    concept_id: int,
    contract_sha256: str,
    model_contract: NativeModelContract = LEGACY_MODEL_CONTRACT,
) -> tuple[dict[str, t.Tensor], dict[str, Any]]:
    state, metadata = load_lora(path)
    expected_shapes = {
        f"layer_{layer}_attn_c_proj_lora_A": (LORA_RANK, model_contract.n_embd)
        for layer in model_contract.lora_layers
    } | {
        f"layer_{layer}_attn_c_proj_lora_B": (model_contract.n_embd, LORA_RANK)
        for layer in model_contract.lora_layers
    }
    if set(state) != set(expected_shapes) or any(
        tensor.shape != expected_shapes[name]
        or tensor.dtype is not t.float32
        or tensor.device.type != "cpu"
        or not tensor.is_contiguous()
        or not bool(t.isfinite(tensor).all())
        for name, tensor in state.items()
    ):
        raise AxBenchAdvError(f"native standalone LoRA state differs for c{concept_id}")
    n_rows = metadata.get("n_rows")
    if not isinstance(n_rows, str) or not n_rows.isdecimal() or int(n_rows) <= 0:
        raise AxBenchAdvError("native standalone LoRA row count is invalid")
    optimizer_steps = LORA_EPOCHS * math.ceil(
        math.ceil(int(n_rows) / LORA_BATCH_ROWS) / LORA_GRAD_ACCUM
    )
    expected = {
        "format": BASELINE_FORMAT,
        "contract_sha256": contract_sha256,
        "rank": str(LORA_RANK),
        "alpha": str(LORA_ALPHA),
        "scaling": str(LORA_ALPHA / math.sqrt(LORA_RANK)),
        "dropout": str(LORA_DROPOUT),
        "layers": str(list(model_contract.lora_layers)),
        "seed": str(_derive_lora_seed(42, concept_id)),
        "seed_domain": LORA_SEED_DOMAIN,
        "n_rows": n_rows,
        "epochs": str(LORA_EPOCHS),
        "batch_rows": str(LORA_BATCH_ROWS),
        "gradient_accumulation_batches": str(LORA_GRAD_ACCUM),
        "learning_rate": str(LORA_LR),
        "weight_decay": str(LORA_WEIGHT_DECAY),
        "optimizer_steps": str(optimizer_steps),
        "scheduler_horizon": str(optimizer_steps),
        "population": "positive TRAIN rows only",
    }
    if metadata != expected:
        raise AxBenchAdvError(f"native standalone LoRA metadata differs for c{concept_id}")
    return state, metadata


def _fit_contract_from_direction(
    direction_fits: Mapping[tuple[str, int], NativeFitRef],
    concept_id: int,
    *,
    identity_sha256: str,
    model_contract: NativeModelContract = LEGACY_MODEL_CONTRACT,
) -> str:
    for method in DIRECTION_METHODS:
        ref = direction_fits.get((method, concept_id))
        if ref is None or ref.direction is None:
            continue
        contract = ref.direction.metadata.get("native_contract")
        if isinstance(contract, dict):
            return digest_json(
                {
                    **contract,
                    "identity_sha256": identity_sha256,
                    "method": "LoRA",
                    "capture_site": None,
                    "intervention_layer": None,
                    "lora_layers": list(model_contract.lora_layers),
                    "population": "positive TRAIN completion targets only",
                }
            )
    raise AxBenchAdvError(f"LoRA c{concept_id} requires a matched direction fit contract")


def _validate_cells(
    *,
    method_root: Path,
    method: str,
    concepts: Sequence[int],
    frame: pd.DataFrame,
    fit_hashes: Mapping[int, str],
    method_identity: Mapping[str, Any],
) -> None:
    cells = method_root / "cells"
    factors = METHOD_FACTORS[method]
    expected_names = {
        native_baseline_cell_filename(concept_id, input_id, factor)
        for concept_id in concepts
        for input_id in range(10)
        for factor in factors
    }
    if (
        not cells.is_dir()
        or {path.name for path in cells.iterdir() if path.is_file()} != expected_names
    ):
        raise AxBenchAdvError(f"native standalone {method} cell coverage differs")
    identity_sha = digest_json(method_identity)
    indexed = frame.set_index(["concept_id", "input_id", "factor"], verify_integrity=True)
    for concept_id in concepts:
        for input_id in range(10):
            for factor in factors:
                record = _json(
                    cells / native_baseline_cell_filename(concept_id, input_id, factor),
                    label=f"native standalone {method} cell",
                )
                row = indexed.loc[(concept_id, input_id, factor)]
                cell = {
                    "concept_id": concept_id,
                    "input_id": input_id,
                    "factor_hex": float(factor).hex(),
                    "seed": derive_cell_seed(42, method, concept_id, input_id, factor),
                    "fit_sha256": fit_hashes[concept_id],
                    "prompt_sha256": hashlib.sha256(
                        str(row["original_prompt"]).encode()
                    ).hexdigest(),
                }
                if (
                    set(record) != {"method_identity_sha256", "cell", "generation"}
                    or record.get("method_identity_sha256") != identity_sha
                    or record.get("cell") != cell
                    or record.get("generation") != row[f"{method}_steered_generation"]
                ):
                    raise AxBenchAdvError(f"native standalone {method} cell differs")


def _validate_archive_method(
    archive_root: Path,
    *,
    method: str,
    union_frame: pd.DataFrame,
    common: Mapping[str, Any] | None,
    direction_fits: Mapping[tuple[str, int], NativeFitRef],
) -> tuple[dict[str, Any], list[NativeFitRef], pd.DataFrame]:
    identity_path = archive_root / "identity.json"
    method_root = archive_root / "methods" / _METHOD_SLUG[method]
    identity = _json(identity_path, label="native standalone identity")
    observed_common = _common_identity(identity)
    model_contract = resolve_native_model_contract(observed_common["geometry"])
    if common is not None and observed_common != common:
        raise AxBenchAdvError("native standalone shard scientific identities differ")
    configuration = cast(dict[str, Any], identity["configuration"])
    if method not in configuration["methods"]:
        raise AxBenchAdvError(f"native standalone shard does not declare {method}")
    concepts = tuple(configuration["concept_ids"])
    run = _json(method_root / "run_manifest.json", label=f"native {method} run manifest")
    fit = _json(method_root / "fit_manifest.json", label=f"native {method} fit manifest")
    identity_sha = digest_json(identity)
    factors = list(METHOD_FACTORS[method])
    expected_operation = (
        "residual += factor * fit_max_activation * saved_fp32_unit_direction"
        if method in DIRECTION_METHODS
        else "trained rsLoRA adapters active at factor 1.0"
    )
    method_identity = {
        "run_identity_sha256": identity_sha,
        "method": method,
        "factors": factors,
        "concept_ids": list(concepts),
        "prompt": "canonical original user prompt without boosted steering instruction",
        "operation": expected_operation,
    }
    if (
        run.get("schema_version") != 1
        or run.get("format") != BASELINE_FORMAT
        or run.get("method") != method
        or run.get("factors") != factors
        or run.get("complete") is not True
        or run.get("identity_sha256") != identity_sha
        or run.get("method_identity") != method_identity
        or run.get("generation")
        != {"seed": 42, "max_new_tokens": 128, "temperature": 1.0, "top_k": 50}
        or run.get("source", {}).get("concept_ids") != list(concepts)
        or run.get("source", {}).get("max_concepts") != 100
        or run.get("source", {}).get("input_sha256")
        != observed_common["inputs"]["source"]["sha256"]
        or _checkpoint_scientific_identity(run.get("checkpoint"))
        != observed_common["inputs"]["checkpoint"]
    ):
        raise AxBenchAdvError(f"native standalone {method} run contract differs")
    if method in DIRECTION_METHODS:
        if run["policy"] != {
            "prompt": "original_prompt",
            "boosted_prompt_used": False,
            "training_split": "TRAIN",
            "layer": model_contract.intervention_layer,
            "lora_layers": None,
        }:
            raise AxBenchAdvError(f"native standalone {method} intervention policy differs")
    elif run["policy"] != {
        "prompt": "original_prompt",
        "boosted_prompt_used": False,
        "training_split": "TRAIN",
        "layer": None,
        "lora_layers": list(model_contract.lora_layers),
    }:
        raise AxBenchAdvError("native standalone LoRA intervention policy differs")
    if (
        sha256(identity_path) != run.get("identity", {}).get("sha256")
        or sha256(method_root / "fit_manifest.json") != run.get("fit_manifest", {}).get("sha256")
        or fit.get("schema_version") != 1
        or fit.get("format") != BASELINE_FORMAT
        or fit.get("complete") is not True
        or fit.get("method") != method
        or fit.get("identity_sha256") != identity_sha
        or fit.get("concept_ids") != list(concepts)
    ):
        raise AxBenchAdvError(f"native standalone {method} identity/fit manifest differs")
    output_path = method_root / "steering_data.parquet"
    if sha256(output_path) != run.get("output", {}).get("sha256"):
        raise AxBenchAdvError(f"native standalone {method} output hash differs")
    frame = pd.read_parquet(output_path)
    expected_keys = [
        (concept_id, input_id, factor)
        for concept_id in concepts
        for input_id in range(10)
        for factor in METHOD_FACTORS[method]
    ]
    if (
        list(frame[["concept_id", "input_id", "factor"]].itertuples(index=False, name=None))
        != expected_keys
        or len(frame) != run["output"].get("rows")
        or not frame["native_user_prompt"].equals(frame["original_prompt"])
    ):
        raise AxBenchAdvError(f"native standalone {method} source grid differs")
    union_subset = union_frame.loc[union_frame["concept_id"].isin(concepts)].reset_index(drop=True)
    if not frame.equals(union_subset):
        raise AxBenchAdvError(f"native standalone {method} shard differs from verified union")
    artifacts = fit.get("artifacts")
    if not isinstance(artifacts, list) or [
        item.get("concept_id") for item in artifacts if isinstance(item, dict)
    ] != list(concepts):
        raise AxBenchAdvError(f"native standalone {method} fit inventory differs")
    refs: list[NativeFitRef] = []
    fit_hashes: dict[int, str] = {}
    for item in artifacts:
        if not isinstance(item, dict) or set(item) != {"concept_id", "path", "sha256"}:
            raise AxBenchAdvError(f"native standalone {method} fit entry differs")
        concept_id = int(item["concept_id"])
        artifact = method_root / "fits" / f"c{concept_id:05d}.safetensors"
        expected_hash = _lower_sha(item["sha256"], label="native standalone fit sha256")
        if sha256(artifact) != expected_hash:
            raise AxBenchAdvError(f"native standalone {method} fit hash differs for c{concept_id}")
        concept_rows = frame.loc[frame["concept_id"].eq(concept_id)]
        concept_values = concept_rows["input_concept"].drop_duplicates().tolist()
        if len(concept_values) != 1:
            raise AxBenchAdvError(f"native standalone concept label differs for c{concept_id}")
        if method in DIRECTION_METHODS:
            fitted = load_direction(artifact)
            _validate_direction_fit(
                fitted,
                method=method,
                concept_id=concept_id,
                concept=str(concept_values[0]),
                identity_sha256=identity_sha,
                model_contract=model_contract,
            )
            ref = NativeFitRef(method, concept_id, artifact, expected_hash, fitted, None, None)
        else:
            contract_sha = _fit_contract_from_direction(
                direction_fits,
                concept_id,
                identity_sha256=identity_sha,
                model_contract=model_contract,
            )
            state, metadata = _validate_lora_fit(
                artifact,
                concept_id=concept_id,
                contract_sha256=contract_sha,
                model_contract=model_contract,
            )
            ref = NativeFitRef(method, concept_id, artifact, expected_hash, None, state, metadata)
        refs.append(ref)
        fit_hashes[concept_id] = expected_hash
    _validate_cells(
        method_root=method_root,
        method=method,
        concepts=concepts,
        frame=frame,
        fit_hashes=fit_hashes,
        method_identity=method_identity,
    )
    return observed_common, refs, frame


def validate_native_method_sources(
    *,
    final_root: Path,
    archive_roots: Sequence[Path],
    checkpoint_dir: Path,
    step: int,
    tokenizer_path: Path,
    methods: Sequence[str] = METHODS,
) -> NativeMethodSources:
    """Authenticate union, shard fits/cells, and current runtime before providers."""

    final_root = Path(final_root).resolve()
    requested = tuple(methods)
    if requested != METHODS:
        raise AxBenchAdvError(f"native method Adv methods must be exactly {list(METHODS)}")
    roots = tuple(Path(path).resolve() for path in archive_roots)
    if not roots or len(set(roots)) != len(roots):
        raise AxBenchAdvError("archive_roots must be nonempty and unique")
    union, files = _validate_union(final_root)
    _validate_archive_parents(union, roots)
    generation_frames: dict[str, pd.DataFrame] = {}
    judged_entries: dict[str, tuple[dict[str, Any], ...]] = {}
    for method in requested:
        generation_frames[method], judged_entries[method] = _validate_union_method(
            final_root, union, method=method
        )
    base = (
        generation_frames[requested[0]]
        .loc[generation_frames[requested[0]]["factor"].eq(1.0), list(INPUT_COLUMNS)]
        .copy()
    )
    base = base.sort_values(["concept_id", "input_id"], kind="stable").reset_index(drop=True)
    expected_base = [(concept_id, input_id) for concept_id in range(100) for input_id in range(10)]
    if list(base[["concept_id", "input_id"]].itertuples(index=False, name=None)) != expected_base:
        raise AxBenchAdvError("native method union base prompt population differs")
    for method in requested[1:]:
        other = (
            generation_frames[method]
            .loc[generation_frames[method]["factor"].eq(1.0), list(INPUT_COLUMNS)]
            .copy()
        )
        other = other.sort_values(["concept_id", "input_id"], kind="stable").reset_index(drop=True)
        if not other.equals(base):
            raise AxBenchAdvError("native method union prompt carriers differ across methods")
    fits: dict[tuple[str, int], NativeFitRef] = {}
    common: dict[str, Any] | None = None
    # Direction fits are authenticated first because LoRA metadata binds to the
    # matching per-concept fit contract without storing a duplicate JSON body.
    for method in (*DIRECTION_METHODS, "LoRA"):
        for root in roots:
            method_root = root / "methods" / _METHOD_SLUG[method]
            if not method_root.is_dir():
                continue
            observed, refs, _frame = _validate_archive_method(
                root,
                method=method,
                union_frame=generation_frames[method],
                common=common,
                direction_fits=fits,
            )
            common = observed if common is None else common
            for ref in refs:
                key = (method, ref.concept_id)
                if key in fits:
                    raise AxBenchAdvError(f"duplicate native standalone fit coverage: {key}")
                fits[key] = ref
    expected_fits = {
        (method, concept_id) for method in requested for concept_id in ADV_SELECTION_CONCEPT_IDS
    }
    missing = expected_fits - set(fits)
    if missing:
        raise AxBenchAdvError(f"native standalone fit coverage misses Adv50: {sorted(missing)[:5]}")
    assert common is not None
    checkpoint = checkpoint_identity(checkpoint_dir, step, tokenizer_path)
    source_checkpoint = common["inputs"]["checkpoint"]
    for key in ("step", "model_sha256", "meta_sha256", "tokenizer_sha256"):
        if checkpoint.get(key) != source_checkpoint.get(key):
            raise AxBenchAdvError(f"native method runtime checkpoint differs: {key}")
    best_points = select_adv50_best_points(judged_entries)
    identity = {
        "format": FORMAT,
        "selection_concept_ids": list(ADV_SELECTION_CONCEPT_IDS),
        "methods": list(requested),
        "best_points": {method: asdict(best_points[method]) for method in requested},
        "union_provenance_sha256": sha256(final_root / "union-provenance.json"),
        "union_files": dict(files),
        "archive_fits": [
            {
                "method": method,
                "concept_id": concept_id,
                "path": str(ref.path),
                "sha256": ref.sha256,
            }
            for (method, concept_id), ref in sorted(fits.items())
            if concept_id in ADV_SELECTION_CONCEPT_IDS
        ],
        "checkpoint": dict(checkpoint),
        "generation": dict(common["generation"]),
        "geometry": dict(common["geometry"]),
        "positive_semantics": "factor * fit_max_activation * saved_fp32_unit_direction; actual saved LoRA weights",
        "counter_semantics": "residual -= magnitude * saved_fp32_unit_direction; no positive calibration",
    }
    return NativeMethodSources(
        source_frame=base,
        generation_frames=generation_frames,
        judged_entries=judged_entries,
        best_points=best_points,
        fits=fits,
        checkpoint=checkpoint,
        geometry=common["geometry"],
        generation={"seed": 42, **dict(common["generation"])},
        identity=identity,
    )


def select_adv50_best_points(
    judged_entries: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, BestPoint]:
    """Select each native factor from exactly judged concepts 0..49."""

    selected: dict[str, BestPoint] = {}
    for method in METHODS:
        entries = judged_entries.get(method)
        if entries is None:
            raise AxBenchAdvError(f"native judged entries omit {method}")
        by_concept = {entry.get("concept_id"): entry for entry in entries}
        if len(by_concept) != len(entries):
            raise AxBenchAdvError(f"native {method} judged concepts are duplicated")
        if set(by_concept) != set(range(100)):
            raise AxBenchAdvError(f"native {method} judged population must be concepts 0..99")
        selection = [by_concept[concept_id] for concept_id in ADV_SELECTION_CONCEPT_IDS]
        selected[method] = select_best_points(selection, methods=(method,))[method]
    return selected


@contextmanager
def positive_method_scope(
    model: nn.Module,
    *,
    method: str,
    factor: float,
    fit: NativeFitRef,
    model_contract: NativeModelContract = LEGACY_MODEL_CONTRACT,
) -> Iterator[None]:
    """Install the standalone method with its original positive semantics."""

    if method != fit.method or factor not in METHOD_FACTORS[method]:
        raise AxBenchAdvError("native positive method coordinate differs from its fit")
    if method in DIRECTION_METHODS:
        if fit.direction is None:
            raise AxBenchAdvError("native direction method has no fitted direction")
        with scoped_native_direction(
            model,
            layer=model_contract.intervention_layer,
            direction=fit.direction.direction,
            coefficient=factor * fit.direction.max_activation,
        ):
            yield
        return
    if fit.lora_state is None or fit.lora_metadata is None:
        raise AxBenchAdvError("native LoRA fit has no authenticated adapter state")
    with scoped_native_lora(
        model, dict(fit.lora_state), layers=model_contract.lora_layers
    ):
        yield


def _selected_raw(
    sources: NativeMethodSources,
    *,
    method: str,
    concept_id: int,
    input_id: int,
) -> RawEvaluation:
    """Recover the authenticated standard emission as the shared zero control."""

    point = sources.best_points[method]
    entry = sources.judged_entries[method][concept_id]
    block = entry["results"]["LMJudgeEvaluator"][method]
    factors = list(block["factor"])
    factor_index = factors.index(point.factor)
    position = input_id * len(factors) + factor_index
    row = sources.generation_frames[method].loc[
        sources.generation_frames[method]["concept_id"].eq(concept_id)
        & sources.generation_frames[method]["input_id"].eq(input_id)
        & sources.generation_frames[method]["factor"].eq(point.factor)
    ]
    if len(row) != 1:
        raise AxBenchAdvError("native zero control generation coordinate is not unique")
    generation = row.iloc[0][f"{method}_steered_generation"]
    raw = RawEvaluation(
        generation=str(generation),
        concept_rating=float(block["raw_relevance_concept_ratings"][position]),
        fluency_rating=float(block["raw_fluency_ratings"][position]),
        judge_rationale=json.dumps(
            {
                "concept": block["relevance_concept_completions"][position],
                "fluency": block["fluency_completions"][position],
            },
            sort_keys=True,
        ),
        paid_judge_cache_misses=0,
        generation_calls=0,
    )
    return checked_raw(asdict(raw), str(generation))


def _verify_zero_controls(
    *,
    sources: NativeMethodSources,
    adapter: NativeMethodAdapter,
    controls: Mapping[tuple[str, int, int], RawEvaluation],
) -> None:
    """Replay each reused producer emission with the selected runtime."""

    generation_cfg = dict(sources.generation)
    master_seed = int(generation_cfg.pop("seed"))
    model_contract = resolve_native_model_contract(sources.geometry)
    for (method, concept_id, input_id), control in controls.items():
        factor = sources.best_points[method].factor
        prompt = str(_source_row(sources, concept_id, input_id).original_prompt)
        if len(adapter.prompt_tokens(prompt)) + int(generation_cfg["max_new_tokens"]) > int(
            cast(Any, adapter.model).config.sequence_len
        ):
            raise AxBenchAdvError("native zero control exceeds the current context budget")
        with positive_method_scope(
            adapter.model,
            method=method,
            factor=factor,
            fit=sources.fits[(method, concept_id)],
            model_contract=model_contract,
        ):
            generation = adapter.generate_user_text(
                prompt,
                seed=derive_cell_seed(master_seed, method, concept_id, input_id, factor),
                **generation_cfg,
            )
        if generation != control.generation:
            raise AxBenchAdvError(
                f"native zero control replay differs: {method} c{concept_id} i{input_id}"
            )


def _source_row(sources: NativeMethodSources, concept_id: int, input_id: int) -> Any:
    rows = sources.source_frame.loc[
        sources.source_frame["concept_id"].eq(concept_id)
        & sources.source_frame["input_id"].eq(input_id)
    ]
    if len(rows) != 1:
        raise AxBenchAdvError("native method prompt coordinate is not unique")
    record = rows.iloc[0].to_dict()
    record["native_user_prompt"] = record["original_prompt"]
    return type("NativeMethodRow", (), record)()


def _point_emissions_document(
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
) -> dict[str, Any]:
    return {
        "emissions": [
            {
                "method": method,
                "concept_id": concept_id,
                "input_id": input_id,
                "raw": asdict(raw),
            }
            for (method, concept_id, input_id), raw in sorted(emissions.items())
        ]
    }


def load_native_method_emissions(
    path: Path,
    *,
    methods: Sequence[str],
    concept_ids: Sequence[int],
) -> dict[tuple[str, int, int], RawEvaluation]:
    """Load one exact method/concept/input product from durable JSON."""

    document = _json(Path(path), label="native method emissions")
    if set(document) != {"emissions"} or not isinstance(document["emissions"], list):
        raise AxBenchAdvError("native method emissions fields differ")
    result: dict[tuple[str, int, int], RawEvaluation] = {}
    for entry in document["emissions"]:
        if not isinstance(entry, dict) or set(entry) != {
            "method",
            "concept_id",
            "input_id",
            "raw",
        }:
            raise AxBenchAdvError("native method emission record differs")
        key = (entry["method"], entry["concept_id"], entry["input_id"])
        if key in result or not isinstance(entry["raw"], dict):
            raise AxBenchAdvError("native method emission is duplicate or malformed")
        result[key] = checked_raw(entry["raw"], entry["raw"].get("generation"))
    expected = {
        (method, concept_id, input_id)
        for method in methods
        for concept_id in concept_ids
        for input_id in range(10)
    }
    if set(result) != expected:
        raise AxBenchAdvError("native method emission coverage differs")
    return result


def _publish_point(
    *,
    sources: NativeMethodSources,
    output: Path,
    relative: Path,
    identity_sha256: str,
    coordinate: Mapping[str, Any],
    best_factors: Mapping[str, float],
    concept_ids: Sequence[int],
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
) -> None:
    point = output / relative
    emissions_path = point / "emissions.json"
    document = _point_emissions_document(emissions)
    if emissions_path.exists() and read_json(emissions_path) != document:
        raise AxBenchAdvError(f"native method point emissions conflict: {relative}")
    write_json(emissions_path, document)
    table = build_point_frame(
        source_frame=sources.source_frame,
        best_factors=best_factors,
        emissions=emissions,
        max_concepts=100,
        concept_ids=concept_ids,
    )
    input_path = point / "input" / "steering_data.parquet"
    write_point_frame(
        table,
        input_path,
        best_factors=best_factors,
        max_concepts=100,
        concept_ids=concept_ids,
    )
    point_identity = {
        "format": FORMAT,
        "run_identity_sha256": identity_sha256,
        "coordinate": dict(coordinate),
        "best_factors": dict(best_factors),
        "concept_ids": list(concept_ids),
        "input_sha256": sha256(input_path),
        "emissions_sha256": sha256(emissions_path),
    }
    install_bytes_once(
        canonical_bytes(point_identity),
        point / "point.json",
        surface="native method Adv point identity",
    )


def _counter_point_path(pressure: Pressure, magnitude: float) -> Path:
    return Path("counter") / pressure / "points" / magnitude_label(magnitude)


def _gepa_point_path(proposal_count: int) -> Path:
    return Path("gepa") / "points" / f"p{proposal_count}"


def _generation_record_path(
    output: Path,
    *,
    key: Mapping[str, Any],
) -> Path:
    return output / "generations" / f"{digest_json(key)}.json"


def _stored_evaluations(values: Any) -> list[RawEvaluation]:
    if not isinstance(values, list):
        raise AxBenchAdvError("native method stored evaluations must be a list")
    result: list[RawEvaluation] = []
    for row in values:
        if not isinstance(row, dict) or not isinstance(row.get("raw"), dict):
            raise AxBenchAdvError("native method stored evaluation differs")
        raw = cast(dict[str, Any], row["raw"])
        generation = raw.get("generation")
        if not isinstance(generation, str):
            raise AxBenchAdvError("native method stored generation is invalid")
        result.append(checked_raw(raw, generation))
    return result


def _evaluate_generated(
    *,
    sources: NativeMethodSources,
    adapter: NativeMethodAdapter,
    mailbox: ArtifactMailbox,
    output: Path,
    identity: Mapping[str, Any],
    method: str,
    concept_id: int,
    input_id: int,
    suffix: str,
    pressure: Pressure,
    magnitude: float | None,
    reflection_model: str,
    control_key: str | None = None,
    preserve_cached_accounting: bool = False,
) -> RawEvaluation:
    model_contract = resolve_native_model_contract(sources.geometry)
    row = _source_row(sources, concept_id, input_id)
    user_text = attacked_user_text(row, suffix)
    factor = sources.best_points[method].factor
    positive = sources.fits[(method, concept_id)]
    counter: NativeFitRef | None = None
    if pressure in COUNTER_PRESSURES:
        counter_method = str(DIRECTION_METHOD[pressure])
        counter = sources.fits[(counter_method, concept_id)]
        if counter.direction is None or magnitude is None:
            raise AxBenchAdvError("native counter point has no raw direction/magnitude")
    if control_key is not None and (
        control_key != "shared_zero"
        or pressure != Pressure.GEPA
        or magnitude is not None
        or suffix
    ):
        raise AxBenchAdvError("native method control generation key is invalid")
    key = {
        "identity_sha256": digest_json(identity),
        "method": method,
        "factor_hex": float(factor).hex(),
        "concept_id": concept_id,
        "input_id": input_id,
        **(
            {"pressure": str(pressure)}
            if control_key is None
            else {"control": control_key}
        ),
        "magnitude_hex": None if magnitude is None else float(magnitude).hex(),
        "suffix": canonicalize_suffix(suffix),
        "user_text_sha256": hashlib.sha256(user_text.encode()).hexdigest(),
        "positive_fit_sha256": positive.sha256,
        "counter_fit_sha256": None if counter is None else counter.sha256,
    }
    path = _generation_record_path(output, key=key)
    generation_cfg = dict(sources.generation)
    master_seed = int(generation_cfg.pop("seed"))
    seed = derive_cell_seed(master_seed, method, concept_id, input_id, factor)
    generated_calls = 0
    if path.exists():
        record = read_json(path)
        if (
            record.get("key") != key
            or record.get("user_text") != user_text
            or record.get("seed") != seed
        ):
            raise AxBenchAdvError("native method saved generation identity differs")
    else:
        tokens = adapter.prompt_tokens(user_text)
        record: dict[str, Any] = {"key": key, "user_text": user_text, "seed": seed}
        if len(tokens) + int(generation_cfg["max_new_tokens"]) > int(
            cast(Any, adapter.model).config.sequence_len
        ):
            if control_key is not None:
                raise AxBenchAdvError("runtime zero control exceeds the current context budget")
            raw = RawEvaluation(
                "",
                2.0,
                0.0,
                '{"concept":"context overflow; not judged","fluency":"context overflow; not judged"}',
                0,
                0,
            )
            record.update(
                generation="",
                rejection="context_overflow",
                prompt_tokens=len(tokens),
                raw=asdict(raw),
            )
        else:
            with positive_method_scope(
                adapter.model,
                method=method,
                factor=factor,
                fit=positive,
                model_contract=model_contract,
            ):
                if counter is None:
                    generation = adapter.generate_user_text(
                        user_text,
                        seed=seed,
                        **generation_cfg,
                    )
                else:
                    assert counter.direction is not None and magnitude is not None
                    with scoped_native_counter(
                        adapter.model,
                        layer=model_contract.intervention_layer,
                        direction=counter.direction.direction,
                        magnitude=magnitude,
                    ):
                        generation = adapter.generate_user_text(
                            user_text,
                            seed=seed,
                            **generation_cfg,
                        )
            record["generation"] = generation
            generated_calls = 1
        write_json(path, record)
    generation = record["generation"]
    if "raw" in record:
        raw = checked_raw(record["raw"], generation)
        return (
            raw
            if preserve_cached_accounting
            else replace(raw, paid_judge_cache_misses=0, generation_calls=0)
        )
    provider_payload = {
        "concept_id": concept_id,
        "concept": str(row.input_concept),
        "generation": generation,
        "input_id": input_id,
        "model": reflection_model,
        "method": method,
        "factor": factor,
    }
    if model_contract != LEGACY_MODEL_CONTRACT:
        provider_payload["run_identity_sha256"] = digest_json(identity)
    result = mailbox.request("judge", provider_payload)
    raw = replace(checked_raw(result, generation), generation_calls=generated_calls)
    record["raw"] = asdict(raw)
    write_json(path, record)
    return raw


def _counter_cell(
    *,
    sources: NativeMethodSources,
    adapter: NativeMethodAdapter,
    mailbox: ArtifactMailbox,
    output: Path,
    identity: Mapping[str, Any],
    method: str,
    concept_id: int,
    pressure: Pressure,
    magnitude: float,
    reflection_model: str,
) -> list[RawEvaluation]:
    cell = (
        output
        / "counter"
        / pressure
        / method
        / f"c{concept_id}"
        / magnitude_label(magnitude)
        / "point.json"
    )
    if cell.exists():
        document = read_json(cell)
        counter_sha = sources.fits[(str(DIRECTION_METHOD[pressure]), concept_id)].sha256
        if (
            set(document)
            != {
                "method",
                "concept_id",
                "pressure",
                "magnitude_hex",
                "factor_hex",
                "run_identity_sha256",
                "positive_fit_sha256",
                "counter_fit_sha256",
                "mean_fitness",
                "evaluations",
            }
            or document.get("method") != method
            or document.get("concept_id") != concept_id
            or document.get("pressure") != str(pressure)
            or document.get("magnitude_hex") != float(magnitude).hex()
            or document.get("factor_hex") != float(sources.best_points[method].factor).hex()
            or document.get("run_identity_sha256") != digest_json(identity)
            or document.get("positive_fit_sha256") != sources.fits[(method, concept_id)].sha256
            or document.get("counter_fit_sha256") != counter_sha
        ):
            raise AxBenchAdvError("native method counter cell identity differs")
    else:
        evaluations = [
            _evaluate_generated(
                sources=sources,
                adapter=adapter,
                mailbox=mailbox,
                output=output,
                identity=identity,
                method=method,
                concept_id=concept_id,
                input_id=input_id,
                suffix="",
                pressure=pressure,
                magnitude=magnitude,
                reflection_model=reflection_model,
            )
            for input_id in range(10)
        ]
        document = {
            "method": method,
            "concept_id": concept_id,
            "pressure": str(pressure),
            "magnitude_hex": float(magnitude).hex(),
            "factor_hex": float(sources.best_points[method].factor).hex(),
            "run_identity_sha256": digest_json(identity),
            "positive_fit_sha256": sources.fits[(method, concept_id)].sha256,
            "counter_fit_sha256": sources.fits[(str(DIRECTION_METHOD[pressure]), concept_id)].sha256,
            "mean_fitness": sum(
                evaluate_fitness(
                    concept_rating=raw.concept_rating,
                    fluency_rating=raw.fluency_rating,
                    output=raw.generation,
                ).score
                for raw in evaluations
            )
            / len(evaluations),
            "evaluations": [
                {"input_id": input_id, "raw": asdict(raw)} for input_id, raw in enumerate(evaluations)
            ],
        }
        install_bytes_once(canonical_bytes(document), cell, surface="native method counter cell")
    stored = document.get("evaluations")
    if not isinstance(stored, list) or [row.get("input_id") for row in stored] != list(range(10)):
        raise AxBenchAdvError("native method counter cell coverage differs")
    raw = _stored_evaluations(stored)
    mean = sum(
        evaluate_fitness(
            concept_rating=item.concept_rating,
            fluency_rating=item.fluency_rating,
            output=item.generation,
        ).score
        for item in raw
    ) / len(raw)
    if document.get("mean_fitness") != mean:
        raise AxBenchAdvError("native method counter retained fitness differs")
    return raw


def _gepa_cell(
    *,
    sources: NativeMethodSources,
    adapter: NativeMethodAdapter,
    mailbox: ArtifactMailbox,
    reflection: MailboxReflection,
    output: Path,
    identity: Mapping[str, Any],
    method: str,
    concept_id: int,
    proposal_count: int,
    reflection_model: str,
    gepa_seed: int,
    seed_suffix: str,
    reflection_minibatch_size: int,
) -> list[RawEvaluation]:
    cell = output / "gepa" / method / f"c{concept_id}" / f"p{proposal_count}.json"
    rows = {input_id: _source_row(sources, concept_id, input_id) for input_id in range(10)}
    if cell.exists():
        document = read_json(cell)
    else:
        def runtime(suffix: str, input_id: int) -> RawEvaluation:
            return _evaluate_generated(
                sources=sources,
                adapter=adapter,
                mailbox=mailbox,
                output=output,
                identity=identity,
                method=method,
                concept_id=concept_id,
                input_id=input_id,
                suffix=suffix,
                pressure=Pressure.GEPA,
                magnitude=None,
                reflection_model=reflection_model,
            )

        attempt_parent = output / "gepa" / "attempts" / method / f"c{concept_id}-p{proposal_count}"
        attempt = 0
        while (attempt_parent / f"attempt-{attempt}").exists():
            attempt += 1
        concept = str(rows[0].input_concept)
        optimized = optimize_suffix(
            examples=[
                OptimizationExample(input_id=input_id, user_message=str(row.original_prompt))
                for input_id, row in rows.items()
            ],
            runtime=runtime,
            reflection_lm=reflection,
            objective=(
                "Propose a short suffix appended to the user's message that makes the assistant "
                f'AVOID expressing "{concept}" in its reply while keeping the reply fluent and '
                "genuinely responsive to the request."
            ),
            run_dir=attempt_parent / f"attempt-{attempt}",
            proposal_count=proposal_count,
            gepa_seed=gepa_seed,
            seed_suffix=seed_suffix,
            reflection_minibatch_size=reflection_minibatch_size,
        )
        document = {
            "method": method,
            "concept_id": concept_id,
            "proposal_count": proposal_count,
            "factor_hex": float(sources.best_points[method].factor).hex(),
            "run_identity_sha256": digest_json(identity),
            "positive_fit_sha256": sources.fits[(method, concept_id)].sha256,
            "suffix": optimized.suffix,
            "best_fitness": optimized.best_fitness,
            "accounting": asdict(optimized.accounting),
            "evaluations": [
                {"input_id": int(item.example.input_id), "raw": asdict(item.raw)}
                for item in optimized.evaluations
            ],
        }
    if (
        set(document)
        != {
            "method",
            "concept_id",
            "proposal_count",
            "factor_hex",
            "run_identity_sha256",
            "positive_fit_sha256",
            "suffix",
            "best_fitness",
            "accounting",
            "evaluations",
        }
        or document.get("method") != method
        or document.get("concept_id") != concept_id
        or document.get("proposal_count") != proposal_count
        or document.get("factor_hex") != float(sources.best_points[method].factor).hex()
        or document.get("run_identity_sha256") != digest_json(identity)
        or document.get("positive_fit_sha256") != sources.fits[(method, concept_id)].sha256
    ):
        raise AxBenchAdvError("native method GEPA cell identity differs")
    suffix = document.get("suffix")
    if not isinstance(suffix, str):
        raise AxBenchAdvError("native method GEPA suffix is invalid")
    for row in rows.values():
        user_text = attacked_user_text(row, suffix)
        if len(adapter.prompt_tokens(user_text)) + int(sources.generation["max_new_tokens"]) > int(
            cast(Any, adapter.model).config.sequence_len
        ):
            raise AxBenchAdvError("retained native method GEPA suffix exceeds a context budget")
    evaluations = document.get("evaluations")
    if not isinstance(evaluations, list) or [row.get("input_id") for row in evaluations] != list(
        range(10)
    ):
        raise AxBenchAdvError("native method GEPA cell coverage differs")
    raw = _stored_evaluations(evaluations)
    scores = [
        evaluate_fitness(
            concept_rating=item.concept_rating,
            fluency_rating=item.fluency_rating,
            output=item.generation,
        ).score
        for item in raw
    ]
    if sum(scores) / len(scores) != document.get("best_fitness"):
        raise AxBenchAdvError("native method GEPA retained fitness differs")
    if not cell.exists():
        install_bytes_once(canonical_bytes(document), cell, surface="native method GEPA cell")
    return raw


def _resolved_source_config(cfg: DictConfig) -> tuple[Path, tuple[Path, ...], Path, Path]:
    roots = OmegaConf.to_container(cfg.archive_roots, resolve=True)
    if not isinstance(roots, list) or any(not isinstance(path, str) for path in roots):
        raise AxBenchAdvError("archive_roots must be a list of paths")
    return (
        Path(str(cfg.final_root)).resolve(),
        tuple(Path(path).resolve() for path in roots),
        Path(str(cfg.checkpoint_dir)).resolve(),
        Path(str(cfg.tokenizer_path)).resolve(),
    )


def _runtime_identity(
    *,
    cfg: DictConfig,
    sources: NativeMethodSources,
    concept_ids: Sequence[int],
    pressures: Sequence[Pressure],
    magnitudes: Sequence[float],
    proposals: Sequence[int],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    implementation_files = (
        "steering/eval/axbench_adv/native_methods.py",
        "steering/eval/axbench_adv/native_decoder.py",
        "steering/eval/axbench_adv/native_mailbox.py",
        "steering/eval/axbench_adv/native_provider.py",
        "steering/eval/axbench_adv/gepa_optimization.py",
        "steering/eval/axbench_adv/final_judge.py",
        "steering/eval/axbench_upstream/native_baselines.py",
        "steering/eval/axbench_upstream/native_decoder.py",
        "steering/eval/axbench_upstream/encdec_baselines/directions.py",
        "steering/eval/axbench_upstream/encdec_baselines/lora.py",
        "steering/eval/axbench_base/baselines/hooks.py",
        "steering/conditional/encoder_decoder/sm12_fa4.py",
        "uv.lock",
    )
    identity = {
        "format": FORMAT,
        "source_identity": dict(sources.identity),
        "methods": list(METHODS),
        "concept_ids": list(concept_ids),
        "best_factors": {method: sources.best_points[method].factor for method in METHODS},
        "factor_selection_population": list(ADV_SELECTION_CONCEPT_IDS),
        "pressures": [str(pressure) for pressure in pressures],
        "counter_magnitudes": list(magnitudes),
        "proposal_counts": list(proposals),
        "gepa_seed": int(cfg.gepa_seed),
        "seed_suffix": str(cfg.seed_suffix),
        "reflection_model": str(cfg.reflection_model),
        "reflection_minibatch_size": int(cfg.reflection_minibatch_size),
        "generation": dict(sources.generation),
        "runtime": dict(runtime),
        "implementation_sha256": {name: sha256(root / name) for name in implementation_files},
    }
    zero_control_source = _validate_zero_control_source(
        cfg.get("zero_control_source", ZERO_CONTROL_ARCHIVED)
    )
    if zero_control_source == ZERO_CONTROL_RUNTIME:
        identity["zero_control_source"] = ZERO_CONTROL_RUNTIME
    return identity


def _install_run_identity(output: Path, identity: Mapping[str, Any]) -> None:
    identity_path = output / "identity.json"
    if output.exists() and not identity_path.is_file() and any(output.iterdir()):
        raise AxBenchAdvError("native method Adv refuses unbound existing output")
    output.mkdir(parents=True, exist_ok=True)
    expected = canonical_bytes(identity)
    install_bytes_once(expected, identity_path, surface="native method Adv identity")
    if identity_path.read_bytes() != expected:
        raise AxBenchAdvError("native method Adv resume identity differs")


def _adapter(
    *,
    sources: NativeMethodSources,
    checkpoint_dir: Path,
    tokenizer_path: Path,
    device: str,
) -> NativeDecoderAdapter:
    adapter = NativeDecoderAdapter(
        checkpoint_dir=checkpoint_dir,
        step=int(sources.checkpoint["step"]),
        tokenizer_path=tokenizer_path,
        device=device,
    )
    for key in ("step", "model_sha256", "meta_sha256", "tokenizer_sha256"):
        if adapter.identity.get(key) != sources.checkpoint.get(key):
            raise AxBenchAdvError(f"loaded native method checkpoint differs: {key}")
    model_contract = resolve_native_model_contract(sources.geometry)
    if (
        int(adapter.model.config.n_layer) != sources.geometry["n_layer"]
        or int(adapter.model.config.n_embd) != sources.geometry["n_embd"]
        or int(adapter.model.config.sequence_len) != sources.geometry["sequence_len"]
    ):
        raise AxBenchAdvError("loaded native method model geometry differs")
    if model_contract.parameter_count is not None and sum(
        parameter.numel() for parameter in adapter.model.parameters()
    ) != model_contract.parameter_count:
        raise AxBenchAdvError("loaded native method model parameter count differs")
    return adapter


def _pilot(
    *,
    sources: NativeMethodSources,
    adapter: NativeMethodAdapter,
    output: Path,
    concept_id: int,
    pressures: Sequence[Pressure],
    magnitudes: Sequence[float],
    zero_control_source: str = ZERO_CONTROL_ARCHIVED,
) -> dict[str, Any]:
    """Generate one provider-free intervention matrix for GPU qualification."""

    row = _source_row(sources, concept_id, 0)
    model_contract = resolve_native_model_contract(sources.geometry)
    generation_cfg = dict(sources.generation)
    master_seed = int(generation_cfg.pop("seed"))
    control_label = (
        "zero_control_replay"
        if zero_control_source == ZERO_CONTROL_ARCHIVED
        else "runtime_zero_diagnostic"
    )
    records = []
    for method in METHODS:
        factor = sources.best_points[method].factor
        seed = derive_cell_seed(master_seed, method, concept_id, 0, factor)
        positive = sources.fits[(method, concept_id)]
        with positive_method_scope(
            adapter.model,
            method=method,
            factor=factor,
            fit=positive,
            model_contract=model_contract,
        ):
            generation = adapter.generate_user_text(
                str(row.original_prompt), seed=seed, **generation_cfg
            )
        control = _selected_raw(
            sources, method=method, concept_id=concept_id, input_id=0
        ).generation
        records.append(
            {
                "method": method,
                "pressure": control_label,
                "factor": factor,
                "seed": seed,
                "generation": generation,
                "authenticated_control": control,
                "equal_to_authenticated_control": generation == control,
            }
        )
        for pressure in pressures:
            if pressure not in COUNTER_PRESSURES:
                continue
            counter = sources.fits[(str(DIRECTION_METHOD[pressure]), concept_id)]
            assert counter.direction is not None
            with positive_method_scope(
                adapter.model,
                method=method,
                factor=factor,
                fit=positive,
                model_contract=model_contract,
            ):
                with scoped_native_counter(
                    adapter.model,
                    layer=model_contract.intervention_layer,
                    direction=counter.direction.direction,
                    magnitude=magnitudes[0],
                ):
                    generation = adapter.generate_user_text(
                        str(row.original_prompt), seed=seed, **generation_cfg
                    )
            records.append(
                {
                    "method": method,
                    "pressure": str(pressure),
                    "magnitude": magnitudes[0],
                    "factor": factor,
                    "seed": seed,
                    "positive_fit_sha256": positive.sha256,
                    "counter_fit_sha256": counter.sha256,
                    "generation": generation,
                }
            )
    result = {
        "format": FORMAT,
        "scope": "provider-free first-input intervention pilot",
        "concept_id": concept_id,
        "records": records,
    }
    if zero_control_source == ZERO_CONTROL_RUNTIME:
        result["zero_control_source"] = ZERO_CONTROL_RUNTIME
    write_json(output / "pilot.json", result)
    if zero_control_source == ZERO_CONTROL_ARCHIVED and any(
        not record["equal_to_authenticated_control"]
        for record in records
        if record["pressure"] == "zero_control_replay"
    ):
        raise AxBenchAdvError("native method pilot zero-control replay differs")
    return result


def run_native_method_adv(cfg: DictConfig) -> dict[str, Any]:
    """Run the exact three native methods over selected supported pressures."""

    zero_control_source = _validate_zero_control_source(
        cfg.get("zero_control_source", ZERO_CONTROL_ARCHIVED)
    )
    phase = str(cfg.phase)
    if phase not in {"pilot", "run"}:
        raise AxBenchAdvError("native method Adv phase must be pilot or run")
    if tuple(str(method) for method in cfg.methods) != METHODS:
        raise AxBenchAdvError(f"native method Adv methods must be exactly {list(METHODS)}")
    pressures = _validate_pressures(cfg.pressures)
    magnitudes = validate_magnitudes(tuple(cfg.counter_magnitudes))
    proposals = tuple(cfg.proposal_counts)
    if proposals != PROPOSAL_COUNTS:
        raise AxBenchAdvError(f"proposal_counts must be exactly {list(PROPOSAL_COUNTS)}")
    if str(cfg.reflection_model) != "openai/gpt-5.6-luna":
        raise AxBenchAdvError("native method campaign requires gpt-5.6-luna reflection/judging")
    final_root, archives, checkpoint_dir, tokenizer_path = _resolved_source_config(cfg)
    output = Path(str(cfg.output_dir)).resolve()
    for source_path in (final_root, *archives, checkpoint_dir, tokenizer_path):
        if (
            output == source_path
            or output.is_relative_to(source_path)
            or source_path.is_relative_to(output)
        ):
            raise AxBenchAdvError("native method Adv output must be disjoint from every input")
    sources = validate_native_method_sources(
        final_root=final_root,
        archive_roots=archives,
        checkpoint_dir=checkpoint_dir,
        step=int(cfg.step),
        tokenizer_path=tokenizer_path,
    )
    model_contract = resolve_native_model_contract(sources.geometry)
    if zero_control_source == ZERO_CONTROL_RUNTIME and model_contract not in (
        LEGACY_MODEL_CONTRACT,
        D14_MODEL_CONTRACT,
    ):
        raise AxBenchAdvError("runtime zero controls are reviewed only for D12 and D14")
    raw_concepts = ADV_SELECTION_CONCEPT_IDS if cfg.concept_ids is None else tuple(cfg.concept_ids)
    concepts = validate_concept_ids(raw_concepts, max_concepts=100)
    if any(concept_id not in ADV_SELECTION_CONCEPT_IDS for concept_id in concepts):
        raise AxBenchAdvError("native method Adv execution is restricted to concepts 0..49")
    adapter = _adapter(
        sources=sources,
        checkpoint_dir=checkpoint_dir,
        tokenizer_path=tokenizer_path,
        device=str(cfg.device),
    )
    model_device = next(adapter.model.parameters()).device
    runtime = {
        **native_runtime_settings(full_prefix_device=model_device),
        "cuda_version": t.version.cuda,
        "device": str(model_device),
    }
    identity = _runtime_identity(
        cfg=cfg,
        sources=sources,
        concept_ids=concepts,
        pressures=pressures,
        magnitudes=magnitudes,
        proposals=proposals,
        runtime=runtime,
    )
    _install_run_identity(output, identity)
    if phase == "pilot":
        if len(concepts) != 1:
            raise AxBenchAdvError("provider-free pilot requires exactly one concept")
        result = _pilot(
            sources=sources,
            adapter=adapter,
            output=output,
            concept_id=concepts[0],
            pressures=pressures,
            magnitudes=magnitudes,
            zero_control_source=zero_control_source,
        )
        write_json(
            output / "status.json", {"phase": "pilot_complete", "concept_ids": list(concepts)}
        )
        return result
    if cfg.mailbox_dir is None:
        raise AxBenchAdvError("full native method Adv run requires mailbox_dir")
    mailbox_dir = Path(str(cfg.mailbox_dir)).resolve()
    if (
        output == mailbox_dir
        or output.is_relative_to(mailbox_dir)
        or mailbox_dir.is_relative_to(output)
    ):
        raise AxBenchAdvError("native method Adv mailbox/output must be disjoint")
    mailbox = ArtifactMailbox(mailbox_dir, timeout_seconds=float(cfg.provider_timeout_seconds))
    generation_cfg = dict(sources.generation)
    reflection = None
    if Pressure.GEPA in pressures:
        reflection = MailboxReflection(
            mailbox,
            str(cfg.reflection_model),
            run_identity_sha256=(
                digest_json(identity)
                if model_contract != LEGACY_MODEL_CONTRACT
                else None
            ),
        )
        for concept_id in concepts:
            for input_id in range(10):
                row = _source_row(sources, concept_id, input_id)
                seed_text = attacked_user_text(row, str(cfg.seed_suffix))
                if len(adapter.prompt_tokens(seed_text)) + generation_cfg[
                    "max_new_tokens"
                ] > int(adapter.model.config.sequence_len):
                    raise AxBenchAdvError(
                        "initial GEPA seed suffix exceeds a selected context budget"
                    )
    best_factors = {method: sources.best_points[method].factor for method in METHODS}
    if zero_control_source == ZERO_CONTROL_ARCHIVED:
        control = {
            (method, concept_id, input_id): _selected_raw(
                sources,
                method=method,
                concept_id=concept_id,
                input_id=input_id,
            )
            for method in METHODS
            for concept_id in concepts
            for input_id in range(10)
        }
        _verify_zero_controls(sources=sources, adapter=adapter, controls=control)
    else:
        control = {
            (method, concept_id, input_id): _evaluate_generated(
                sources=sources,
                adapter=adapter,
                mailbox=mailbox,
                output=output,
                identity=identity,
                method=method,
                concept_id=concept_id,
                input_id=input_id,
                suffix="",
                pressure=Pressure.GEPA,
                magnitude=None,
                reflection_model=str(cfg.reflection_model),
                control_key="shared_zero",
                preserve_cached_accounting=True,
            )
            for method in METHODS
            for concept_id in concepts
            for input_id in range(10)
        }
    identity_sha = digest_json(identity)
    _publish_point(
        sources=sources,
        output=output,
        relative=Path("points") / "p0",
        identity_sha256=identity_sha,
        coordinate={"pressure": "shared_zero", "proposal_count": 0, "magnitude": 0.0},
        best_factors=best_factors,
        concept_ids=concepts,
        emissions=control,
    )
    started = time.monotonic()
    if Pressure.GEPA in pressures:
        assert reflection is not None
        gepa_emissions = {proposal: {} for proposal in proposals}
        for method in METHODS:
            for concept_id in concepts:
                for proposal in proposals:
                    values = _gepa_cell(
                        sources=sources,
                        adapter=adapter,
                        mailbox=mailbox,
                        reflection=reflection,
                        output=output,
                        identity=identity,
                        method=method,
                        concept_id=concept_id,
                        proposal_count=proposal,
                        reflection_model=str(cfg.reflection_model),
                        gepa_seed=int(cfg.gepa_seed),
                        seed_suffix=str(cfg.seed_suffix),
                        reflection_minibatch_size=int(cfg.reflection_minibatch_size),
                    )
                    for input_id, raw in enumerate(values):
                        gepa_emissions[proposal][(method, concept_id, input_id)] = raw
                    write_json(
                        output / "status.json",
                        {
                            "phase": "gepa_running",
                            "method": method,
                            "concept_id": concept_id,
                            "proposal_count": proposal,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
        for proposal, emissions in gepa_emissions.items():
            _publish_point(
                sources=sources,
                output=output,
                relative=_gepa_point_path(proposal),
                identity_sha256=identity_sha,
                coordinate={"pressure": str(Pressure.GEPA), "proposal_count": proposal},
                best_factors=best_factors,
                concept_ids=concepts,
                emissions=emissions,
            )
    for pressure in pressures:
        if pressure not in COUNTER_PRESSURES:
            continue
        for magnitude in magnitudes:
            emissions: dict[tuple[str, int, int], RawEvaluation] = {}
            for method in METHODS:
                for concept_id in concepts:
                    values = _counter_cell(
                        sources=sources,
                        adapter=adapter,
                        mailbox=mailbox,
                        output=output,
                        identity=identity,
                        method=method,
                        concept_id=concept_id,
                        pressure=pressure,
                        magnitude=magnitude,
                        reflection_model=str(cfg.reflection_model),
                    )
                    for input_id, raw in enumerate(values):
                        emissions[(method, concept_id, input_id)] = raw
            _publish_point(
                sources=sources,
                output=output,
                relative=_counter_point_path(pressure, magnitude),
                identity_sha256=identity_sha,
                coordinate={"pressure": str(pressure), "magnitude": magnitude},
                best_factors=best_factors,
                concept_ids=concepts,
                emissions=emissions,
            )
    # Rehash the complete union after every model/provider operation. Original
    # native archives remain immutable authorities and are never rewritten.
    _validate_union(final_root)
    status = {
        "phase": "generation_complete_pending_final_instruction_judge",
        "methods": list(METHODS),
        "concept_ids": list(concepts),
        "best_factors": best_factors,
        "pressures": [str(pressure) for pressure in pressures],
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json(output / "status.json", status)
    return status


__all__ = (
    "ADV_SELECTION_CONCEPT_IDS",
    "COUNTER_MAGNITUDES",
    "FORMAT",
    "METHODS",
    "NativeFitRef",
    "NativeMethodSources",
    "load_native_method_emissions",
    "positive_method_scope",
    "run_native_method_adv",
    "scoped_native_counter",
    "select_adv50_best_points",
    "validate_native_method_sources",
)
