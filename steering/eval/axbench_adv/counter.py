"""Counter artifact preflight using BPB-Adv's direction identity contract."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch as t

from scripts.eval.axbench_adv.counter_hooks import load_vendor_direction, validate_direction
from scripts.eval.axbench_adv.pressures import (
    COEFFICIENT_SEMANTICS,
    DIRECTION_FAMILY,
    DIRECTION_METHOD,
    Pressure,
)
from steering._common.strict_json import (
    string_keyed_mapping,
)
from steering.eval.axbench_adv.direction_fitting import (
    CAPTURE_SITE,
    FIT_MANIFEST_NAME,
    FIT_MANIFEST_SCHEMA_VERSION,
    fit_recipe_identity,
    require_counter_calibration,
    require_fit_direction_normalization,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.fitness import evaluate_fitness
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.types import ROOT_METHODS
from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactRequest,
    LoadedDirectionArtifact,
    load_direction_artifact,
)
from steering.eval.axbench_base.direction_scale import require_direction_max_activation
from steering.immutable_io import is_sha256_hex

_FIT_FAMILIES = frozenset({"diffmean", "linear_probe"})


@dataclass(frozen=True)
class ProjectCounterExpectations:
    """Scientific coordinates required to apply project counter directions."""

    target_model_sha256: str
    target_metadata_sha256: str
    concept_texts: Mapping[int, str]
    decoder_width: int
    layer: int
    fit_seed: int
    site: str = CAPTURE_SITE

    def __post_init__(self) -> None:
        for field_name in ("target_model_sha256", "target_metadata_sha256"):
            value = getattr(self, field_name)
            if not is_sha256_hex(value):
                raise AxBenchAdvError(f"project counter {field_name} must be lowercase SHA-256")
        if type(self.decoder_width) is not int or self.decoder_width <= 0:
            raise AxBenchAdvError("project counter decoder_width must be positive")
        if type(self.layer) is not int or self.layer < 0:
            raise AxBenchAdvError("project counter layer must be nonnegative")
        if type(self.fit_seed) is not int or self.fit_seed < 0:
            raise AxBenchAdvError("project counter fit_seed must be nonnegative")
        if self.site != CAPTURE_SITE:
            raise AxBenchAdvError(f"project counter site must be exactly {CAPTURE_SITE!r}")
        concepts = dict(self.concept_texts)
        if not concepts or any(
            type(concept_id) is not int
            or concept_id < 0
            or not isinstance(concept, str)
            or not concept
            for concept_id, concept in concepts.items()
        ):
            raise AxBenchAdvError(
                "project counter concept_texts must map nonnegative IDs to nonempty text"
            )


@dataclass(frozen=True)
class ProjectFitArtifact:
    """One validated project direction and its exact result digest."""

    loaded: LoadedDirectionArtifact
    sha256: str


@dataclass(frozen=True)
class ProjectDirectionFitPreflight:
    """Complete scientifically compatible project direction inventory."""

    root: Path
    manifest_sha256: str
    artifacts: Mapping[tuple[str, int], ProjectFitArtifact]


def _file_identity_sha(value: Mapping[str, Any], *, surface: str) -> str:
    if not is_sha256_hex(value.get("sha256")):
        raise AxBenchAdvError(f"{surface} file identity is malformed")
    return value["sha256"]


def preflight_project_direction_fit(
    root: Path,
    expectations: ProjectCounterExpectations,
) -> ProjectDirectionFitPreflight:
    """Validate the scientific coordinates needed to apply a project fit."""

    try:
        root = Path(root).expanduser().resolve(strict=True)
        manifest_path = root / FIT_MANIFEST_NAME
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchAdvError(f"invalid project fit manifest under {root}") from exc
    if not root.is_dir() or not isinstance(manifest, Mapping):
        raise AxBenchAdvError(f"invalid project fit manifest under {root}")
    if (
        type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != FIT_MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "complete"
    ):
        raise AxBenchAdvError("project fit manifest is not a completed supported schema")
    shared = string_keyed_mapping(manifest.get("shared_identity"))
    checkpoint = string_keyed_mapping(shared.get("checkpoint")) if shared else None
    target_model = string_keyed_mapping(checkpoint.get("target_model")) if checkpoint else None
    companion = string_keyed_mapping(checkpoint.get("companion_metadata")) if checkpoint else None
    fit = string_keyed_mapping(shared.get("fit")) if shared else None
    if target_model is None or companion is None or fit is None:
        raise AxBenchAdvError("project fit scientific identity is malformed")
    expected_fit = fit_recipe_identity(
        layer=expectations.layer,
        decoder_width=expectations.decoder_width,
        seed=expectations.fit_seed,
    )
    mismatches = {
        "target_model_sha256": (
            expectations.target_model_sha256,
            _file_identity_sha(target_model, surface="fit target model"),
        ),
        "target_metadata_sha256": (
            expectations.target_metadata_sha256,
            _file_identity_sha(companion, surface="fit target metadata"),
        ),
        "fit_recipe": (expected_fit, dict(fit)),
    }
    different = [
        f"{name}: expected={expected!r}, fit={actual!r}"
        for name, (expected, actual) in mismatches.items()
        if expected != actual
    ]
    if different:
        raise AxBenchAdvError(
            "project fit identity does not match the validated target: " + "; ".join(different)
        )

    inventory = manifest.get("artifacts")
    if not isinstance(inventory, list):
        raise AxBenchAdvError("project fit manifest artifacts must be a list")
    expected_coordinates = {
        (family, concept_id)
        for family in _FIT_FAMILIES
        for concept_id in expectations.concept_texts
    }
    retained: dict[tuple[str, int], ProjectFitArtifact] = {}
    for index, raw_record in enumerate(inventory):
        normalized_record = string_keyed_mapping(raw_record)
        if normalized_record is None or not {
            "path",
            "family",
            "concept_id",
        }.issubset(normalized_record):
            raise AxBenchAdvError(f"project fit artifact record {index} is malformed")
        record: dict[str, Any] = dict(normalized_record)
        family = record["family"]
        concept_id = record["concept_id"]
        if (
            not isinstance(family, str)
            or family not in _FIT_FAMILIES
            or type(concept_id) is not int
            or concept_id < 0
            or not isinstance(record["path"], str)
        ):
            raise AxBenchAdvError(f"project fit artifact record {index} has invalid coordinates")
        coordinate = (family, concept_id)
        if coordinate in retained:
            raise AxBenchAdvError(f"project fit manifest duplicates coordinate {coordinate}")
        expected_relative = (
            Path(family) / "encdec" / "positive" / f"concept_{concept_id:04d}.safetensors"
        )
        if record["path"] != expected_relative.as_posix():
            raise AxBenchAdvError(
                f"project fit artifact {coordinate} has noncanonical path {record['path']!r}"
            )
        path = root / expected_relative
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise AxBenchAdvError(f"missing project fit artifact: {path}") from exc
        digest = hashlib.sha256(payload).hexdigest()
        request = DirectionArtifactRequest(
            family=family,
            target_model="encdec",
            mode="positive",
            concept_id=concept_id,
            layer=expectations.layer,
        )
        loaded = load_direction_artifact(
            payload,
            artifact_path=path,
            request=request,
            error_type=AxBenchAdvError,
        )
        if loaded.direction.dtype is not t.float32:
            raise AxBenchAdvError(
                f"project fit artifact direction must be float32 for {path}; "
                f"got {loaded.direction.dtype}"
            )
        require_fit_direction_normalization(
            loaded.direction,
            family=family,
            surface=f"project fit artifact {path}",
            error_cls=AxBenchAdvError,
        )
        metadata = loaded.metadata_dict()
        require_counter_calibration(
            metadata,
            direction=loaded.direction,
            surface=f"project fit artifact {path}",
            error_cls=AxBenchAdvError,
        )
        if metadata.get("capture_site") != expectations.site:
            raise AxBenchAdvError(f"project fit artifact capture site mismatch for {path}")
        if loaded.direction.numel() != expectations.decoder_width:
            raise AxBenchAdvError(
                f"project fit artifact width mismatch for {path}: "
                f"{loaded.direction.numel()} != {expectations.decoder_width}"
            )
        try:
            provenance = json.loads(metadata["provenance_json"])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise AxBenchAdvError(f"project fit artifact provenance is malformed: {path}") from exc
        if (
            not isinstance(provenance, Mapping)
            or provenance.get("family") != family
            or type(provenance.get("concept_id")) is not int
            or provenance.get("concept_id") != concept_id
            or provenance.get("concept") != expectations.concept_texts.get(concept_id)
        ):
            raise AxBenchAdvError(f"project fit artifact concept provenance mismatch: {path}")
        retained[coordinate] = ProjectFitArtifact(
            loaded=loaded,
            sha256=digest,
        )
    if set(retained) != expected_coordinates:
        raise AxBenchAdvError(
            "project fit artifact coverage differs from expected families/concepts: "
            f"missing={sorted(expected_coordinates - set(retained))[:10]}, "
            f"unexpected={sorted(set(retained) - expected_coordinates)[:10]}"
        )
    return ProjectDirectionFitPreflight(
        root=root,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        artifacts=MappingProxyType(retained),
    )


@dataclass(frozen=True)
class CounterDirection:
    direction: t.Tensor
    provenance: Mapping[str, Any]


@dataclass(frozen=True)
class GcgActivationDirection:
    """Positive project DiffMean calibration used only by activation-L2."""

    direction: t.Tensor
    max_activation: float
    provenance: Mapping[str, Any]


def load_project_gcg_activation_direction(
    project_fit: ProjectDirectionFitPreflight,
    *,
    concept_id: int,
) -> GcgActivationDirection:
    """Resolve activation-L2's own-model DiffMean without counter provenance."""

    retained = project_fit.artifacts.get(("diffmean", concept_id))
    if retained is None:
        raise AxBenchAdvError(
            f"missing project DiffMean fit for activation-L2 concept {concept_id}"
        )
    loaded = retained.loaded
    metadata = loaded.metadata_dict()
    scale = require_direction_max_activation(
        metadata,
        direction=loaded.direction,
        expected_concept_id=concept_id,
        expected_mode="positive",
        expected_target_model="encdec",
        expected_layer=loaded.identity.layer,
        surface=f"AxBench-Adv activation-L2 project DiffMean c{concept_id}",
        error_cls=AxBenchAdvError,
    )
    if scale <= 0.0:
        raise AxBenchAdvError("activation-L2 requires a nonzero positive DiffMean scale")
    return GcgActivationDirection(
        direction=loaded.direction,
        max_activation=scale,
        provenance={
            "objective": "activation_l2",
            "family": "diffmean",
            "concept_id": concept_id,
            "path": str(loaded.path),
            "sha256": retained.sha256,
            "fit_manifest_sha256": project_fit.manifest_sha256,
            "layer": loaded.identity.layer,
            "site": metadata.get("capture_site"),
            "mode": "positive",
            "target_model": "encdec",
            "direction_norm": float(loaded.direction.float().norm().item()),
            "max_activation": scale,
        },
    )


def load_counter_direction(
    *,
    pressure: Pressure,
    method: str,
    concept_id: int,
    baseline_train: Path,
    baseline_layer: int,
    project_fit: ProjectDirectionFitPreflight | None,
) -> CounterDirection:
    project = method in ROOT_METHODS
    if project:
        if project_fit is None:
            raise AxBenchAdvError(
                f"{pressure}/{method}/c{concept_id}: checkpoint-bound project direction "
                "fit preflight is required"
            )
        family = DIRECTION_FAMILY[pressure]
        retained = project_fit.artifacts.get((family, concept_id))
        if retained is None:
            raise AxBenchAdvError(
                f"missing direction artifact for {pressure}/{method}/c{concept_id} "
                f"under project fit {project_fit.root}"
            )
        loaded = retained.loaded
        path = loaded.path
        layer = loaded.identity.layer
        digest = retained.sha256
        direction = loaded.direction
    else:
        path = baseline_train / f"{DIRECTION_METHOD[pressure]}_weight.pt"
        layer = baseline_layer
        path = path.resolve()
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise AxBenchAdvError(
                f"missing direction artifact for {pressure}/{method}/c{concept_id}: {path}"
            ) from exc
        digest = hashlib.sha256(payload).hexdigest()
        direction = load_vendor_direction(path, concept_id, digest)
    validate_direction(direction)
    return CounterDirection(
        direction,
        {
            "pressure": pressure.value,
            "method": method,
            "concept_id": concept_id,
            "path": str(path),
            "sha256": digest,
            "layer": layer,
            "family": DIRECTION_FAMILY[pressure],
            "target_model": "encdec" if project else "vendored_baseline",
            "mode": "positive",
            "coefficient_semantics": COEFFICIENT_SEMANTICS,
            "direction_norm": float(direction.float().norm()),
        },
    )


def validate_counter_provenance(
    records: object,
    *,
    pressure: Pressure,
    methods: Sequence[str],
    concept_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Validate manifest ownership and the shared BPB-style artifact coordinates."""
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise AxBenchAdvError("counter_directions must be a list of provenance objects")
    normalized_records: list[dict[str, Any]] = []
    for raw_row in records:
        row = string_keyed_mapping(raw_row)
        if row is None:
            raise AxBenchAdvError(
                "counter_directions must contain only string-keyed provenance objects"
            )
        normalized_records.append(dict(row))
    selected = [row for row in normalized_records if row.get("pressure") == pressure]
    keys = [(row.get("method"), row.get("concept_id")) for row in selected]
    if len(keys) != len(set(keys)) or set(keys) != {(m, c) for m in methods for c in concept_ids}:
        raise AxBenchAdvError(f"counter direction provenance coverage differs for {pressure}")
    for row in selected:
        digest = row.get("sha256")
        layer = row.get("layer")
        norm = row.get("direction_norm")
        if (
            type(row.get("concept_id")) is not int
            or not is_sha256_hex(digest)
            or type(layer) is not int
            or layer < 0
            or row.get("family") != DIRECTION_FAMILY[pressure]
            or row.get("mode") != "positive"
            or row.get("coefficient_semantics") != COEFFICIENT_SEMANTICS
            or row.get("target_model")
            != ("encdec" if row["method"] in ROOT_METHODS else "vendored_baseline")
            or not isinstance(row.get("path"), str)
            or not Path(row["path"]).is_absolute()
            or isinstance(norm, bool)
            or not isinstance(norm, (int, float))
            or not math.isfinite(norm)
            or norm < 0
        ):
            raise AxBenchAdvError(
                f"invalid counter direction provenance for {pressure}/{row.get('method')}/c{row.get('concept_id')}"
            )
    return selected


def mean_counter_fitness(evaluations: Sequence[RawEvaluation]) -> float:
    """Use GEPA's exact per-generation adequacy gate and suppression objective."""
    if not evaluations:
        raise AxBenchAdvError("counter fitness requires generations")
    return sum(
        evaluate_fitness(
            concept_rating=raw.concept_rating,
            fluency_rating=raw.fluency_rating,
            output=raw.generation,
        ).score
        for raw in evaluations
    ) / len(evaluations)
