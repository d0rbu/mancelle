"""Shared identity contract for DiffMean and linear-probe directions.

Every consumer must validate the artifact's logical coordinates and training
layer before installing its residual hook.  Path-backed callers perform one
descriptor-bound byte read and parse metadata plus tensors from that payload.
Snapshot-backed callers parse both from the same retained byte payload, so
validation cannot describe one version of a file while the returned tensor
comes from another.
"""

from __future__ import annotations

import os
import stat
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import NoReturn

import torch as t
from safetensors import SafetensorError
from safetensors.torch import load as load_safetensors_bytes

from steering.safetensors_io import read_safetensors_metadata_from_bytes

type DirectionError = type[Exception]

_DIRECTION_FAMILIES = frozenset({"diffmean", "linear_probe"})
_TARGET_MODELS = frozenset({"decoder_only", "encdec"})
_MODES = frozenset({"positive", "negative"})
_NON_DIRECTION_BASELINE_FAMILIES = frozenset({"lora", "prompting"})
_CLUSTER_METADATA_FIELDS = frozenset(
    {
        "feature_contract_fingerprint",
        "source_corpus_sha256",
    }
)
_ARTIFACT_KIND_BY_FAMILY = {
    "diffmean": "diffmean_direction_v1",
    "linear_probe": "linear_probe_direction_v1",
}


@dataclass(frozen=True, slots=True)
class DirectionArtifactIdentity:
    """Canonical semantic identity stamped on one direction artifact."""

    family: str
    target_model: str
    mode: str
    concept_id: int
    layer: int
    artifact_kind: str | None = None

    @property
    def coordinate(self) -> tuple[str, str, str, int]:
        return (self.family, self.target_model, self.mode, self.concept_id)


@dataclass(frozen=True, slots=True)
class DirectionArtifactRequest:
    """One direction coordinate required by a runtime configuration."""

    family: str
    target_model: str
    mode: str
    concept_id: int
    layer: int

    @property
    def coordinate(self) -> tuple[str, str, str, int]:
        return (self.family, self.target_model, self.mode, self.concept_id)


@dataclass(frozen=True, slots=True)
class LoadedDirectionArtifact:
    """Validated tensor and identity decoded from one fused source read."""

    path: Path
    identity: DirectionArtifactIdentity
    direction: t.Tensor
    metadata: tuple[tuple[str, str], ...]

    def metadata_dict(self) -> dict[str, str]:
        return dict(self.metadata)


@dataclass(frozen=True, slots=True)
class DirectionTreeValidation:
    """Validated direction inventory and its single training layer."""

    artifacts: tuple[LoadedDirectionArtifact, ...]
    direction_training_layer: int | None

    def by_coordinate(
        self,
    ) -> dict[tuple[str, str, str, int], LoadedDirectionArtifact]:
        return {artifact.identity.coordinate: artifact for artifact in self.artifacts}


def expected_artifact_kind(family: str) -> str:
    try:
        return _ARTIFACT_KIND_BY_FAMILY[family]
    except KeyError as exc:
        raise ValueError(f"unsupported direction family {family!r}") from exc


def _raise(error_type: DirectionError, message: str) -> NoReturn:
    raise error_type(message)


def _canonical_non_negative_decimal(
    metadata: Mapping[str, str],
    key: str,
    *,
    path: Path,
    error_type: DirectionError,
) -> int:
    raw = metadata.get(key)
    if raw is None:
        _raise(error_type, f"direction artifact {path} is missing metadata {key!r}")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise error_type(
            f"direction artifact {path} has noncanonical {key} metadata {raw!r}; "
            "expected a non-negative canonical decimal"
        ) from exc
    if value < 0 or str(value) != raw:
        _raise(
            error_type,
            f"direction artifact {path} has noncanonical {key} metadata {raw!r}; "
            "expected a non-negative canonical decimal",
        )
    return value


def _path_coordinates(
    path: Path,
    *,
    error_type: DirectionError,
) -> tuple[str, str, str, int]:
    parts = path.parts
    if len(parts) < 4:
        _raise(
            error_type,
            f"direction artifact path {path} does not contain "
            "<family>/<target>/<mode>/concept_<id>.safetensors coordinates",
        )
    family, target_model, mode, filename = parts[-4:]
    if family not in _DIRECTION_FAMILIES:
        _raise(error_type, f"direction artifact path {path} has unknown family {family!r}")
    if target_model not in _TARGET_MODELS:
        _raise(
            error_type,
            f"direction artifact path {path} has unknown target model {target_model!r}",
        )
    if mode not in _MODES:
        _raise(error_type, f"direction artifact path {path} has unknown mode {mode!r}")
    prefix = "concept_"
    suffix = ".safetensors"
    if not filename.startswith(prefix) or not filename.endswith(suffix):
        _raise(error_type, f"direction artifact path {path} has malformed filename {filename!r}")
    raw_concept_id = filename[len(prefix) : -len(suffix)]
    try:
        concept_id = int(raw_concept_id)
    except ValueError as exc:
        raise error_type(
            f"direction artifact path {path} has malformed concept coordinate {raw_concept_id!r}"
        ) from exc
    if concept_id < 0 or filename != f"concept_{concept_id:04d}.safetensors":
        _raise(
            error_type,
            f"direction artifact path {path} has noncanonical concept filename "
            f"{filename!r}; expected {f'concept_{concept_id:04d}.safetensors'!r}",
        )
    return family, target_model, mode, concept_id


def _validate_decoded_direction(
    *,
    path: Path,
    metadata: Mapping[str, str],
    tensors: Mapping[str, t.Tensor],
    request: DirectionArtifactRequest | None,
    error_type: DirectionError,
) -> LoadedDirectionArtifact:
    family, path_target, path_mode, path_concept_id = _path_coordinates(
        path,
        error_type=error_type,
    )
    concept_id = _canonical_non_negative_decimal(
        metadata,
        "concept_id",
        path=path,
        error_type=error_type,
    )
    layer = _canonical_non_negative_decimal(
        metadata,
        "layer",
        path=path,
        error_type=error_type,
    )
    target_model = metadata.get("target_model")
    mode = metadata.get("mode")
    actual_coordinates = {
        "family": family,
        "target_model": target_model,
        "mode": mode,
        "concept_id": concept_id,
    }
    path_coordinates = {
        "family": family,
        "target_model": path_target,
        "mode": path_mode,
        "concept_id": path_concept_id,
    }
    if (
        target_model is None
        or target_model not in _TARGET_MODELS
        or mode is None
        or mode not in _MODES
    ):
        _raise(
            error_type,
            f"direction artifact {path} has invalid metadata coordinates "
            f"{actual_coordinates}; path coordinates are {path_coordinates}",
        )
    if target_model != path_target or mode != path_mode or concept_id != path_concept_id:
        _raise(
            error_type,
            f"direction artifact identity mismatch at {path}: path coordinates "
            f"{path_coordinates}, metadata coordinates {actual_coordinates}",
        )

    expected_kind = expected_artifact_kind(family)
    artifact_kind = metadata.get("artifact_kind")
    cluster_stamped = any(key in metadata for key in _CLUSTER_METADATA_FIELDS)
    if artifact_kind is not None and artifact_kind != expected_kind:
        _raise(
            error_type,
            f"direction artifact {path} has artifact_kind={artifact_kind!r}; "
            f"expected {expected_kind!r} for family={family!r}",
        )
    if cluster_stamped and artifact_kind is None:
        _raise(
            error_type,
            f"cluster-stamped direction artifact {path} is missing required "
            f"artifact_kind={expected_kind!r}",
        )

    if "direction" not in tensors:
        _raise(
            error_type,
            f"direction artifact {path} must contain tensor key 'direction'; "
            f"found {sorted(tensors)}",
        )
    direction = tensors["direction"]
    if direction.ndim != 1 or direction.numel() == 0:
        _raise(
            error_type,
            f"direction artifact {path} must contain one non-empty 1D direction; "
            f"got shape={tuple(direction.shape)}",
        )
    if not direction.is_floating_point():
        _raise(
            error_type,
            f"direction artifact {path} direction must be floating point; "
            f"got dtype={direction.dtype}",
        )
    try:
        finite = bool(t.isfinite(direction).all().item())
    except (NotImplementedError, RuntimeError):
        # Some floating dtypes (notably CPU float8_e4m3fn) are valid
        # safetensors but lack a native isfinite kernel. Their entire finite
        # range is representable in float32, so use it only for validation and
        # retain the original tensor/dtype for the consumer.
        try:
            finite = bool(t.isfinite(direction.to(dtype=t.float32)).all().item())
        except (NotImplementedError, RuntimeError) as exc:
            raise error_type(
                f"direction artifact {path} finiteness cannot be validated for "
                f"dtype={direction.dtype}"
            ) from exc
    if not finite:
        _raise(error_type, f"direction artifact {path} direction contains NaN or Inf")

    identity = DirectionArtifactIdentity(
        family=family,
        target_model=target_model,
        mode=mode,
        concept_id=concept_id,
        layer=layer,
        artifact_kind=artifact_kind,
    )
    if request is not None:
        requested_coordinates = {
            "family": request.family,
            "target_model": request.target_model,
            "mode": request.mode,
            "concept_id": request.concept_id,
        }
        if request.coordinate != identity.coordinate:
            _raise(
                error_type,
                f"direction artifact request mismatch at {path}: requested "
                f"coordinates {requested_coordinates}, actual coordinates "
                f"{actual_coordinates}",
            )
        if request.layer != layer:
            _raise(
                error_type,
                f"direction artifact layer mismatch at {path}: requested/applied "
                f"layer={request.layer}, artifact training layer={layer}; "
                f"requested coordinates {requested_coordinates}, actual "
                f"coordinates {actual_coordinates}",
            )
    return LoadedDirectionArtifact(
        path=path,
        identity=identity,
        direction=direction,
        metadata=tuple(sorted(metadata.items())),
    )


def _read_regular_file_bytes(path: Path, *, error_type: DirectionError) -> bytes:
    """Read through an ``O_NOFOLLOW`` ``openat`` walk of all coordinates."""

    if len(path.parts) < 4 or ".." in path.parts:
        _raise(error_type, f"direction artifact path is not a safe coordinate path: {path}")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor: int | None = None
    try:
        # ``parents[3]`` is the baselines root for
        # family/target/mode/file. Higher workspace components may be managed
        # symlinks, but no component that supplies artifact semantics may be.
        descriptor = os.open(path.parents[3], directory_flags)
        for component in path.parts[-4:-1]:
            child_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child_descriptor
        file_descriptor = os.open(
            path.name,
            file_flags,
            dir_fd=descriptor,
        )
        os.close(descriptor)
        descriptor = file_descriptor
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
            descriptor = None
        raise error_type(
            f"direction artifact path must be a real regular file: {path}: {exc}"
        ) from exc
    try:
        opened_stat = os.fstat(descriptor)
        if not stat.S_ISREG(opened_stat.st_mode):
            _raise(
                error_type,
                f"direction artifact path must be a real regular file: {path}",
            )
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            return handle.read()
    except OSError as exc:
        raise error_type(f"failed reading direction artifact {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def load_direction_artifact(
    source: Path | bytes,
    *,
    artifact_path: Path | None = None,
    request: DirectionArtifactRequest | None = None,
    error_type: DirectionError,
) -> LoadedDirectionArtifact:
    """Load and validate one path-backed or retained-byte direction artifact."""

    if isinstance(source, Path):
        if artifact_path is not None and artifact_path != source:
            _raise(
                error_type,
                f"path-backed direction source {source} disagrees with "
                f"artifact_path={artifact_path}",
            )
        path = source
        try:
            payload = _read_regular_file_bytes(path, error_type=error_type)
            metadata = dict(read_safetensors_metadata_from_bytes(payload) or {})
            tensors = load_safetensors_bytes(payload)
        except error_type:
            raise
        except (RuntimeError, SafetensorError, ValueError) as exc:
            raise error_type(f"direction artifact {path} is not valid safetensors: {exc}") from exc
    elif type(source) is bytes:
        if artifact_path is None:
            _raise(error_type, "retained direction bytes require their logical artifact_path")
        path = artifact_path
        try:
            metadata = dict(read_safetensors_metadata_from_bytes(source) or {})
            tensors = load_safetensors_bytes(source)
        except (SafetensorError, RuntimeError, ValueError) as exc:
            raise error_type(
                f"direction artifact retained for {path} is not valid safetensors: {exc}"
            ) from exc
    else:
        _raise(
            error_type,
            f"direction artifact source must be Path or exact bytes, got {type(source).__name__}",
        )
    return _validate_decoded_direction(
        path=path,
        metadata=metadata,
        tensors=tensors,
        request=request,
        error_type=error_type,
    )


def direction_artifact_path(root: Path, request: DirectionArtifactRequest) -> Path:
    """Resolve one canonical request beneath a ``baselines/`` root."""

    return (
        root
        / request.family
        / request.target_model
        / request.mode
        / f"concept_{request.concept_id:04d}.safetensors"
    )


def _direction_payload_paths(
    payloads: Mapping[str, bytes],
    *,
    root: Path,
    error_type: DirectionError,
) -> list[tuple[Path, bytes]]:
    result: list[tuple[Path, bytes]] = []
    allowed_top_level = _DIRECTION_FAMILIES | _NON_DIRECTION_BASELINE_FAMILIES
    for raw_relative, payload in payloads.items():
        relative = PurePosixPath(raw_relative)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            _raise(error_type, f"direction tree has unsafe entry {raw_relative!r}")
        top_level = relative.parts[0]
        if top_level not in allowed_top_level:
            _raise(
                error_type,
                f"direction tree {root} has unknown top-level entry {top_level!r}",
            )
        if top_level in _NON_DIRECTION_BASELINE_FAMILIES:
            continue
        if len(relative.parts) != 4:
            _raise(
                error_type,
                f"direction tree {root} has malformed entry {raw_relative!r}; "
                "expected family/target/mode/concept_<id>.safetensors",
            )
        if type(payload) is not bytes:
            _raise(
                error_type,
                f"direction tree retained payload {raw_relative!r} must be exact bytes",
            )
        result.append((root.joinpath(*relative.parts), payload))
    return sorted(result, key=lambda item: item[0])


def _validate_retained_tree_manifest(
    entries: Iterable[tuple[str, str]],
    *,
    payloads: Mapping[str, bytes],
    root: Path,
    error_type: DirectionError,
) -> None:
    manifest = tuple(entries)
    if len({relative for relative, _kind in manifest}) != len(manifest):
        _raise(error_type, f"direction tree {root} retained manifest has duplicates")
    file_entries: set[str] = set()
    allowed_top_level = _DIRECTION_FAMILIES | _NON_DIRECTION_BASELINE_FAMILIES
    for raw_relative, kind in manifest:
        relative = PurePosixPath(raw_relative)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not relative.parts
            or kind not in {"dir", "file"}
        ):
            _raise(
                error_type,
                f"direction tree {root} has malformed retained entry "
                f"{raw_relative!r} kind={kind!r}",
            )
        top_level = relative.parts[0]
        if top_level not in allowed_top_level:
            _raise(
                error_type,
                f"direction tree {root} has unknown top-level entry {top_level!r}",
            )
        if kind == "file":
            file_entries.add(raw_relative)
        if top_level in _NON_DIRECTION_BASELINE_FAMILIES:
            continue
        depth = len(relative.parts)
        if kind == "dir":
            if depth > 3:
                _raise(
                    error_type,
                    f"direction tree {root} has malformed directory {raw_relative!r}",
                )
            if depth >= 2 and relative.parts[1] not in _TARGET_MODELS:
                _raise(
                    error_type,
                    f"direction tree {root} has unknown target directory {raw_relative!r}",
                )
            if depth == 3 and relative.parts[2] not in _MODES:
                _raise(
                    error_type,
                    f"direction tree {root} has unknown mode directory {raw_relative!r}",
                )
        elif depth != 4:
            _raise(
                error_type,
                f"direction tree {root} has malformed file {raw_relative!r}",
            )
    if file_entries != set(payloads):
        _raise(
            error_type,
            f"direction tree {root} retained manifest/payload files disagree: "
            f"missing_payloads={sorted(file_entries - set(payloads))[:20]}, "
            f"unlisted_payloads={sorted(set(payloads) - file_entries)[:20]}",
        )


def _direction_files(
    root: Path,
    *,
    error_type: DirectionError,
) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        _raise(error_type, f"direction tree root must be a real directory: {root}")
    allowed_top_level = _DIRECTION_FAMILIES | _NON_DIRECTION_BASELINE_FAMILIES
    files: list[Path] = []
    for top in sorted(root.iterdir()):
        if top.name not in allowed_top_level:
            _raise(
                error_type,
                f"direction tree {root} has unknown top-level entry {top.name!r}",
            )
        if top.name in _NON_DIRECTION_BASELINE_FAMILIES:
            continue
        if top.is_symlink() or not top.is_dir():
            _raise(error_type, f"direction family entry must be a real directory: {top}")
        for child in sorted(top.rglob("*")):
            if child.is_symlink():
                _raise(error_type, f"direction tree must not contain symlinks: {child}")
            relative = child.relative_to(root)
            if child.is_file():
                if len(relative.parts) != 4:
                    _raise(
                        error_type,
                        f"direction tree {root} has malformed entry {relative.as_posix()!r}",
                    )
                files.append(child)
            elif child.is_dir():
                depth = len(relative.parts)
                if depth > 3:
                    _raise(
                        error_type,
                        f"direction tree {root} has malformed directory {relative.as_posix()!r}",
                    )
                if depth >= 2 and relative.parts[1] not in _TARGET_MODELS:
                    _raise(
                        error_type,
                        f"direction tree {root} has unknown target directory "
                        f"{relative.as_posix()!r}",
                    )
                if depth == 3 and relative.parts[2] not in _MODES:
                    _raise(
                        error_type,
                        f"direction tree {root} has unknown mode directory {relative.as_posix()!r}",
                    )
            else:
                _raise(
                    error_type,
                    f"direction tree may contain only regular files/directories: {child}",
                )
    return files


def validate_direction_tree(
    root: Path,
    *,
    retained_payloads: Mapping[str, bytes] | None = None,
    retained_child_kinds: Iterable[tuple[str, str]] | None = None,
    error_type: DirectionError,
) -> DirectionTreeValidation:
    """Validate every present direction and derive its one training layer."""

    artifacts: list[LoadedDirectionArtifact] = []
    if retained_payloads is None:
        if retained_child_kinds is not None:
            _raise(
                error_type,
                "retained_child_kinds requires retained_payloads",
            )
        for path in _direction_files(root, error_type=error_type):
            artifacts.append(load_direction_artifact(path, error_type=error_type))
    else:
        if retained_child_kinds is not None:
            _validate_retained_tree_manifest(
                retained_child_kinds,
                payloads=retained_payloads,
                root=root,
                error_type=error_type,
            )
        for path, payload in _direction_payload_paths(
            retained_payloads,
            root=root,
            error_type=error_type,
        ):
            artifacts.append(
                load_direction_artifact(
                    payload,
                    artifact_path=path,
                    error_type=error_type,
                )
            )
    coordinates = [artifact.identity.coordinate for artifact in artifacts]
    if len(coordinates) != len(set(coordinates)):
        _raise(error_type, f"direction tree {root} contains duplicate logical coordinates")
    layers = sorted({artifact.identity.layer for artifact in artifacts})
    if len(layers) > 1:
        examples = [
            f"{artifact.path}:layer={artifact.identity.layer}"
            for artifact in artifacts
            if artifact.identity.layer in {layers[0], layers[-1]}
        ]
        _raise(
            error_type,
            f"direction artifacts under {root} disagree on training layer: "
            f"layers={layers}; examples={examples[:6]}",
        )
    return DirectionTreeValidation(
        artifacts=tuple(artifacts),
        direction_training_layer=layers[0] if layers else None,
    )


def require_direction_artifacts(
    validation: DirectionTreeValidation,
    requirements: Iterable[DirectionArtifactRequest],
    *,
    surface: str,
    error_type: DirectionError,
) -> None:
    """Prove all selected runtime coordinates exist at their configured layer."""

    requested = sorted(
        set(requirements),
        key=lambda item: (
            item.family,
            item.target_model,
            item.mode,
            item.concept_id,
            item.layer,
        ),
    )
    if not requested:
        return
    if validation.direction_training_layer is None:
        _raise(
            error_type,
            f"{surface} selects direction-consuming methods, but the source "
            "contract has direction_training_layer=null",
        )
    by_coordinate = validation.by_coordinate()
    for request in requested:
        artifact = by_coordinate.get(request.coordinate)
        if artifact is None:
            _raise(
                error_type,
                f"{surface} requires missing direction coordinate "
                f"family={request.family!r}, target_model={request.target_model!r}, "
                f"mode={request.mode!r}, concept_id={request.concept_id}, "
                f"configured layer={request.layer}",
            )
        if artifact.identity.layer != request.layer:
            _raise(
                error_type,
                f"{surface} direction layer mismatch at {artifact.path}: "
                f"requested/applied layer={request.layer}, artifact training "
                f"layer={artifact.identity.layer}; requested coordinates "
                f"family={request.family!r}, target_model={request.target_model!r}, "
                f"mode={request.mode!r}, concept_id={request.concept_id}; actual "
                f"coordinates family={artifact.identity.family!r}, "
                f"target_model={artifact.identity.target_model!r}, "
                f"mode={artifact.identity.mode!r}, "
                f"concept_id={artifact.identity.concept_id}",
            )


__all__ = [
    "DirectionArtifactIdentity",
    "DirectionArtifactRequest",
    "DirectionTreeValidation",
    "LoadedDirectionArtifact",
    "direction_artifact_path",
    "expected_artifact_kind",
    "load_direction_artifact",
    "require_direction_artifacts",
    "validate_direction_tree",
]
