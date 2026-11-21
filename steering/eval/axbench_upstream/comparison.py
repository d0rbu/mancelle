"""Strict artifact comparison for AxBench-Upstream performance campaigns."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from safetensors import SafetensorError
from safetensors.torch import load

from scripts.eval.axbench_upstream.merge_methods import (
    INPUT_COLUMNS,
    KEY_COLUMNS,
    NL_PROJECT_METHODS,
    FrameSource,
    methods_for_source,
    validate_steering_frame,
)
from steering.eval.axbench_upstream.work_grid import (
    CellKey,
    build_encdec_cell_grid,
    build_project_cell_grid,
)
from steering.immutable_io import ImmutableArtifactError, stream_hash_file_descriptor_bound
from steering.safetensors_io import (
    canonicalize_safetensors_bytes,
    read_safetensors_metadata_from_bytes,
)

BundleKind = Literal["project", "encdec"]

DEFAULT_OPERATIONAL_MANIFEST_PATHS = frozenset(
    {
        "repository_git_commit",
        "repository_git_worktree",
        "configuration.output_path",
        "configuration.output_main_path",
        "configuration.output_aux_path",
        "configuration.manifest_path",
        "configuration.train_dir",
        "train_artifacts.directory",
    }
)


class ArtifactComparisonError(AssertionError):
    """A candidate artifact is not Tier-A equivalent to its reference."""


@dataclass(frozen=True)
class FileComparison:
    """Strict-byte comparison result for one file."""

    reference_path: str
    candidate_path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, int | str]:
        return {
            "reference_path": self.reference_path,
            "candidate_path": self.candidate_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


@dataclass(frozen=True)
class BundleComparison:
    """Machine-readable successful comparison report."""

    kind: BundleKind
    parquets: Mapping[str, FileComparison]
    training_artifacts: Mapping[str, FileComparison]
    manifest_reference_sha256: str
    manifest_candidate_sha256: str
    cell_count: int
    cell_grid_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parquets": {key: value.to_dict() for key, value in sorted(self.parquets.items())},
            "training_artifacts": {
                key: value.to_dict() for key, value in sorted(self.training_artifacts.items())
            },
            "manifest_reference_sha256": self.manifest_reference_sha256,
            "manifest_candidate_sha256": self.manifest_candidate_sha256,
            "cell_count": self.cell_count,
            "cell_grid_sha256": self.cell_grid_sha256,
        }


def _require_file(path: Path, *, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ArtifactComparisonError(f"{label} is not a readable file: {resolved}")
    return resolved


def _require_distinct_paths(reference: Path, candidate: Path, *, label: str) -> None:
    """Reject self-comparison, including symlink and hardlink aliases."""

    try:
        aliased = reference.samefile(candidate)
    except OSError as exc:
        raise ArtifactComparisonError(f"could not compare {label} path identities: {exc}") from exc
    if aliased:
        raise ArtifactComparisonError(
            f"reference and candidate {label} must be distinct files or directories: "
            f"{reference} aliases {candidate}"
        )


def _file_snapshot(path: Path, *, label: str) -> tuple[bytes, str]:
    """Read and digest one mutation-checked regular-file snapshot."""

    try:
        digest = stream_hash_file_descriptor_bound(path, retain_payload=True)
    except (OSError, ImmutableArtifactError) as exc:
        raise ArtifactComparisonError(
            f"{label} is not a stable readable file: {path}: {exc}"
        ) from exc
    payload = digest.payload
    if payload is None:  # pragma: no cover - guaranteed by retain_payload=True
        raise AssertionError("descriptor-bound comparison snapshot did not retain its payload")
    return payload, digest.sha256


def _first_sequence_difference(reference: Sequence[Any], candidate: Sequence[Any]) -> str:
    for index, (reference_value, candidate_value) in enumerate(
        zip(reference, candidate, strict=False)
    ):
        if reference_value != candidate_value:
            return f"index {index}: reference={reference_value!r}, candidate={candidate_value!r}"
    return f"length: reference={len(reference)}, candidate={len(candidate)}"


def _compare_parquet_snapshots(
    reference: Path,
    candidate: Path,
    *,
    label: str,
) -> tuple[FileComparison, pa.Table, pa.Table]:
    """Compare one immutable byte snapshot per path and retain decoded tables."""

    reference = _require_file(reference, label=f"reference {label}")
    candidate = _require_file(candidate, label=f"candidate {label}")
    _require_distinct_paths(reference, candidate, label=label)
    reference_payload, reference_sha = _file_snapshot(reference, label=f"reference {label}")
    candidate_payload, candidate_sha = _file_snapshot(candidate, label=f"candidate {label}")
    try:
        reference_table = pq.read_table(pa.BufferReader(reference_payload))
        candidate_table = pq.read_table(pa.BufferReader(candidate_payload))
    except (OSError, ValueError, pa.ArrowException) as exc:
        raise ArtifactComparisonError(f"{label} is not a valid Parquet pair: {exc}") from exc
    if not reference_table.schema.equals(candidate_table.schema, check_metadata=True):
        raise ArtifactComparisonError(
            f"{label} Parquet schema differs: reference={reference_table.schema}, "
            f"candidate={candidate_table.schema}"
        )
    if reference_table.num_rows != candidate_table.num_rows:
        raise ArtifactComparisonError(
            f"{label} Parquet row count differs: {reference_table.num_rows} "
            f"!= {candidate_table.num_rows}"
        )
    for name in reference_table.column_names:
        reference_column = reference_table[name]
        candidate_column = candidate_table[name]
        if not reference_column.equals(candidate_column):
            detail = _first_sequence_difference(
                reference_column.to_pylist(), candidate_column.to_pylist()
            )
            raise ArtifactComparisonError(f"{label} Parquet column {name!r} differs at {detail}")
    if reference_payload != candidate_payload:
        raise ArtifactComparisonError(
            f"{label} Parquet logical tables are equal but serialized bytes differ; "
            f"{label} bytes differ: reference(size={len(reference_payload)}, "
            f"sha256={reference_sha}) != candidate(size={len(candidate_payload)}, "
            f"sha256={candidate_sha})"
        )
    return (
        FileComparison(
            reference_path=str(reference),
            candidate_path=str(candidate),
            sha256=reference_sha,
            size_bytes=len(reference_payload),
        ),
        reference_table,
        candidate_table,
    )


def compare_parquet(reference: Path, candidate: Path, *, label: str) -> FileComparison:
    """Require byte equality and explain the first logical Parquet difference."""

    compared, _reference_table, _candidate_table = _compare_parquet_snapshots(
        reference,
        candidate,
        label=label,
    )
    return compared


def _compare_safetensors_snapshots(
    reference: Path,
    candidate: Path,
    *,
    label: str,
) -> tuple[FileComparison, str, str]:
    """Require exact logical and canonical-byte equality for one artifact.

    A legacy reference may have nondeterministic JSON key order in its header.
    The candidate itself must already be canonical; only the reference is
    normalized before the final byte comparison.
    """

    reference = _require_file(reference, label=f"reference {label}")
    candidate = _require_file(candidate, label=f"candidate {label}")
    _require_distinct_paths(reference, candidate, label=label)
    reference_payload, reference_raw_sha = _file_snapshot(
        reference,
        label=f"reference {label}",
    )
    candidate_payload, candidate_raw_sha = _file_snapshot(
        candidate,
        label=f"candidate {label}",
    )
    try:
        reference_metadata = dict(read_safetensors_metadata_from_bytes(reference_payload) or {})
        candidate_metadata = dict(read_safetensors_metadata_from_bytes(candidate_payload) or {})
        reference_tensors = load(reference_payload)
        candidate_tensors = load(candidate_payload)
    except (OSError, ValueError, SafetensorError) as exc:
        raise ArtifactComparisonError(
            f"{label} is not a valid strict safetensors pair: {exc}"
        ) from exc
    if reference_metadata != candidate_metadata:
        raise ArtifactComparisonError(
            f"{label} safetensors metadata differs: reference={reference_metadata!r}, "
            f"candidate={candidate_metadata!r}"
        )
    if set(reference_tensors) != set(candidate_tensors):
        raise ArtifactComparisonError(
            f"{label} safetensors keys differ: reference={sorted(reference_tensors)}, "
            f"candidate={sorted(candidate_tensors)}"
        )
    for key in sorted(reference_tensors):
        reference_tensor = reference_tensors[key]
        candidate_tensor = candidate_tensors[key]
        if reference_tensor.dtype != candidate_tensor.dtype:
            raise ArtifactComparisonError(
                f"{label} tensor {key!r} dtype differs: "
                f"{reference_tensor.dtype} != {candidate_tensor.dtype}"
            )
        if reference_tensor.shape != candidate_tensor.shape:
            raise ArtifactComparisonError(
                f"{label} tensor {key!r} shape differs: "
                f"{tuple(reference_tensor.shape)} != {tuple(candidate_tensor.shape)}"
            )
        if not t.equal(reference_tensor, candidate_tensor):
            mismatch = (reference_tensor != candidate_tensor).nonzero()
            index = tuple(int(value) for value in mismatch[0].tolist())
            raise ArtifactComparisonError(
                f"{label} tensor {key!r} differs at {index}: "
                f"reference={reference_tensor[index].item()!r}, "
                f"candidate={candidate_tensor[index].item()!r}"
            )
    try:
        canonical_reference = canonicalize_safetensors_bytes(reference_payload)
        canonical_candidate = canonicalize_safetensors_bytes(candidate_payload)
    except ValueError as exc:
        raise ArtifactComparisonError(
            f"{label} tensors and metadata are equal but a header is not canonicalizable: {exc}"
        ) from exc
    if candidate_payload != canonical_candidate:
        raise ArtifactComparisonError(
            f"{label} safetensor tensors and metadata are equal but candidate bytes are not "
            "canonical"
        )
    if canonical_reference != candidate_payload:
        reference_sha = hashlib.sha256(canonical_reference).hexdigest()
        candidate_sha = hashlib.sha256(candidate_payload).hexdigest()
        raise ArtifactComparisonError(
            f"{label} safetensor tensors and metadata are equal but canonical bytes differ: "
            f"reference(size={len(canonical_reference)}, sha256={reference_sha}) != "
            f"candidate(size={len(candidate_payload)}, sha256={candidate_sha})"
        )
    return (
        FileComparison(
            reference_path=str(reference),
            candidate_path=str(candidate),
            sha256=candidate_raw_sha,
            size_bytes=len(candidate_payload),
        ),
        reference_raw_sha,
        candidate_raw_sha,
    )


def compare_safetensors(reference: Path, candidate: Path, *, label: str) -> FileComparison:
    """Compare one pair and return the canonical candidate file evidence."""

    compared, _reference_raw_sha, _candidate_raw_sha = _compare_safetensors_snapshots(
        reference,
        candidate,
        label=label,
    )
    return compared


def _json_object_snapshot(path: Path, *, label: str) -> tuple[dict[str, Any], str]:
    path = _require_file(path, label=label)
    raw, sha256 = _file_snapshot(path, label=label)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for key, value in pairs:
            if key in payload:
                raise ValueError(f"duplicate JSON object key {key!r}")
            payload[key] = value
        return payload

    try:
        payload = json.loads(
            raw.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ArtifactComparisonError(f"{label} is not strict JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ArtifactComparisonError(f"{label} must contain a JSON object: {path}")
    return payload, sha256


def _replace_manifest_path(payload: dict[str, Any], dotted_path: str) -> None:
    parts = dotted_path.split(".")
    current: Any = payload
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict) and parts[-1] in current:
        current[parts[-1]] = "<operational>"


def _normalized_manifest(
    payload: dict[str, Any],
    *,
    operational_paths: frozenset[str],
) -> dict[str, Any]:
    normalized = json.loads(json.dumps(payload, sort_keys=True, allow_nan=False))
    for path in sorted(operational_paths):
        _replace_manifest_path(normalized, path)
    return normalized


def _first_json_difference(reference: Any, candidate: Any, *, path: str = "$") -> str:
    if type(reference) is not type(candidate):
        return f"{path}: type {type(reference).__name__} != {type(candidate).__name__}"
    if isinstance(reference, dict):
        reference_keys = set(reference)
        candidate_keys = set(candidate)
        if reference_keys != candidate_keys:
            return (
                f"{path}: keys differ; missing={sorted(reference_keys - candidate_keys)}, "
                f"unexpected={sorted(candidate_keys - reference_keys)}"
            )
        for key in sorted(reference):
            if reference[key] != candidate[key]:
                return _first_json_difference(reference[key], candidate[key], path=f"{path}.{key}")
        raise AssertionError("dictionary values differ but no differing key was found")
    if isinstance(reference, list):
        if len(reference) != len(candidate):
            return f"{path}: length {len(reference)} != {len(candidate)}"
        for index, (reference_value, candidate_value) in enumerate(
            zip(reference, candidate, strict=True)
        ):
            if reference_value != candidate_value:
                return _first_json_difference(
                    reference_value, candidate_value, path=f"{path}[{index}]"
                )
        raise AssertionError("list values differ but no differing index was found")
    return f"{path}: reference={reference!r}, candidate={candidate!r}"


def _replace_verified_training_digests(payload: dict[str, Any]) -> None:
    train_artifacts = payload.get("train_artifacts")
    files = train_artifacts.get("files") if isinstance(train_artifacts, dict) else None
    if not isinstance(files, list):
        raise ArtifactComparisonError(
            "verified encoder-decoder manifest has no train_artifacts.files list"
        )
    for entry in files:
        if not isinstance(entry, dict) or "sha256" not in entry:
            raise ArtifactComparisonError(
                "verified encoder-decoder manifest has an invalid training artifact entry"
            )
        entry["sha256"] = "<verified-raw-artifact-digest>"


def _assert_manifest_semantics_equal(
    reference: dict[str, Any],
    candidate: dict[str, Any],
    *,
    operational_paths: frozenset[str],
    normalize_verified_training_digests: bool = False,
) -> None:
    normalized_reference = _normalized_manifest(reference, operational_paths=operational_paths)
    normalized_candidate = _normalized_manifest(candidate, operational_paths=operational_paths)
    if normalize_verified_training_digests:
        _replace_verified_training_digests(normalized_reference)
        _replace_verified_training_digests(normalized_candidate)
    if normalized_reference != normalized_candidate:
        difference = _first_json_difference(normalized_reference, normalized_candidate)
        raise ArtifactComparisonError(f"manifest semantic content differs at {difference}")


def compare_manifests(
    reference: Path,
    candidate: Path,
    *,
    operational_paths: frozenset[str] = DEFAULT_OPERATIONAL_MANIFEST_PATHS,
) -> tuple[dict[str, Any], dict[str, Any], str, str]:
    """Require semantic equality after replacing literal operational fields."""

    reference = _require_file(reference, label="reference manifest")
    candidate = _require_file(candidate, label="candidate manifest")
    _require_distinct_paths(reference, candidate, label="manifest")
    reference_payload, reference_sha = _json_object_snapshot(reference, label="reference manifest")
    candidate_payload, candidate_sha = _json_object_snapshot(candidate, label="candidate manifest")
    _assert_manifest_semantics_equal(
        reference_payload,
        candidate_payload,
        operational_paths=operational_paths,
    )
    return reference_payload, candidate_payload, reference_sha, candidate_sha


def _config_path(manifest: Mapping[str, Any], key: str, *, label: str) -> Path:
    config = manifest.get("configuration")
    if not isinstance(config, dict):
        raise ArtifactComparisonError(f"{label} manifest has no object configuration")
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ArtifactComparisonError(f"{label} manifest configuration.{key} is not a path")
    return Path(value)


def _seed(manifest: Mapping[str, Any], *, label: str) -> int:
    config = manifest.get("configuration")
    value = config.get("seed") if isinstance(config, dict) else None
    if type(value) is not int or value < 0:
        raise ArtifactComparisonError(
            f"{label} manifest configuration.seed must be a nonnegative int excluding bool"
        )
    return value


def _max_concepts(manifest: Mapping[str, Any], *, label: str) -> int:
    configuration = manifest.get("configuration")
    value = configuration.get("max_concepts") if isinstance(configuration, dict) else None
    if type(value) is not int or value <= 0:
        raise ArtifactComparisonError(
            f"{label} manifest configuration.max_concepts must be a positive int excluding bool"
        )
    return value


def _validate_frame(
    frame: Any,
    *,
    source: FrameSource,
    max_concepts: int,
    label: str,
    methods: tuple[str, ...] | None = None,
    concept_ids: object = None,
) -> None:
    try:
        validate_steering_frame(
            frame,
            source=source,
            max_concepts=max_concepts,
            methods=methods,
            concept_ids=concept_ids,
        )
    except (TypeError, ValueError) as exc:
        raise ArtifactComparisonError(
            f"{label} violates its production frame contract: {exc}"
        ) from exc


def _validate_encdec_aux_alignment(
    main: Any,
    aux: Any,
    *,
    label: str,
) -> None:
    stable_inputs = tuple(column for column in INPUT_COLUMNS if column != "steered_input")
    main_factor_one = main.loc[main["factor"] == 1.0].sort_values(list(KEY_COLUMNS), kind="stable")
    sorted_aux = aux.sort_values(list(KEY_COLUMNS), kind="stable")
    if (
        not main_factor_one.loc[:, list(stable_inputs)]
        .reset_index(drop=True)
        .equals(sorted_aux.loc[:, list(stable_inputs)].reset_index(drop=True))
    ):
        raise ArtifactComparisonError(
            f"{label} encoder-decoder aux inputs do not match main factor==1 inputs"
        )


def _grid_digest(cells: Sequence[CellKey]) -> str:
    payload = json.dumps(
        [cell.to_dict() for cell in cells],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _compare_grids(reference: Sequence[CellKey], candidate: Sequence[CellKey]) -> str:
    if reference != candidate:
        detail = _first_sequence_difference(reference, candidate)
        raise ArtifactComparisonError(f"logical cell grid differs at {detail}")
    return _grid_digest(reference)


def _artifact_inventory(
    manifest: Mapping[str, Any],
    *,
    label: str,
) -> tuple[Path, dict[tuple[str, int, str], str]]:
    configuration = manifest.get("configuration")
    max_concepts = configuration.get("max_concepts") if isinstance(configuration, dict) else None
    if type(max_concepts) is not int or max_concepts <= 0:
        raise ArtifactComparisonError(
            f"{label} manifest configuration.max_concepts must be a positive int excluding bool"
        )
    train_artifacts = manifest.get("train_artifacts")
    if not isinstance(train_artifacts, dict):
        raise ArtifactComparisonError(f"{label} manifest has no train_artifacts object")
    directory = train_artifacts.get("directory")
    files = train_artifacts.get("files")
    if not isinstance(directory, str) or not isinstance(files, list):
        raise ArtifactComparisonError(
            f"{label} manifest train_artifacts requires string directory and list files"
        )
    inventory: dict[tuple[str, int, str], str] = {}
    seen_paths: set[str] = set()
    method_suffixes = {
        "EncDecDiffMean": "diffmean",
        "EncDecLinearProbe": "linear_probe",
        "EncDecLoRA": "lora",
    }
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ArtifactComparisonError(
                f"{label} manifest train_artifacts.files[{index}] must be an object"
            )
        method = entry.get("method")
        concept_id = entry.get("concept_id")
        path = entry.get("path")
        sha256 = entry.get("sha256")
        if (
            not isinstance(method, str)
            or type(concept_id) is not int
            or concept_id < 0
            or not isinstance(path, str)
            or not path
            or not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise ArtifactComparisonError(
                f"{label} manifest train_artifacts.files[{index}] has invalid identity"
            )
        suffix = method_suffixes.get(method)
        expected_path = (
            f"concept_{concept_id:05d}_{suffix}.safetensors" if suffix is not None else None
        )
        if path != expected_path:
            raise ArtifactComparisonError(
                f"{label} manifest train_artifacts.files[{index}] path {path!r} "
                f"does not match method/concept identity; expected {expected_path!r}"
            )
        if path in seen_paths:
            raise ArtifactComparisonError(
                f"{label} manifest repeats training artifact path {path!r}"
            )
        seen_paths.add(path)
        key = (method, concept_id, path)
        if key in inventory:
            raise ArtifactComparisonError(
                f"{label} manifest repeats training artifact identity {key!r}"
            )
        inventory[key] = sha256
    expected_inventory_keys = {
        (
            method,
            concept_id,
            f"concept_{concept_id:05d}_{suffix}.safetensors",
        )
        for concept_id in range(max_concepts)
        for method, suffix in method_suffixes.items()
    }
    if set(inventory) != expected_inventory_keys:
        raise ArtifactComparisonError(
            f"{label} manifest training artifact identities differ from the exact "
            f"3 x {max_concepts} grid: missing={sorted(expected_inventory_keys - set(inventory))}, "
            f"unexpected={sorted(set(inventory) - expected_inventory_keys)}"
        )
    resolved_directory = Path(directory).expanduser().resolve()
    if not resolved_directory.is_dir():
        raise ArtifactComparisonError(
            f"{label} manifest train_artifacts.directory is not a readable directory: "
            f"{resolved_directory}"
        )
    actual_names: set[str] = set()
    for path in resolved_directory.iterdir():
        if path.is_symlink():
            raise ArtifactComparisonError(f"{label} training directory contains a symlink: {path}")
        if not path.is_file():
            raise ArtifactComparisonError(
                f"{label} training directory contains a non-file entry: {path}"
            )
        actual_names.add(path.name)
    if actual_names != seen_paths:
        raise ArtifactComparisonError(
            f"{label} training directory inventory differs from its manifest: "
            f"missing={sorted(seen_paths - actual_names)}, "
            f"unexpected={sorted(actual_names - seen_paths)}"
        )
    return resolved_directory, inventory


def compare_project_bundle(
    reference_manifest_path: Path,
    candidate_manifest_path: Path,
    *,
    include_nl_steering: bool = False,
) -> BundleComparison:
    """Compare one project-method artifact bundle as a Tier-A candidate."""

    if type(include_nl_steering) is not bool:
        raise TypeError("include_nl_steering must be a boolean")
    methods = (
        NL_PROJECT_METHODS
        if include_nl_steering
        else methods_for_source(FrameSource.PROJECT)
    )
    operational_paths = DEFAULT_OPERATIONAL_MANIFEST_PATHS
    if include_nl_steering:
        # Producer records are relative to the output manifest. Their hashes
        # and complete scientific provenance remain exact comparison fields.
        operational_paths |= frozenset({
            "methods.NLSteering.target_source.description_embeddings.path",
            "methods.NLSteering.target_source.concept_embeddings.path",
        })

    reference, candidate, reference_manifest_sha, candidate_manifest_sha = compare_manifests(
        reference_manifest_path,
        candidate_manifest_path,
        operational_paths=operational_paths,
    )
    reference_output = _config_path(reference, "output_path", label="reference")
    candidate_output = _config_path(candidate, "output_path", label="candidate")
    parquet, reference_table, candidate_table = _compare_parquet_snapshots(
        reference_output,
        candidate_output,
        label="project output",
    )
    grids = []
    for label, manifest, table in (
        ("reference", reference, reference_table),
        ("candidate", candidate, candidate_table),
    ):
        frame = table.to_pandas()
        _validate_frame(
            frame,
            source=FrameSource.PROJECT,
            max_concepts=_max_concepts(manifest, label=label),
            label=f"{label} project output",
            methods=methods,
            concept_ids=manifest["configuration"].get("concept_ids"),
        )
        grids.append(
            build_project_cell_grid(frame, seed=_seed(manifest, label=label), methods=methods)
        )
    reference_grid, candidate_grid = grids
    grid_sha = _compare_grids(reference_grid, candidate_grid)
    return BundleComparison(
        kind="project",
        parquets={"project": parquet},
        training_artifacts={},
        manifest_reference_sha256=reference_manifest_sha,
        manifest_candidate_sha256=candidate_manifest_sha,
        cell_count=len(reference_grid),
        cell_grid_sha256=grid_sha,
    )


def compare_encdec_bundle(
    reference_manifest_path: Path,
    candidate_manifest_path: Path,
) -> BundleComparison:
    """Compare one encoder-decoder artifact bundle as a Tier-A candidate."""

    reference_manifest_path = _require_file(reference_manifest_path, label="reference manifest")
    candidate_manifest_path = _require_file(candidate_manifest_path, label="candidate manifest")
    _require_distinct_paths(
        reference_manifest_path,
        candidate_manifest_path,
        label="manifest",
    )
    reference, reference_manifest_sha = _json_object_snapshot(
        reference_manifest_path, label="reference manifest"
    )
    candidate, candidate_manifest_sha = _json_object_snapshot(
        candidate_manifest_path, label="candidate manifest"
    )
    main, reference_main_table, candidate_main_table = _compare_parquet_snapshots(
        _config_path(reference, "output_main_path", label="reference"),
        _config_path(candidate, "output_main_path", label="candidate"),
        label="encoder-decoder main",
    )
    aux, reference_aux_table, candidate_aux_table = _compare_parquet_snapshots(
        _config_path(reference, "output_aux_path", label="reference"),
        _config_path(candidate, "output_aux_path", label="candidate"),
        label="encoder-decoder aux",
    )
    parquets = {"main": main, "aux": aux}
    reference_main = reference_main_table.to_pandas()
    candidate_main = candidate_main_table.to_pandas()
    reference_aux = reference_aux_table.to_pandas()
    candidate_aux = candidate_aux_table.to_pandas()
    reference_max_concepts = _max_concepts(reference, label="reference")
    candidate_max_concepts = _max_concepts(candidate, label="candidate")
    for frame, source, max_concepts, label in (
        (
            reference_main,
            FrameSource.ENCDEC_MAIN,
            reference_max_concepts,
            "reference encoder-decoder main",
        ),
        (
            candidate_main,
            FrameSource.ENCDEC_MAIN,
            candidate_max_concepts,
            "candidate encoder-decoder main",
        ),
        (
            reference_aux,
            FrameSource.ENCDEC_AUX,
            reference_max_concepts,
            "reference encoder-decoder aux",
        ),
        (
            candidate_aux,
            FrameSource.ENCDEC_AUX,
            candidate_max_concepts,
            "candidate encoder-decoder aux",
        ),
    ):
        _validate_frame(frame, source=source, max_concepts=max_concepts, label=label)
    _validate_encdec_aux_alignment(
        reference_main,
        reference_aux,
        label="reference",
    )
    _validate_encdec_aux_alignment(
        candidate_main,
        candidate_aux,
        label="candidate",
    )
    reference_grid = build_encdec_cell_grid(
        reference_main, seed=_seed(reference, label="reference")
    )
    candidate_grid = build_encdec_cell_grid(
        candidate_main, seed=_seed(candidate, label="candidate")
    )
    grid_sha = _compare_grids(reference_grid, candidate_grid)

    reference_dir, reference_inventory = _artifact_inventory(reference, label="reference")
    candidate_dir, candidate_inventory = _artifact_inventory(candidate, label="candidate")
    _require_distinct_paths(reference_dir, candidate_dir, label="training directory")
    if set(reference_inventory) != set(candidate_inventory):
        raise ArtifactComparisonError(
            "encoder-decoder training-artifact manifest identities differ: "
            f"missing={sorted(set(reference_inventory) - set(candidate_inventory))}, "
            f"unexpected={sorted(set(candidate_inventory) - set(reference_inventory))}"
        )
    training: dict[str, FileComparison] = {}
    for method, concept_id, relative_path in sorted(reference_inventory):
        label = f"{method} concept {concept_id} ({relative_path})"
        reference_artifact = reference_dir / relative_path
        candidate_artifact = candidate_dir / relative_path
        compared, reference_actual_sha, candidate_actual_sha = _compare_safetensors_snapshots(
            reference_artifact,
            candidate_artifact,
            label=label,
        )
        reference_declared_sha = reference_inventory[(method, concept_id, relative_path)]
        candidate_declared_sha = candidate_inventory[(method, concept_id, relative_path)]
        if reference_actual_sha != reference_declared_sha:
            raise ArtifactComparisonError(
                f"reference {label} raw sha256 does not match its manifest: "
                f"declared={reference_declared_sha}, actual={reference_actual_sha}"
            )
        if candidate_actual_sha != candidate_declared_sha:
            raise ArtifactComparisonError(
                f"candidate {label} raw sha256 does not match its manifest: "
                f"declared={candidate_declared_sha}, actual={candidate_actual_sha}"
            )
        if compared.sha256 != candidate_actual_sha:
            raise AssertionError("canonical candidate digest changed across one comparison")
        training[f"{concept_id}:{method}:{relative_path}"] = compared

    _assert_manifest_semantics_equal(
        reference,
        candidate,
        operational_paths=DEFAULT_OPERATIONAL_MANIFEST_PATHS,
        normalize_verified_training_digests=True,
    )

    return BundleComparison(
        kind="encdec",
        parquets=parquets,
        training_artifacts=training,
        manifest_reference_sha256=reference_manifest_sha,
        manifest_candidate_sha256=candidate_manifest_sha,
        cell_count=len(reference_grid),
        cell_grid_sha256=grid_sha,
    )


def compare_bundle(
    kind: BundleKind,
    reference_manifest_path: Path,
    candidate_manifest_path: Path,
    *,
    include_nl_steering: bool = False,
) -> BundleComparison:
    """Dispatch one strict project or encoder-decoder bundle comparison."""

    if type(include_nl_steering) is not bool:
        raise TypeError("include_nl_steering must be a boolean")
    if kind == "project":
        return compare_project_bundle(
            reference_manifest_path,
            candidate_manifest_path,
            include_nl_steering=include_nl_steering,
        )
    if kind == "encdec":
        if include_nl_steering:
            raise ValueError("include_nl_steering is supported only for project bundles")
        return compare_encdec_bundle(reference_manifest_path, candidate_manifest_path)
    raise ValueError(f"unknown AxBench-Upstream bundle kind: {kind!r}")


__all__ = (
    "DEFAULT_OPERATIONAL_MANIFEST_PATHS",
    "ArtifactComparisonError",
    "BundleComparison",
    "FileComparison",
    "compare_bundle",
    "compare_encdec_bundle",
    "compare_manifests",
    "compare_parquet",
    "compare_project_bundle",
    "compare_safetensors",
)
