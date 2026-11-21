"""Build a model-free AxBench input frame for project-only evaluations."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    INPUTS_PER_CONCEPT,
    validate_project_input_frame,
)
from scripts.eval.axbench_upstream.run_baselines import AXBENCH_UPSTREAM_COMMIT
from steering._common.strict_json import strict_json_object
from steering.eval.axbench_upstream.data import load_metadata
from steering.immutable_io import (
    ImmutableArtifactError,
    stream_hash_file_descriptor_bound,
)

PROJECT_INPUT_FORMAT = "axbench-upstream-project-input-v1"
AXBENCH_BENCHMARK_REPO = "pyvene/axbench-concept500"
AXBENCH_BENCHMARK_REVISION = "ad8a5d60c4616b599c24dd6689f05f696ec610f3"
ALPACA_EVAL_REPO = "tatsu-lab/alpaca_eval"
ALPACA_EVAL_REVISION = "2edc6fad8be6b14ea7230aabfd08188da6b8b814"
ALPACA_EVAL_FILE = "alpaca_eval.json"
_PREPARED_DATASETS = {
    "benchmark": {"repo": AXBENCH_BENCHMARK_REPO, "revision": AXBENCH_BENCHMARK_REVISION},
    "alpaca_eval": {
        "repo": ALPACA_EVAL_REPO,
        "revision": ALPACA_EVAL_REVISION,
        "file": ALPACA_EVAL_FILE,
    },
}
_PREPARED_SOURCE_PATHS = ("generate/metadata.jsonl", "data/alpaca_eval.json")
_CONFIG_KEYS = frozenset(
    {
        "prepared_dir",
        "output_path",
        "manifest_path",
        "max_concepts",
        "force",
    }
)


def _sha256(path: Path) -> str:
    return stream_hash_file_descriptor_bound(path).sha256


def _strict_json_object_with_digest(path: Path) -> tuple[dict[str, Any], str]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        digest = stream_hash_file_descriptor_bound(path, retain_payload=True)
        payload = cast(bytes, digest.payload)
        loaded = json.loads(
            payload.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=strict_json_object,
        )
    except (
        ImmutableArtifactError,
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        raise ValueError(f"expected a readable strict JSON object at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return loaded, digest.sha256


def _sae_id(reference: object, *, concept_id: int) -> int:
    if not isinstance(reference, str) or not reference:
        raise ValueError(f"metadata concept_id={concept_id} has no nonempty ref")
    try:
        value = int(reference.rsplit("/", 1)[-1])
    except ValueError as exc:
        raise ValueError(
            f"metadata concept_id={concept_id} ref does not end in an integer SAE id: "
            f"{reference!r}"
        ) from exc
    if value < 0:
        raise ValueError(f"metadata concept_id={concept_id} SAE id must be nonnegative")
    return value


def _require_fields(
    value: object,
    expected: Mapping[str, object],
    *,
    label: str,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    for field, expected_value in expected.items():
        actual = value.get(field)
        if actual != expected_value or type(actual) is not type(expected_value):
            raise ValueError(f"{label} field {field!r} must equal {expected_value!r}")


def _validated_prepared_sha256(prepared_dir: Path) -> str:
    path = prepared_dir / "prepare_manifest.json"
    manifest, manifest_sha256 = _strict_json_object_with_digest(path)
    _require_fields(
        manifest,
        {
            "schema_version": 1,
            "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
        },
        label="prepared manifest",
    )
    for key, expected in _PREPARED_DATASETS.items():
        _require_fields(manifest.get(key), expected, label=f"prepared manifest {key}")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise ValueError("prepared manifest artifacts must be an object")
    for relative in _PREPARED_SOURCE_PATHS:
        artifact = artifacts.get(relative)
        if not isinstance(artifact, dict):
            raise ValueError(f"prepared manifest has no artifact identity for {relative!r}")
        expected_sha256 = artifact.get("sha256")
        expected_bytes = artifact.get("bytes")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise ValueError(f"prepared manifest artifact {relative!r} has no valid sha256")
        if type(expected_bytes) is not int or expected_bytes < 0:
            raise ValueError(f"prepared manifest artifact {relative!r} has no valid byte count")
        source_path = prepared_dir / relative
        if not source_path.is_file():
            raise FileNotFoundError(f"prepared source artifact does not exist: {source_path}")
        digest = stream_hash_file_descriptor_bound(source_path)
        if digest.byte_length != expected_bytes or digest.sha256 != expected_sha256:
            raise ValueError(
                f"prepared source artifact {relative!r} does not match prepare_manifest.json"
            )
    return manifest_sha256


def _require_prepared_unchanged(prepared_dir: Path, prepared_sha256: str) -> None:
    if _validated_prepared_sha256(prepared_dir) != prepared_sha256:
        raise ValueError("prepared AxBench sources changed while project input was being built")


def _build_project_input_frame(
    prepared_dir: Path,
    *,
    max_concepts: int,
) -> tuple[pd.DataFrame, str]:
    if type(max_concepts) is not int or max_concepts <= 0:
        raise ValueError("max_concepts must be a positive integer")
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    prepared_sha256 = _validated_prepared_sha256(prepared_dir)
    metadata = load_metadata(prepared_dir / "generate" / "metadata.jsonl")
    if max_concepts > len(metadata):
        raise ValueError(f"max_concepts={max_concepts} exceeds {len(metadata)} metadata rows")

    alpaca_path = prepared_dir / "data" / "alpaca_eval.json"
    alpaca = pd.read_json(alpaca_path)
    if "instruction" not in alpaca.columns or len(alpaca) < INPUTS_PER_CONCEPT:
        raise ValueError(
            f"prepared AlpacaEval data must contain at least {INPUTS_PER_CONCEPT} instructions"
        )
    if not alpaca["instruction"].map(lambda value: isinstance(value, str)).all():
        raise ValueError("prepared AlpacaEval instructions must all be strings")

    rows: list[dict[str, object]] = []
    for concept_id, item in enumerate(metadata[:max_concepts]):
        concept = item.get("concept")
        if not isinstance(concept, str) or not concept:
            raise ValueError(f"metadata concept_id={concept_id} has no nonempty concept")
        reference = item.get("ref")
        sae_id = _sae_id(reference, concept_id=concept_id)
        prompts = alpaca.sample(INPUTS_PER_CONCEPT, random_state=concept_id)[
            "instruction"
        ].tolist()
        steering_prefix = (
            "You must answer the question with content related to "
            f"{concept} even if it is not related to the question or it does not make sense."
        )
        for input_id, prompt_value in enumerate(prompts):
            prompt = str(prompt_value)
            for factor in FULL_FACTORS:
                rows.append(
                    {
                        "dataset_name": "AlpacaEval",
                        "concept_id": concept_id,
                        "input_concept": concept,
                        "input_id": input_id,
                        "factor": factor,
                        "original_prompt": prompt,
                        "steered_input": f"{steering_prefix}\n\nQuestion: {prompt}",
                        "input": prompt,
                        "suppress_original": "",
                        "suppress_rewrite": "",
                        "steered_prompt": "",
                        "defense": "[]",
                        "sae_link": str(reference),
                        "sae_id": sae_id,
                    }
                )

    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    for column in ("concept_id", "input_id", "sae_id"):
        frame[column] = frame[column].astype("int64")
    frame["factor"] = frame["factor"].astype("float64")
    validate_project_input_frame(frame, max_concepts=max_concepts)
    _require_prepared_unchanged(prepared_dir, prepared_sha256)
    return frame, prepared_sha256


def build_project_input_frame(prepared_dir: Path, *, max_concepts: int) -> pd.DataFrame:
    """Reproduce AxBench's AlpacaEval input grid without loading any model."""

    frame, _ = _build_project_input_frame(prepared_dir, max_concepts=max_concepts)
    return frame


def _manifest_payload(
    *,
    max_concepts: int,
    rows: int,
    prepared_sha256: str,
    output_sha256: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "format": PROJECT_INPUT_FORMAT,
        "source_kind": "model_free_project_input",
        "max_concepts": max_concepts,
        "inputs_per_concept": INPUTS_PER_CONCEPT,
        "factors": list(FULL_FACTORS),
        "rows": rows,
        "model_loaded": False,
        "model_generations_present": False,
        "generation_prompt_column": "original_prompt",
        "prepared_manifest_sha256": prepared_sha256,
        "output_sha256": output_sha256,
    }


def _validate_project_input_manifest(
    manifest: Mapping[str, Any],
    *,
    input_sha256: str,
    prepared_dir: Path,
    max_concepts: int,
    row_count: int,
) -> str:
    prepared_sha256 = _validated_prepared_sha256(prepared_dir)
    _require_fields(
        manifest,
        _manifest_payload(
            max_concepts=max_concepts,
            rows=row_count,
            prepared_sha256=prepared_sha256,
            output_sha256=input_sha256,
        ),
        label="project input manifest",
    )
    return prepared_sha256


def load_project_input_artifact(
    *,
    input_path: Path,
    manifest_path: Path,
    prepared_dir: Path,
    max_concepts: int,
) -> tuple[pd.DataFrame, str, str]:
    """Load and bind a model-free Parquet to its producer and prepared-source manifest."""

    input_path = Path(input_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    if input_path == manifest_path:
        raise ValueError("project input Parquet and manifest paths must be distinct")
    if input_path.is_relative_to(prepared_dir) or manifest_path.is_relative_to(prepared_dir):
        raise ValueError("project input artifacts must be outside the prepared source directory")
    manifest, manifest_sha256 = _strict_json_object_with_digest(manifest_path)
    input_sha256 = _sha256(input_path)
    frame = pd.read_parquet(input_path)
    if _sha256(input_path) != input_sha256:
        raise ValueError("project input Parquet changed while it was being loaded")
    validate_project_input_frame(frame, max_concepts=max_concepts)
    prepared_sha256 = _validate_project_input_manifest(
        manifest,
        input_sha256=input_sha256,
        prepared_dir=prepared_dir,
        max_concepts=max_concepts,
        row_count=len(frame),
    )
    if _sha256(manifest_path) != manifest_sha256:
        raise ValueError("project input manifest changed while it was being loaded")
    _require_prepared_unchanged(prepared_dir, prepared_sha256)
    return frame, input_sha256, manifest_sha256


def write_project_input(
    *,
    prepared_dir: Path,
    output_path: Path,
    manifest_path: Path,
    max_concepts: int,
    force: bool = False,
) -> pd.DataFrame:
    """Build, atomically publish, and self-validate a project input artifact."""

    if type(force) is not bool:
        raise ValueError("force must be a boolean")
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    output_path = Path(output_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    if (
        output_path == manifest_path
        or output_path.is_relative_to(prepared_dir)
        or manifest_path.is_relative_to(prepared_dir)
    ):
        raise ValueError("project input output and manifest paths must be disjoint from all inputs")
    existing = [path for path in (output_path, manifest_path) if path.exists()]
    if existing and not force:
        raise FileExistsError(f"project input artifact already exists: {existing}; pass force=true")

    frame, prepared_sha256 = _build_project_input_frame(
        prepared_dir,
        max_concepts=max_concepts,
    )
    if force and manifest_path.exists():
        manifest_path.unlink()
    atomic_write_parquet(pa.Table.from_pandas(frame, preserve_index=False), output_path)
    reloaded = pd.read_parquet(output_path)
    validate_project_input_frame(reloaded, max_concepts=max_concepts)
    if not frame.equals(reloaded):
        raise ValueError("project input Parquet round trip changed the frame")

    manifest = _manifest_payload(
        max_concepts=max_concepts,
        rows=len(frame),
        prepared_sha256=prepared_sha256,
        output_sha256=_sha256(output_path),
    )
    _require_prepared_unchanged(prepared_dir, prepared_sha256)
    atomic_json_dump(manifest, manifest_path)
    return frame


def write_project_input_from_dictconfig(cfg: DictConfig) -> pd.DataFrame:
    """Convert the thin Hydra config and call :func:`write_project_input`."""

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError("AxBench project-input config must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("AxBench project-input config keys must be strings")
    config = cast(Mapping[str, Any], raw)
    missing = sorted(_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - _CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(
            f"AxBench project-input config keys do not match; missing={missing}, unknown={unknown}"
        )
    return write_project_input(
        prepared_dir=Path(str(config["prepared_dir"])),
        output_path=Path(str(config["output_path"])),
        manifest_path=Path(str(config["manifest_path"])),
        max_concepts=config["max_concepts"],
        force=config["force"],
    )


__all__ = [
    "ALPACA_EVAL_FILE",
    "ALPACA_EVAL_REPO",
    "ALPACA_EVAL_REVISION",
    "AXBENCH_BENCHMARK_REPO",
    "AXBENCH_BENCHMARK_REVISION",
    "PROJECT_INPUT_FORMAT",
    "build_project_input_frame",
    "load_project_input_artifact",
    "write_project_input",
    "write_project_input_from_dictconfig",
]
