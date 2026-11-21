"""Falsification tests for strict AxBench-Upstream artifact comparison."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import steering.eval.axbench_upstream.comparison as comparison_module
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    NL_PROJECT_METHODS,
    FrameSource,
    method_columns,
    methods_for_source,
)
from steering.eval.axbench_upstream.comparison import (
    DEFAULT_OPERATIONAL_MANIFEST_PATHS,
    ArtifactComparisonError,
    compare_bundle,
    compare_encdec_bundle,
    compare_manifests,
    compare_parquet,
    compare_project_bundle,
    compare_safetensors,
)
from steering.immutable_io import ImmutableArtifactError
from steering.safetensors_io import deterministic_safetensors_bytes


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "concept_id": pd.Series([0, 0], dtype="int64"),
            "input_id": pd.Series([0, 0], dtype="int64"),
            "factor": pd.Series([0.0, 1.0], dtype="float64"),
            "generation": pd.Series(["alpha", "beta"], dtype="object"),
            "nullable": pd.Series([1, None], dtype="Int64"),
        }
    )


def _write_parquet(path: Path, frame: pd.DataFrame, *, compression: str = "snappy") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path, compression=compression)


def _write_tensor(
    path: Path,
    *,
    tensors: dict[str, t.Tensor] | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        deterministic_safetensors_bytes(
            tensors or {"direction": t.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=t.float32)},
            metadata=metadata or {"format": "comparison-test-v1"},
        )
    )


def _reserialize_safetensor(source: Path, destination: Path) -> None:
    """Write a distinct but valid serialization of the same safetensor payload."""

    raw = source.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    encoded = json.dumps(header, indent=1, ensure_ascii=False).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    destination.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_size :])


def _reorder_safetensor_header(source: Path, destination: Path) -> None:
    """Reorder JSON keys without changing header values or tensor payload."""

    raw = source.read_bytes()
    header_size = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8 : 8 + header_size].decode("utf-8"))
    metadata = header.get("__metadata__")
    if isinstance(metadata, dict):
        header["__metadata__"] = dict(reversed(list(metadata.items())))
    encoded = json.dumps(
        dict(reversed(list(header.items()))),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    destination.write_bytes(struct.pack("<Q", len(encoded)) + encoded + raw[8 + header_size :])


def _manifest(*, output: str, marker: str = "same") -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repository_git_commit": "commit-a",
        "repository_git_worktree": {"root": "/reference", "dirty": False},
        "configuration": {
            "output_path": output,
            "manifest_path": f"{output}.manifest.json",
            "main_path": "/semantic/input.parquet",
            "max_concepts": 1,
            "seed": 42,
        },
        "marker": marker,
    }


def _bundle_frame(
    source: FrameSource,
    *,
    methods: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    factors = (1.0,) if source is FrameSource.ENCDEC_AUX else FULL_FACTORS
    rows: list[dict[str, Any]] = []
    for input_id in range(10):
        for factor in factors:
            rows.append(
                {
                    "dataset_name": "AlpacaEval",
                    "concept_id": 0,
                    "input_concept": "test concept",
                    "input_id": input_id,
                    "factor": factor,
                    "original_prompt": f"original prompt {input_id}",
                    "steered_input": (
                        f"steered prompt {input_id}"
                        if source is FrameSource.ENCDEC_AUX
                        else f"input {input_id}"
                    ),
                    "input": f"input {input_id}",
                    "suppress_original": f"suppress original {input_id}",
                    "suppress_rewrite": f"suppress rewrite {input_id}",
                    "steered_prompt": f"steered prompt {input_id}",
                    "defense": "none",
                    "sae_link": "test-sae",
                    "sae_id": 0,
                }
            )
    frame = pd.DataFrame(rows, columns=INPUT_COLUMNS)
    frame["concept_id"] = frame["concept_id"].astype("int64")
    frame["input_id"] = frame["input_id"].astype("int64")
    frame["sae_id"] = frame["sae_id"].astype("int64")
    frame["factor"] = frame["factor"].astype("float64")
    methods = methods_for_source(source) if methods is None else methods
    for method in methods:
        generation_column, metric_column = method_columns((method,))
        frame[generation_column] = f"healthy generated text for {method}"
        frame[metric_column] = pd.Series(
            [float(index + 1) for index in range(len(frame))],
            dtype="float64",
        )
    return frame.loc[:, [*INPUT_COLUMNS, *method_columns(methods)]]


def _project_bundle(
    root: Path,
    *,
    marker: str = "same",
    include_nl_steering: bool = False,
) -> Path:
    output = root / "inference" / "steering_data.parquet"
    manifest = root / "manifest.json"
    methods = NL_PROJECT_METHODS if include_nl_steering else None
    _write_parquet(output, _bundle_frame(FrameSource.PROJECT, methods=methods))
    payload = _manifest(output=str(output), marker=marker)
    payload["configuration"]["manifest_path"] = str(manifest)
    _write_json(manifest, payload)
    return manifest


def _encdec_bundle(root: Path, *, marker: str = "same") -> Path:
    main = root / "main" / "inference" / "steering_data.parquet"
    aux = root / "aux" / "inference" / "steering_data.parquet"
    train = root / "train"
    manifest = root / "manifest.json"
    _write_parquet(main, _bundle_frame(FrameSource.ENCDEC_MAIN))
    _write_parquet(aux, _bundle_frame(FrameSource.ENCDEC_AUX))
    artifacts = []
    for method, suffix, tensor in (
        ("EncDecDiffMean", "diffmean", {"direction": t.arange(4, dtype=t.float32)}),
        (
            "EncDecLinearProbe",
            "linear_probe",
            {"direction": t.arange(4, dtype=t.float32) + 1},
        ),
        ("EncDecLoRA", "lora", {"adapter": t.arange(6, dtype=t.float32).reshape(2, 3)}),
    ):
        filename = f"concept_00000_{suffix}.safetensors"
        path = train / filename
        _write_tensor(path, tensors=tensor, metadata={"method": method})
        artifacts.append(
            {
                "path": filename,
                "method": method,
                "concept_id": 0,
                "sha256": _sha256(path),
            }
        )
    payload = {
        "schema_version": 1,
        "repository_git_commit": "commit-a",
        "repository_git_worktree": {"root": "/reference", "dirty": False},
        "configuration": {
            "output_main_path": str(main),
            "output_aux_path": str(aux),
            "manifest_path": str(manifest),
            "train_dir": str(train),
            "main_path": "/semantic/main.parquet",
            "max_concepts": 1,
            "seed": 42,
        },
        "train_artifacts": {"directory": str(train), "files": artifacts},
        "marker": marker,
    }
    _write_json(manifest, payload)
    return manifest


def _mutate_json(path: Path, mutate: Any) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    _write_json(path, payload)


def test_identical_parquet_copy_passes_with_hash_and_size(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame())
    shutil.copyfile(reference, candidate)

    result = compare_parquet(reference, candidate, label="test frame")

    assert result.sha256 == _sha256(reference) == _sha256(candidate)
    assert result.size_bytes == reference.stat().st_size
    assert result.reference_path == str(reference.resolve())
    assert result.candidate_path == str(candidate.resolve())


@pytest.mark.parametrize("alias_kind", ["same", "symlink", "hardlink"])
def test_parquet_comparison_rejects_reference_candidate_aliases(
    tmp_path: Path,
    alias_kind: str,
) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame())
    if alias_kind == "same":
        candidate = reference
    elif alias_kind == "symlink":
        candidate.symlink_to(reference)
    else:
        candidate.hardlink_to(reference)

    with pytest.raises(ArtifactComparisonError, match="must be distinct"):
        compare_parquet(reference, candidate, label="test frame")


def test_path_identity_oserror_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame())
    _write_parquet(candidate, _frame())

    def fail_samefile(_self: Path, _other: Path) -> bool:
        raise OSError("synthetic samefile failure")

    monkeypatch.setattr(Path, "samefile", fail_samefile)
    with pytest.raises(ArtifactComparisonError, match="could not compare test frame path"):
        compare_parquet(reference, candidate, label="test frame")


def test_descriptor_bound_snapshot_failure_is_wrapped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame())
    _write_parquet(candidate, _frame())
    monkeypatch.setattr(
        comparison_module,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ImmutableArtifactError("synthetic mutation")
        ),
    )

    with pytest.raises(ArtifactComparisonError, match="not a stable readable file"):
        compare_parquet(reference, candidate, label="test frame")


def test_bundle_grid_uses_the_already_compared_parquet_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")

    def reject_path_reopen(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("bundle comparison must not reopen a validated Parquet path")

    monkeypatch.setattr(pd, "read_parquet", reject_path_reopen)

    result = compare_project_bundle(reference, candidate)

    assert result.cell_count == 240


def test_project_bundle_comparator_requires_explicit_nl_selection(tmp_path: Path) -> None:
    reference = _project_bundle(tmp_path / "reference", include_nl_steering=True)
    candidate = _project_bundle(tmp_path / "candidate", include_nl_steering=True)

    with pytest.raises(ArtifactComparisonError, match="unknown method result columns.*NLSteering"):
        compare_project_bundle(reference, candidate)

    result = compare_project_bundle(reference, candidate, include_nl_steering=True)
    assert result.cell_count == 360


@pytest.mark.parametrize("changed_field", [None, "sha256", "source_texts_sha256", "embedding_space"])
@pytest.mark.parametrize("artifact", ["description_embeddings", "concept_embeddings"])
def test_nl_bundle_relocation_ignores_only_paths_and_preserves_source_identity(
    tmp_path: Path, artifact: str, changed_field: str | None,
) -> None:
    reference = _project_bundle(tmp_path / "reference", include_nl_steering=True)
    candidate = _project_bundle(tmp_path / "nested" / "candidate", include_nl_steering=True)
    shared = tmp_path / "shared-inputs"
    shared.mkdir()
    for name in ("description_embeddings", "concept_embeddings"):
        (shared / f"{name}.npz").write_bytes(f"byte-bound {name}".encode())
    for manifest in (reference, candidate):
        payload = json.loads(manifest.read_text())
        source = {
            name: {
                "path": os.path.relpath(shared / f"{name}.npz", manifest.parent),
                "sha256": _sha256(shared / f"{name}.npz"),
                "source_texts_sha256": "a" * 64,
                "embedding_space": {"model": "same-model", "revision": "b" * 40,
                                    "encoding": "same", "library_versions": {"encoder": "1"}},
            }
            for name in ("description_embeddings", "concept_embeddings")
        }
        payload["methods"] = {"NLSteering": {"target_source": source}}
        if manifest == candidate and changed_field is not None:
            source[artifact][changed_field] = (
                {**cast(dict[str, Any], source[artifact][changed_field]), "encoding": "different"}
                if changed_field == "embedding_space" else "c" * 64
            )
        _write_json(manifest, payload)
    if changed_field is None:
        assert compare_project_bundle(reference, candidate, include_nl_steering=True).cell_count == 360
        # The standalone/default comparator retains its original allowlist.
        with pytest.raises(ArtifactComparisonError, match="manifest semantic content differs"):
            compare_manifests(reference, candidate)
    else:
        with pytest.raises(ArtifactComparisonError, match="manifest semantic content differs"):
            compare_project_bundle(reference, candidate, include_nl_steering=True)


def test_missing_reference_parquet_fails_before_read(tmp_path: Path) -> None:
    with pytest.raises(
        ArtifactComparisonError, match="reference test frame is not a readable file"
    ):
        compare_parquet(tmp_path / "missing", tmp_path / "candidate", label="test frame")


def test_missing_candidate_parquet_fails_before_read(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    _write_parquet(reference, _frame())

    with pytest.raises(
        ArtifactComparisonError, match="candidate test frame is not a readable file"
    ):
        compare_parquet(reference, tmp_path / "missing", label="test frame")


def test_parquet_row_count_drift_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame())
    _write_parquet(candidate, _frame().iloc[:1])

    with pytest.raises(ArtifactComparisonError, match="row count differs: 2 != 1"):
        compare_parquet(reference, candidate, label="test frame")


def test_parquet_row_reorder_reports_first_column_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame())
    _write_parquet(candidate, _frame().iloc[::-1].reset_index(drop=True))

    with pytest.raises(
        ArtifactComparisonError,
        match=r"column 'factor' differs at index 0: reference=0.0, candidate=1.0",
    ):
        compare_parquet(reference, candidate, label="test frame")


def test_parquet_generation_cell_mutation_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    frame = _frame()
    mutated = frame.copy()
    mutated.loc[1, "generation"] = "changed"
    _write_parquet(reference, frame)
    _write_parquet(candidate, mutated)

    with pytest.raises(
        ArtifactComparisonError,
        match="column 'generation' differs at index 1: reference='beta', candidate='changed'",
    ):
        compare_parquet(reference, candidate, label="test frame")


def test_parquet_dtype_drift_is_a_schema_difference(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    mutated = _frame()
    mutated["concept_id"] = mutated["concept_id"].astype("int32")
    _write_parquet(reference, _frame())
    _write_parquet(candidate, mutated)

    with pytest.raises(ArtifactComparisonError, match="Parquet schema differs"):
        compare_parquet(reference, candidate, label="test frame")


def test_parquet_null_mask_drift_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    mutated = _frame()
    mutated.loc[1, "nullable"] = 2
    _write_parquet(reference, _frame())
    _write_parquet(candidate, mutated)

    with pytest.raises(
        ArtifactComparisonError,
        match="column 'nullable' differs at index 1: reference=None, candidate=2",
    ):
        compare_parquet(reference, candidate, label="test frame")


def test_logically_equal_parquet_with_different_encoding_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    _write_parquet(reference, _frame(), compression="snappy")
    _write_parquet(candidate, _frame(), compression="gzip")

    with pytest.raises(
        ArtifactComparisonError,
        match="logical tables are equal but serialized bytes differ",
    ):
        compare_parquet(reference, candidate, label="test frame")


def test_parquet_schema_metadata_drift_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.parquet"
    candidate = tmp_path / "candidate.parquet"
    table = pa.Table.from_pandas(_frame(), preserve_index=False)
    _write_parquet(reference, _frame())
    pq.write_table(table.replace_schema_metadata({b"campaign": b"changed"}), candidate)

    with pytest.raises(ArtifactComparisonError, match="Parquet schema differs"):
        compare_parquet(reference, candidate, label="test frame")


def test_identical_safetensor_copy_passes_with_hash_and_size(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference)
    shutil.copyfile(reference, candidate)

    result = compare_safetensors(reference, candidate, label="direction")

    assert result.sha256 == _sha256(reference)
    assert result.size_bytes == reference.stat().st_size


def test_safetensor_comparison_rejects_hardlink_alias(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference)
    candidate.hardlink_to(reference)

    with pytest.raises(ArtifactComparisonError, match="must be distinct"):
        compare_safetensors(reference, candidate, label="direction")


@pytest.mark.parametrize("corrupt_side", ["reference", "candidate"])
def test_malformed_safetensor_is_wrapped_as_artifact_comparison_error(
    tmp_path: Path,
    corrupt_side: str,
) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference)
    _write_tensor(candidate)
    (reference if corrupt_side == "reference" else candidate).write_bytes(b"not-safetensors")

    with pytest.raises(ArtifactComparisonError, match="not a valid strict safetensors pair"):
        compare_safetensors(reference, candidate, label="direction")


def test_legacy_header_key_order_is_normalized_against_canonical_candidate(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "legacy-reference.safetensors"
    candidate = tmp_path / "canonical-candidate.safetensors"
    _write_tensor(
        candidate,
        metadata={"zeta": "1", "alpha": "2", "mid": "3"},
    )
    _reorder_safetensor_header(candidate, reference)
    assert reference.read_bytes() != candidate.read_bytes()

    result = compare_safetensors(reference, candidate, label="direction")

    assert result.sha256 == _sha256(candidate)
    assert result.size_bytes == candidate.stat().st_size


def test_noncanonical_candidate_is_rejected_even_when_reference_bytes_match(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.safetensors"
    reference = tmp_path / "legacy-reference.safetensors"
    candidate = tmp_path / "legacy-candidate.safetensors"
    _write_tensor(canonical)
    _reorder_safetensor_header(canonical, reference)
    shutil.copyfile(reference, candidate)

    with pytest.raises(ArtifactComparisonError, match="candidate bytes are not canonical"):
        compare_safetensors(reference, candidate, label="direction")


def test_safetensor_metadata_drift_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference, metadata={"format": "v1"})
    _write_tensor(candidate, metadata={"format": "v2"})

    with pytest.raises(ArtifactComparisonError, match="safetensors metadata differs"):
        compare_safetensors(reference, candidate, label="direction")


def test_safetensor_key_drift_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference, tensors={"direction": t.ones(2)})
    _write_tensor(candidate, tensors={"adapter": t.ones(2)})

    with pytest.raises(ArtifactComparisonError, match="safetensors keys differ"):
        compare_safetensors(reference, candidate, label="direction")


def test_safetensor_dtype_drift_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference, tensors={"direction": t.ones(2, dtype=t.float32)})
    _write_tensor(candidate, tensors={"direction": t.ones(2, dtype=t.float64)})

    with pytest.raises(
        ArtifactComparisonError, match="dtype differs: torch.float32 != torch.float64"
    ):
        compare_safetensors(reference, candidate, label="direction")


def test_safetensor_shape_drift_is_explained(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference, tensors={"direction": t.ones(2, 2)})
    _write_tensor(candidate, tensors={"direction": t.ones(4)})

    with pytest.raises(ArtifactComparisonError, match=r"shape differs: \(2, 2\) != \(4,\)"):
        compare_safetensors(reference, candidate, label="direction")


def test_safetensor_value_drift_reports_exact_index_and_values(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    source = t.tensor([[1.0, 2.0], [3.0, 4.0]])
    mutated = source.clone()
    mutated[1, 0] = 99.0
    _write_tensor(reference, tensors={"direction": source})
    _write_tensor(candidate, tensors={"direction": mutated})

    with pytest.raises(
        ArtifactComparisonError,
        match=r"tensor 'direction' differs at \(1, 0\): reference=3.0, candidate=99.0",
    ):
        compare_safetensors(reference, candidate, label="direction")


def test_logically_equal_noncanonical_candidate_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.safetensors"
    candidate = tmp_path / "candidate.safetensors"
    _write_tensor(reference)
    _reserialize_safetensor(reference, candidate)

    with pytest.raises(
        ArtifactComparisonError,
        match="candidate bytes are not canonical",
    ):
        compare_safetensors(reference, candidate, label="direction")


def test_manifest_operational_allowlist_is_literal_and_sufficient(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference_payload = _manifest(output="/reference/output.parquet")
    candidate_payload = deepcopy(reference_payload)
    candidate_payload["repository_git_commit"] = "commit-b"
    candidate_payload["repository_git_worktree"] = {"root": "/candidate", "dirty": True}
    candidate_payload["configuration"]["output_path"] = "/candidate/output.parquet"
    candidate_payload["configuration"]["manifest_path"] = "/candidate/manifest.json"
    _write_json(reference, reference_payload)
    _write_json(candidate, candidate_payload)

    loaded_reference, loaded_candidate, reference_sha, candidate_sha = compare_manifests(
        reference, candidate
    )

    assert loaded_reference == reference_payload
    assert loaded_candidate == candidate_payload
    assert reference_sha == _sha256(reference)
    assert candidate_sha == _sha256(candidate)
    assert "configuration.main_path" not in DEFAULT_OPERATIONAL_MANIFEST_PATHS


def test_manifest_comparison_rejects_symlink_alias(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    _write_json(reference, _manifest(output="/reference/output.parquet"))
    candidate.symlink_to(reference)

    with pytest.raises(ArtifactComparisonError, match="must be distinct"):
        compare_manifests(reference, candidate)


def test_manifest_unallowlisted_configuration_drift_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference_payload = _manifest(output="/reference/output.parquet")
    candidate_payload = deepcopy(reference_payload)
    candidate_payload["configuration"]["main_path"] = "/different/input.parquet"
    _write_json(reference, reference_payload)
    _write_json(candidate, candidate_payload)

    with pytest.raises(
        ArtifactComparisonError,
        match=r"semantic content differs at \$.configuration.main_path",
    ):
        compare_manifests(reference, candidate)


def test_manifest_missing_key_is_rejected(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    payload = _manifest(output="/reference/output.parquet")
    _write_json(reference, payload)
    del payload["marker"]
    _write_json(candidate, payload)

    with pytest.raises(ArtifactComparisonError, match=r"missing=\['marker'\], unexpected=\[\]"):
        compare_manifests(reference, candidate)


@pytest.mark.parametrize("invalid", ["[]", "null", '"text"', "3"])
def test_manifest_top_level_must_be_an_object(tmp_path: Path, invalid: str) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(invalid, encoding="utf-8")
    candidate.write_text(invalid, encoding="utf-8")

    with pytest.raises(ArtifactComparisonError, match="must contain a JSON object"):
        compare_manifests(reference, candidate)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_manifest_rejects_nonfinite_json_constants(tmp_path: Path, constant: str) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text(f'{{"value": {constant}}}', encoding="utf-8")
    candidate.write_text('{"value": 1}', encoding="utf-8")

    with pytest.raises(ArtifactComparisonError, match="non-finite JSON constant"):
        compare_manifests(reference, candidate)


def test_manifest_rejects_duplicate_object_keys(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    candidate.write_text('{"schema_version": 1}', encoding="utf-8")

    with pytest.raises(ArtifactComparisonError, match="duplicate JSON object key"):
        compare_manifests(reference, candidate)


def test_custom_operational_allowlist_does_not_inherit_defaults(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    candidate = tmp_path / "candidate.json"
    reference_payload = _manifest(output="/reference/output.parquet", marker="a")
    candidate_payload = _manifest(output="/reference/output.parquet", marker="b")
    _write_json(reference, reference_payload)
    _write_json(candidate, candidate_payload)

    compare_manifests(reference, candidate, operational_paths=frozenset({"marker"}))
    with pytest.raises(ArtifactComparisonError, match="marker"):
        compare_manifests(reference, candidate, operational_paths=frozenset())


def test_project_bundle_exact_copy_reports_grid_digest_and_cell_count(tmp_path: Path) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")

    result = compare_project_bundle(reference, candidate)

    assert result.kind == "project"
    assert result.cell_count == 240
    assert len(result.cell_grid_sha256) == 64
    assert set(result.parquets) == {"project"}
    assert result.training_artifacts == {}
    assert result.to_dict()["parquets"]["project"]["sha256"] == _sha256(
        tmp_path / "reference" / "inference" / "steering_data.parquet"
    )


def test_project_bundle_dispatch_matches_direct_comparator(tmp_path: Path) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")

    assert compare_bundle("project", reference, candidate) == compare_project_bundle(
        reference, candidate
    )


@pytest.mark.parametrize(
    ("concept_ids", "error"),
    [
        ([1], None),
        (None, "exact key product"),
        ([0], "exact key product"),
        ([], "nonempty"),
        ([True], "global integers"),
        ([1, 1], "duplicates"),
        ([2], "global integers"),
        ([-1], "global integers"),
        ("1", "nonempty"),
    ],
)
def test_project_bundle_validates_declared_subset(
    tmp_path: Path, concept_ids: Any, error: str | None,
) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")
    for manifest in (reference, candidate):
        _mutate_json(
            manifest,
            lambda payload: payload["configuration"].update(
                max_concepts=2, concept_ids=concept_ids,
            ),
        )
        _write_parquet(
            manifest.parent / "inference/steering_data.parquet",
            _bundle_frame(FrameSource.PROJECT).assign(concept_id=1),
        )

    if error is not None:
        with pytest.raises(ArtifactComparisonError, match=error):
            compare_project_bundle(reference, candidate)
    else:
        assert compare_project_bundle(reference, candidate).cell_count == 240


def test_public_project_comparator_rejects_nonboolean_nl_selection(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="include_nl_steering must be a boolean"):
        compare_project_bundle(
            tmp_path / "reference.json",
            tmp_path / "candidate.json",
            include_nl_steering=cast(Any, 1),
        )


def test_project_bundle_generation_drift_fails(tmp_path: Path) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")
    candidate_output = tmp_path / "candidate" / "inference" / "steering_data.parquet"
    frame = _bundle_frame(FrameSource.PROJECT)
    frame.loc[0, "SelfConditioning_steered_generation"] = "not-the-oracle"
    _write_parquet(candidate_output, frame)

    with pytest.raises(ArtifactComparisonError, match="steered_generation.*not-the-oracle"):
        compare_project_bundle(reference, candidate)


def test_project_bundle_rejects_candidate_pointing_at_reference_output(tmp_path: Path) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")
    reference_output = tmp_path / "reference" / "inference" / "steering_data.parquet"
    _mutate_json(
        candidate,
        lambda payload: payload["configuration"].__setitem__(
            "output_path",
            str(reference_output),
        ),
    )

    with pytest.raises(ArtifactComparisonError, match="project output must be distinct"):
        compare_project_bundle(reference, candidate)


def test_identically_truncated_project_bundles_fail_production_completeness(
    tmp_path: Path,
) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")
    truncated = _bundle_frame(FrameSource.PROJECT).iloc[:-1].reset_index(drop=True)
    _write_parquet(tmp_path / "reference" / "inference" / "steering_data.parquet", truncated)
    _write_parquet(tmp_path / "candidate" / "inference" / "steering_data.parquet", truncated)

    with pytest.raises(ArtifactComparisonError, match="exact key product"):
        compare_project_bundle(reference, candidate)


def test_project_bundle_seed_drift_fails_at_manifest(tmp_path: Path) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")
    _mutate_json(candidate, lambda payload: payload["configuration"].__setitem__("seed", 43))

    with pytest.raises(ArtifactComparisonError, match=r"\$.configuration.seed"):
        compare_project_bundle(reference, candidate)


@pytest.mark.parametrize("invalid", [None, 0, True])
def test_project_bundle_requires_positive_manifest_concept_count(
    tmp_path: Path,
    invalid: Any,
) -> None:
    reference = _project_bundle(tmp_path / "reference")
    candidate = _project_bundle(tmp_path / "candidate")
    for manifest in (reference, candidate):
        _mutate_json(
            manifest,
            lambda payload: payload["configuration"].__setitem__("max_concepts", invalid),
        )

    with pytest.raises(ArtifactComparisonError, match="max_concepts must be a positive int"):
        compare_project_bundle(reference, candidate)


def test_encdec_bundle_exact_copy_checks_every_artifact(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")

    result = compare_encdec_bundle(reference, candidate)

    assert result.kind == "encdec"
    assert result.cell_count == 260
    assert set(result.parquets) == {"main", "aux"}
    assert len(result.training_artifacts) == 3
    assert set(result.training_artifacts) == {
        "0:EncDecDiffMean:concept_00000_diffmean.safetensors",
        "0:EncDecLinearProbe:concept_00000_linear_probe.safetensors",
        "0:EncDecLoRA:concept_00000_lora.safetensors",
    }


def test_identically_truncated_encdec_bundles_fail_production_completeness(
    tmp_path: Path,
) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    truncated = _bundle_frame(FrameSource.ENCDEC_MAIN).iloc[:-1].reset_index(drop=True)
    for root in (tmp_path / "reference", tmp_path / "candidate"):
        _write_parquet(root / "main" / "inference" / "steering_data.parquet", truncated)

    with pytest.raises(ArtifactComparisonError, match="exact key product"):
        compare_encdec_bundle(reference, candidate)


def test_identical_aux_main_input_mismatch_is_rejected_per_bundle(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    mismatched_aux = _bundle_frame(FrameSource.ENCDEC_AUX)
    mismatched_aux.loc[0, "original_prompt"] = "different but internally valid prompt"
    for root in (tmp_path / "reference", tmp_path / "candidate"):
        _write_parquet(root / "aux" / "inference" / "steering_data.parquet", mismatched_aux)

    with pytest.raises(ArtifactComparisonError, match="aux inputs do not match main"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_bundle_rejects_aliased_training_directories(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    reference_train = tmp_path / "reference" / "train"
    _mutate_json(
        candidate,
        lambda payload: payload["train_artifacts"].__setitem__(
            "directory",
            str(reference_train),
        ),
    )

    with pytest.raises(ArtifactComparisonError, match="training directory must be distinct"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_bundle_accepts_verified_legacy_raw_digests_against_canonical_candidate(
    tmp_path: Path,
) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    reference_payload = json.loads(reference.read_text(encoding="utf-8"))
    reference_train = tmp_path / "reference" / "train"
    candidate_train = tmp_path / "candidate" / "train"
    for entry in reference_payload["train_artifacts"]["files"]:
        relative_path = entry["path"]
        reference_artifact = reference_train / relative_path
        _reorder_safetensor_header(reference_artifact, reference_artifact)
        entry["sha256"] = _sha256(reference_artifact)
        assert entry["sha256"] != _sha256(candidate_train / relative_path)
    _write_json(reference, reference_payload)

    result = compare_encdec_bundle(reference, candidate)

    assert len(result.training_artifacts) == 3
    assert {artifact.sha256 for artifact in result.training_artifacts.values()} == {
        _sha256(path) for path in candidate_train.glob("*.safetensors")
    }


def test_encdec_bundle_dispatch_matches_direct_comparator(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")

    assert compare_bundle("encdec", reference, candidate) == compare_encdec_bundle(
        reference, candidate
    )


def test_bundle_dispatch_rejects_nl_selection_for_encdec(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="supported only for project bundles"):
        compare_bundle(
            "encdec",
            tmp_path / "reference.json",
            tmp_path / "candidate.json",
            include_nl_steering=True,
        )


def test_bundle_dispatch_rejects_nonboolean_nl_selection(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="include_nl_steering must be a boolean"):
        compare_bundle(
            "encdec",
            tmp_path / "reference.json",
            tmp_path / "candidate.json",
            include_nl_steering=cast(Any, 1),
        )


def test_unknown_bundle_kind_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown AxBench-Upstream bundle kind"):
        compare_bundle(cast(Any, "wrong"), tmp_path / "a", tmp_path / "b")


def test_encdec_training_tensor_mutation_fails_even_if_manifests_match(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    artifact = tmp_path / "candidate" / "train" / "concept_00000_diffmean.safetensors"
    _write_tensor(
        artifact,
        tensors={"direction": t.tensor([0.0, 1.0, 2.0, 99.0])},
        metadata={"method": "EncDecDiffMean"},
    )

    with pytest.raises(ArtifactComparisonError, match="tensor 'direction' differs"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_declared_digest_drift_fails_at_manifest(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    _mutate_json(
        candidate,
        lambda payload: payload["train_artifacts"]["files"][0].__setitem__("sha256", "0" * 64),
    )

    with pytest.raises(ArtifactComparisonError, match="sha256"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_reference_declared_digest_must_match_legacy_raw_file(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    _mutate_json(
        reference,
        lambda payload: payload["train_artifacts"]["files"][0].__setitem__("sha256", "0" * 64),
    )

    with pytest.raises(ArtifactComparisonError, match="reference .* raw sha256"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_undeclared_manifest_drift_still_fails_after_digest_normalization(
    tmp_path: Path,
) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate", marker="changed")

    with pytest.raises(ArtifactComparisonError, match=r"semantic content differs at \$.marker"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_extra_training_file_fails_exact_directory_inventory(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    (tmp_path / "candidate" / "train" / "unexpected.txt").write_text("extra", encoding="utf-8")

    with pytest.raises(ArtifactComparisonError, match="directory inventory differs"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_same_extra_training_file_fails_each_bundle_independently(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    for root in (tmp_path / "reference", tmp_path / "candidate"):
        (root / "train" / "unexpected.txt").write_text("same extra", encoding="utf-8")

    with pytest.raises(
        ArtifactComparisonError,
        match="reference training directory inventory differs",
    ):
        compare_encdec_bundle(reference, candidate)


def test_encdec_same_missing_method_identity_fails_exact_grid(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    for manifest in (reference, candidate):
        _mutate_json(
            manifest,
            lambda payload: payload["train_artifacts"]["files"].pop(),
        )

    with pytest.raises(ArtifactComparisonError, match="exact 3 x 1 grid"):
        compare_encdec_bundle(reference, candidate)


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_encdec_training_directory_rejects_nonfile_entries(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    entry = tmp_path / "reference" / "train" / "unexpected"
    if entry_kind == "directory":
        entry.mkdir()
        match = "non-file entry"
    else:
        entry.symlink_to(tmp_path / "reference" / "train" / "concept_00000_lora.safetensors")
        match = "contains a symlink"

    with pytest.raises(ArtifactComparisonError, match=match):
        compare_encdec_bundle(reference, candidate)


def test_encdec_missing_training_file_fails_as_unreadable(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    (tmp_path / "candidate" / "train" / "concept_00000_lora.safetensors").unlink()

    with pytest.raises(ArtifactComparisonError, match="candidate training directory inventory"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_missing_training_directory_fails_cleanly(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    shutil.rmtree(tmp_path / "candidate" / "train")

    with pytest.raises(ArtifactComparisonError, match="not a readable directory"):
        compare_encdec_bundle(reference, candidate)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda entry: entry.__setitem__("concept_id", True), "invalid identity"),
        (lambda entry: entry.__setitem__("sha256", "g" * 64), "invalid identity"),
        (lambda entry: entry.__setitem__("sha256", "0" * 63), "invalid identity"),
        (lambda entry: entry.__setitem__("method", "Wrong"), "expected None"),
        (lambda entry: entry.__setitem__("path", "../escape.safetensors"), "does not match"),
        (lambda entry: entry.__setitem__("path", "/tmp/escape.safetensors"), "does not match"),
    ],
)
def test_encdec_rejects_malformed_training_inventory(
    tmp_path: Path,
    mutation: Any,
    match: str,
) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    _mutate_json(reference, lambda payload: mutation(payload["train_artifacts"]["files"][0]))
    _mutate_json(candidate, lambda payload: mutation(payload["train_artifacts"]["files"][0]))

    with pytest.raises(ArtifactComparisonError, match=match):
        compare_encdec_bundle(reference, candidate)


def test_encdec_rejects_duplicate_training_identity(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")

    def duplicate(payload: dict[str, Any]) -> None:
        payload["train_artifacts"]["files"].append(deepcopy(payload["train_artifacts"]["files"][0]))

    _mutate_json(reference, duplicate)
    _mutate_json(candidate, duplicate)

    with pytest.raises(ArtifactComparisonError, match="repeats training artifact path"):
        compare_encdec_bundle(reference, candidate)


def test_encdec_rejects_nonlist_training_inventory(tmp_path: Path) -> None:
    reference = _encdec_bundle(tmp_path / "reference")
    candidate = _encdec_bundle(tmp_path / "candidate")
    for path in (reference, candidate):
        _mutate_json(path, lambda payload: payload["train_artifacts"].__setitem__("files", {}))

    with pytest.raises(ArtifactComparisonError, match="string directory and list files"):
        compare_encdec_bundle(reference, candidate)
