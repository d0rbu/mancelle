"""Contracts for the model-free AxBench project input producer."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from omegaconf import OmegaConf

from scripts.eval.axbench_upstream.build_project_input import main as build_project_input_main
from scripts.eval.axbench_upstream.merge_methods import (
    FULL_FACTORS,
    INPUT_COLUMNS,
    validate_project_input_frame,
)
from scripts.eval.axbench_upstream.run_baselines import AXBENCH_UPSTREAM_COMMIT
from steering.eval.axbench_upstream import project_input as producer
from steering.eval.axbench_upstream.project_input import (
    ALPACA_EVAL_FILE,
    ALPACA_EVAL_REPO,
    ALPACA_EVAL_REVISION,
    AXBENCH_BENCHMARK_REPO,
    AXBENCH_BENCHMARK_REVISION,
    PROJECT_INPUT_FORMAT,
    build_project_input_frame,
    load_project_input_artifact,
    write_project_input,
    write_project_input_from_dictconfig,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _prepared(tmp_path: Path, *, concepts: int = 2) -> Path:
    prepared = tmp_path / "prepared"
    (prepared / "generate").mkdir(parents=True)
    (prepared / "data").mkdir()
    metadata = [
        {
            "concept_id": concept_id,
            "concept": f"concept-{concept_id}",
            "ref": f"https://example.test/sae/{100 + concept_id}",
        }
        for concept_id in range(concepts)
    ]
    metadata_path = prepared / "generate" / "metadata.jsonl"
    metadata_path.write_text(
        "".join(json.dumps(row) + "\n" for row in metadata),
        encoding="utf-8",
    )
    alpaca_path = prepared / "data" / "alpaca_eval.json"
    alpaca_path.write_text(
        json.dumps(
            [
                {
                    "instruction": f"ordinary question mentioning Qwen number {index}"
                }
                for index in range(20)
            ]
        ),
        encoding="utf-8",
    )
    (prepared / "prepare_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
                "benchmark": {
                    "repo": AXBENCH_BENCHMARK_REPO,
                    "revision": AXBENCH_BENCHMARK_REVISION,
                },
                "alpaca_eval": {
                    "repo": ALPACA_EVAL_REPO,
                    "revision": ALPACA_EVAL_REVISION,
                    "file": ALPACA_EVAL_FILE,
                },
                "artifacts": {
                    "generate/metadata.jsonl": {
                        "bytes": metadata_path.stat().st_size,
                        "sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
                    },
                    "data/alpaca_eval.json": {
                        "bytes": alpaca_path.stat().st_size,
                        "sha256": hashlib.sha256(alpaca_path.read_bytes()).hexdigest(),
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return prepared


def _refresh_artifact_identity(prepared: Path, relative: str) -> None:
    manifest_path = prepared / "prepare_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source = prepared / relative
    manifest["artifacts"][relative] = {
        "bytes": source.stat().st_size,
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _published(prepared: Path, root: Path) -> tuple[Path, Path, pd.DataFrame]:
    output = root / "steering_input.parquet"
    manifest = root / "manifest.json"
    frame = write_project_input(
        prepared_dir=prepared,
        output_path=output,
        manifest_path=manifest,
        max_concepts=2,
    )
    return output, manifest, frame


def test_pinned_source_constants_match_the_vendored_preparer() -> None:
    tree = ast.parse((_REPO_ROOT / "vendor" / "axbench" / "prepare.py").read_text())
    assignments = {
        target.id: ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {
            "BENCHMARK_REPO",
            "BENCHMARK_REVISION",
            "ALPACA_REPO",
            "ALPACA_REVISION",
            "ALPACA_FILE",
        }
    }
    assert assignments == {
        "BENCHMARK_REPO": AXBENCH_BENCHMARK_REPO,
        "BENCHMARK_REVISION": AXBENCH_BENCHMARK_REVISION,
        "ALPACA_REPO": ALPACA_EVAL_REPO,
        "ALPACA_REVISION": ALPACA_EVAL_REVISION,
        "ALPACA_FILE": ALPACA_EVAL_FILE,
    }


@pytest.mark.slow
def test_root_producer_exactly_matches_the_frozen_vendored_factory(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    root_frame = build_project_input_frame(prepared, max_concepts=2)
    oracle_path = tmp_path / "vendor-oracle.parquet"
    program = f'''
import json
import sys
from pathlib import Path

import pandas as pd
from axbench.utils.dataset import SteeringDatasetFactory

prepared = Path(sys.argv[1])
output = Path(sys.argv[2])
metadata = [json.loads(line) for line in (prepared / "generate/metadata.jsonl").read_text().splitlines()]

class Tokenizer:
    chat_template = None

factory = SteeringDatasetFactory(
    tokenizer=Tokenizer(),
    dump_dir=str(output.parent),
    master_data_dir=str(prepared / "data"),
)
parts = []
for concept_id, item in enumerate(metadata[:2]):
    frame = factory.create_eval_df(
        concepts=[item["concept"]],
        subset_n=10,
        steering_factors={list(FULL_FACTORS)!r},
        steering_datasets=["AlpacaEval"],
        concept_id=concept_id,
        steering_model_name="raw-model",
        steer_data_type="concept",
        defense=[],
    )
    frame["concept_id"] = concept_id
    frame["defense"] = frame["defense"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
    )
    frame["sae_link"] = item["ref"]
    frame["sae_id"] = int(item["ref"].rsplit("/", 1)[-1])
    parts.append(frame)
result = pd.concat(parts, ignore_index=True).loc[:, {list(INPUT_COLUMNS)!r}]
for column in ("concept_id", "input_id", "sae_id"):
    result[column] = result[column].astype("int64")
result["factor"] = result["factor"].astype("float64")
result.to_parquet(output, index=False)
'''
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(_REPO_ROOT / "vendor" / "axbench")
    result = subprocess.run(
        [
            "uv",
            "run",
            "--project",
            str(_REPO_ROOT / "vendor" / "axbench"),
            "--frozen",
            "--offline",
            "python",
            "-c",
            program,
            str(prepared),
            str(oracle_path),
        ],
        cwd=_REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert root_frame.equals(pd.read_parquet(oracle_path))


def test_builds_exact_model_free_grid_without_qwen_framing(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)

    frame = build_project_input_frame(prepared, max_concepts=2)

    validate_project_input_frame(frame, max_concepts=2)
    assert tuple(frame.columns) == INPUT_COLUMNS
    assert len(frame) == 2 * 10 * len(FULL_FACTORS)
    assert frame["input"].equals(frame["original_prompt"])
    assert frame.astype(str).apply(
        lambda column: column.str.contains("Qwen", regex=False).any()
    ).any()
    expected_prompts = (
        pd.read_json(prepared / "data" / "alpaca_eval.json")
        .sample(10, random_state=0)["instruction"]
        .tolist()
    )
    observed_prompts = (
        frame.loc[frame["concept_id"].eq(0), ["input_id", "original_prompt"]]
        .drop_duplicates()
        .sort_values("input_id")["original_prompt"]
        .tolist()
    )
    assert observed_prompts == expected_prompts


def test_five_concepts_have_ten_independent_inputs_at_every_factor(tmp_path: Path) -> None:
    frame = build_project_input_frame(_prepared(tmp_path, concepts=5), max_concepts=5)

    validate_project_input_frame(frame, max_concepts=5)
    assert len(frame) == 5 * 10 * len(FULL_FACTORS) == 600
    assert frame.groupby("factor", sort=False).size().eq(5 * 10).all()
    assert frame.groupby(["concept_id", "factor"], sort=False).size().eq(10).all()
    assert frame[["concept_id", "input_id"]].drop_duplicates().shape[0] == 5 * 10


def test_project_input_validator_rejects_baseline_result_columns(tmp_path: Path) -> None:
    frame = build_project_input_frame(_prepared(tmp_path), max_concepts=2)
    frame["DiffMean_steered_generation"] = "not part of the input contract"

    with pytest.raises(ValueError, match="unknown method result.*DiffMean_steered_generation"):
        validate_project_input_frame(frame, max_concepts=2)


def test_atomic_writer_publishes_manifest_last_and_roundtrips(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    output = tmp_path / "artifact" / "steering_input.parquet"
    manifest_path = tmp_path / "artifact" / "manifest.json"

    frame = write_project_input(
        prepared_dir=prepared,
        output_path=output,
        manifest_path=manifest_path,
        max_concepts=2,
    )

    assert frame.equals(pd.read_parquet(output))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == PROJECT_INPUT_FORMAT
    assert manifest["source_kind"] == "model_free_project_input"
    assert manifest["model_loaded"] is False
    assert manifest["model_generations_present"] is False
    assert manifest["rows"] == len(frame)
    loaded, input_sha256, manifest_sha256 = load_project_input_artifact(
        input_path=output,
        manifest_path=manifest_path,
        prepared_dir=prepared,
        max_concepts=2,
    )
    assert loaded.equals(frame)
    assert input_sha256 == hashlib.sha256(output.read_bytes()).hexdigest()
    assert manifest_sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="pass force=true"):
        write_project_input(
            prepared_dir=prepared,
            output_path=output,
            manifest_path=manifest_path,
            max_concepts=2,
        )


@pytest.mark.parametrize("max_concepts", [True, False, 0, -1, 1.5])
def test_max_concepts_must_be_a_positive_plain_integer(
    tmp_path: Path,
    max_concepts: object,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        build_project_input_frame(
            _prepared(tmp_path),
            max_concepts=cast(Any, max_concepts),
        )


def test_requested_concept_prefix_must_exist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exceeds 2 metadata rows"):
        build_project_input_frame(_prepared(tmp_path), max_concepts=3)


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ("[]", "expected a JSON object"),
        ('{"schema_version": 1, "schema_version": 1}', "duplicate JSON object key"),
        ('{"schema_version": NaN}', "non-finite JSON constant"),
        ("{", "readable strict JSON object"),
    ],
)
def test_prepared_manifest_is_strict_json(
    tmp_path: Path,
    payload: str,
    expected: str,
) -> None:
    prepared = _prepared(tmp_path)
    (prepared / "prepare_manifest.json").write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        build_project_input_frame(prepared, max_concepts=2)


def test_prepared_manifest_must_be_readable(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    (prepared / "prepare_manifest.json").unlink()

    with pytest.raises(ValueError, match="readable strict JSON object"):
        build_project_input_frame(prepared, max_concepts=2)


@pytest.mark.parametrize(
    ("mutation", "expected"),
    [
        (("schema_version", True), "schema_version"),
        (("schema_version", 2), "schema_version"),
        (("axbench_upstream_commit", "wrong"), "axbench_upstream_commit"),
        (("benchmark", None), "benchmark must be an object"),
        (("benchmark.revision", " "), "benchmark field 'revision'"),
        (("benchmark.repo", "wrong"), "benchmark field 'repo'"),
        (("alpaca_eval", None), "alpaca_eval must be an object"),
        (("alpaca_eval.revision", "wrong"), "alpaca_eval field 'revision'"),
        (("alpaca_eval.file", "wrong.json"), "alpaca_eval field 'file'"),
        (("artifacts", None), "artifacts must be an object"),
    ],
)
def test_prepared_manifest_identity_is_exact(
    tmp_path: Path,
    mutation: tuple[str, object],
    expected: str,
) -> None:
    prepared = _prepared(tmp_path)
    path = prepared / "prepare_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    dotted, value = mutation
    target = manifest
    parts = dotted.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        build_project_input_frame(prepared, max_concepts=2)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("entry", None, "no artifact identity"),
        ("sha256", "bad", "valid sha256"),
        ("bytes", True, "valid byte count"),
        ("bytes", -1, "valid byte count"),
    ],
)
def test_prepared_artifact_identity_is_well_formed(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    prepared = _prepared(tmp_path)
    path = prepared / "prepare_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    relative = "generate/metadata.jsonl"
    if field == "entry":
        manifest["artifacts"][relative] = value
    else:
        manifest["artifacts"][relative][field] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        build_project_input_frame(prepared, max_concepts=2)


@pytest.mark.parametrize("relative", ["generate/metadata.jsonl", "data/alpaca_eval.json"])
def test_prepared_source_drift_or_absence_is_rejected(
    tmp_path: Path,
    relative: str,
) -> None:
    prepared = _prepared(tmp_path)
    source = prepared / relative
    source.write_bytes(source.read_bytes() + b" ")
    with pytest.raises(ValueError, match="does not match prepare_manifest"):
        build_project_input_frame(prepared, max_concepts=2)

    source.unlink()
    with pytest.raises(FileNotFoundError, match="source artifact does not exist"):
        build_project_input_frame(prepared, max_concepts=2)


def test_prepared_identity_detects_manifest_replacement_with_equal_json(
    tmp_path: Path,
) -> None:
    prepared = _prepared(tmp_path)
    prepared_sha256 = producer._validated_prepared_sha256(prepared)
    manifest = prepared / "prepare_manifest.json"
    manifest.write_bytes(manifest.read_bytes() + b" ")

    with pytest.raises(ValueError, match="sources changed while"):
        producer._require_prepared_unchanged(prepared, prepared_sha256)


@pytest.mark.parametrize(
    ("concept", "reference", "expected"),
    [
        (None, "https://example.test/sae/100", "no nonempty concept"),
        ("", "https://example.test/sae/100", "no nonempty concept"),
        ("ok", None, "no nonempty ref"),
        ("ok", "https://example.test/sae/not-an-id", "does not end in an integer"),
        ("ok", "https://example.test/sae/-1", "must be nonnegative"),
    ],
)
def test_metadata_semantics_are_strict(
    tmp_path: Path,
    concept: object,
    reference: object,
    expected: str,
) -> None:
    prepared = _prepared(tmp_path)
    metadata = prepared / "generate" / "metadata.jsonl"
    rows = [json.loads(line) for line in metadata.read_text(encoding="utf-8").splitlines()]
    rows[0]["concept"] = concept
    rows[0]["ref"] = reference
    metadata.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    _refresh_artifact_identity(prepared, "generate/metadata.jsonl")

    with pytest.raises(ValueError, match=expected):
        build_project_input_frame(prepared, max_concepts=2)


@pytest.mark.parametrize(
    ("rows", "expected"),
    [
        ([{"wrong": "column"}] * 10, "at least 10 instructions"),
        ([{"instruction": f"ok-{index}"} for index in range(9)], "at least 10 instructions"),
        (
            [{"instruction": 3}, *({"instruction": f"ok-{index}"} for index in range(10))],
            "instructions must all be strings",
        ),
    ],
)
def test_alpaca_instruction_schema_is_strict(
    tmp_path: Path,
    rows: list[dict[str, object]],
    expected: str,
) -> None:
    prepared = _prepared(tmp_path)
    alpaca = prepared / "data" / "alpaca_eval.json"
    alpaca.write_text(json.dumps(rows), encoding="utf-8")
    _refresh_artifact_identity(prepared, "data/alpaca_eval.json")

    with pytest.raises(ValueError, match=expected):
        build_project_input_frame(prepared, max_concepts=2)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("schema_version", True, "schema_version"),
        ("format", "wrong", "format"),
        ("source_kind", "wrong", "source_kind"),
        ("max_concepts", 1, "max_concepts"),
        ("inputs_per_concept", 9, "inputs_per_concept"),
        ("rows", 0, "rows"),
        ("model_loaded", True, "model_loaded"),
        ("model_generations_present", True, "model_generations_present"),
        ("generation_prompt_column", "input", "generation_prompt_column"),
        ("factors", [0.0], "factors"),
        ("output_sha256", None, "output_sha256"),
        ("prepared_manifest_sha256", None, "prepared_manifest_sha256"),
    ],
)
def test_consumer_rejects_every_manifest_contract_corruption(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    prepared = _prepared(tmp_path)
    output, manifest_path, _ = _published(prepared, tmp_path / "artifact")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected):
        load_project_input_artifact(
            input_path=output,
            manifest_path=manifest_path,
            prepared_dir=prepared,
            max_concepts=2,
        )


def test_consumer_rejects_swapped_parquet_and_same_path(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    output, manifest, frame = _published(prepared, tmp_path / "artifact")
    changed = frame.copy()
    first_input = changed["concept_id"].eq(0) & changed["input_id"].eq(0)
    changed.loc[first_input, "original_prompt"] = "changed"
    changed.loc[first_input, "input"] = "changed"
    changed.loc[first_input, "steered_input"] = "changed"
    changed.to_parquet(output, index=False)

    with pytest.raises(ValueError, match="output_sha256"):
        load_project_input_artifact(
            input_path=output,
            manifest_path=manifest,
            prepared_dir=prepared,
            max_concepts=2,
        )
    with pytest.raises(ValueError, match="paths must be distinct"):
        load_project_input_artifact(
            input_path=output,
            manifest_path=output,
            prepared_dir=prepared,
            max_concepts=2,
        )


@pytest.mark.parametrize("artifact_in_root", ["input", "manifest"])
def test_consumer_rejects_artifacts_inside_prepared_source_tree(
    tmp_path: Path,
    artifact_in_root: str,
) -> None:
    prepared = _prepared(tmp_path)
    input_path = tmp_path / "input.parquet"
    manifest_path = tmp_path / "manifest.json"
    if artifact_in_root == "input":
        input_path = prepared / "published" / "input.parquet"
    else:
        manifest_path = prepared / "published" / "manifest.json"

    with pytest.raises(ValueError, match="outside the prepared source directory"):
        load_project_input_artifact(
            input_path=input_path,
            manifest_path=manifest_path,
            prepared_dir=prepared,
            max_concepts=2,
        )


def test_consumer_detects_parquet_and_manifest_replacement_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    output, manifest, _ = _published(prepared, tmp_path / "artifact")
    real_read = pd.read_parquet

    def replace_parquet(path: Path) -> pd.DataFrame:
        frame = real_read(path)
        Path(path).write_bytes(Path(path).read_bytes() + b"changed")
        return frame

    monkeypatch.setattr(producer.pd, "read_parquet", replace_parquet)
    with pytest.raises(ValueError, match="Parquet changed"):
        load_project_input_artifact(
            input_path=output,
            manifest_path=manifest,
            prepared_dir=prepared,
            max_concepts=2,
        )

    monkeypatch.setattr(producer.pd, "read_parquet", real_read)
    output, manifest, _ = _published(prepared, tmp_path / "artifact-2")
    real_validate = producer._validate_project_input_manifest

    def replace_manifest(
        manifest_payload: Mapping[str, Any],
        *,
        input_sha256: str,
        prepared_dir: Path,
        max_concepts: int,
        row_count: int,
    ) -> object:
        result = real_validate(
            manifest_payload,
            input_sha256=input_sha256,
            prepared_dir=prepared_dir,
            max_concepts=max_concepts,
            row_count=row_count,
        )
        manifest.write_bytes(manifest.read_bytes() + b" ")
        return result

    monkeypatch.setattr(producer, "_validate_project_input_manifest", replace_manifest)
    with pytest.raises(ValueError, match="manifest changed"):
        load_project_input_artifact(
            input_path=output,
            manifest_path=manifest,
            prepared_dir=prepared,
            max_concepts=2,
        )


def test_consumer_detects_prepared_source_replacement_during_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    output, manifest, _ = _published(prepared, tmp_path / "artifact")
    real_validate = producer._validate_project_input_manifest

    def replace_source(
        manifest_payload: Mapping[str, Any],
        *,
        input_sha256: str,
        prepared_dir: Path,
        max_concepts: int,
        row_count: int,
    ) -> object:
        result = real_validate(
            manifest_payload,
            input_sha256=input_sha256,
            prepared_dir=prepared_dir,
            max_concepts=max_concepts,
            row_count=row_count,
        )
        source = prepared_dir / "data" / "alpaca_eval.json"
        source.write_bytes(source.read_bytes() + b" ")
        return result

    monkeypatch.setattr(producer, "_validate_project_input_manifest", replace_source)
    with pytest.raises(ValueError, match="does not match prepare_manifest"):
        load_project_input_artifact(
            input_path=output,
            manifest_path=manifest,
            prepared_dir=prepared,
            max_concepts=2,
        )


@pytest.mark.parametrize("force", [None, 0, 1, "true"])
def test_force_must_be_a_boolean(tmp_path: Path, force: object) -> None:
    with pytest.raises(ValueError, match="force must be a boolean"):
        write_project_input(
            prepared_dir=_prepared(tmp_path),
            output_path=tmp_path / "output.parquet",
            manifest_path=tmp_path / "manifest.json",
            max_concepts=2,
            force=cast(Any, force),
        )


@pytest.mark.parametrize(
    "alias",
    ["same", "manifest-input", "output-input", "manifest-in-root", "output-in-root"],
)
def test_writer_rejects_path_aliases(tmp_path: Path, alias: str) -> None:
    prepared = _prepared(tmp_path)
    output = tmp_path / "output.parquet"
    manifest = tmp_path / "manifest.json"
    if alias == "same":
        manifest = output
    elif alias == "manifest-input":
        manifest = prepared / "prepare_manifest.json"
    elif alias == "output-input":
        output = prepared / "data" / "alpaca_eval.json"
    elif alias == "manifest-in-root":
        manifest = prepared / "published" / "manifest.json"
    else:
        output = prepared / "published" / "output.parquet"

    with pytest.raises(ValueError, match="paths must be disjoint"):
        write_project_input(
            prepared_dir=prepared,
            output_path=output,
            manifest_path=manifest,
            max_concepts=2,
        )


def test_force_replaces_both_artifacts_and_manifest_is_published_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    output, manifest, first = _published(prepared, tmp_path / "artifact")
    output.write_bytes(b"old parquet")
    manifest.write_text("old manifest", encoding="utf-8")
    write_project_input(
        prepared_dir=prepared,
        output_path=output,
        manifest_path=manifest,
        max_concepts=2,
        force=True,
    )
    assert pd.read_parquet(output).equals(first)
    assert json.loads(manifest.read_text(encoding="utf-8"))["format"] == PROJECT_INPUT_FORMAT

    real_json_dump = producer.atomic_json_dump

    def fail_manifest(*_args: object, **_kwargs: object) -> None:
        assert output.is_file()
        raise RuntimeError("manifest publication failed")

    monkeypatch.setattr(producer, "atomic_json_dump", fail_manifest)
    with pytest.raises(RuntimeError, match="publication failed"):
        write_project_input(
            prepared_dir=prepared,
            output_path=output,
            manifest_path=manifest,
            max_concepts=2,
            force=True,
        )
    assert output.is_file()
    assert not manifest.exists()
    monkeypatch.setattr(producer, "atomic_json_dump", real_json_dump)


def test_writer_refuses_to_publish_if_prepared_sources_change_after_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    output = tmp_path / "artifact" / "output.parquet"
    manifest = tmp_path / "artifact" / "manifest.json"
    real_write = producer.atomic_write_parquet

    def write_then_change_source(table: object, path: Path) -> None:
        real_write(table, path)
        source = prepared / "generate" / "metadata.jsonl"
        source.write_bytes(source.read_bytes() + b" ")

    monkeypatch.setattr(producer, "atomic_write_parquet", write_then_change_source)
    with pytest.raises(ValueError, match="does not match prepare_manifest"):
        write_project_input(
            prepared_dir=prepared,
            output_path=output,
            manifest_path=manifest,
            max_concepts=2,
        )
    assert output.is_file()
    assert not manifest.exists()


def test_writer_rejects_a_valid_but_nonidentical_parquet_roundtrip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared = _prepared(tmp_path)
    real_read = pd.read_parquet

    def reordered(path: Path) -> pd.DataFrame:
        return real_read(path).iloc[::-1].reset_index(drop=True)

    monkeypatch.setattr(producer.pd, "read_parquet", reordered)
    with pytest.raises(ValueError, match="round trip changed"):
        write_project_input(
            prepared_dir=prepared,
            output_path=tmp_path / "output.parquet",
            manifest_path=tmp_path / "manifest.json",
            max_concepts=2,
        )


def test_hydra_config_ingress_is_exact_and_functional(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    raw = {
        "prepared_dir": str(prepared),
        "output_path": str(tmp_path / "artifact" / "input.parquet"),
        "manifest_path": str(tmp_path / "artifact" / "manifest.json"),
        "max_concepts": 2,
        "force": False,
    }
    frame = write_project_input_from_dictconfig(OmegaConf.create(raw))
    assert len(frame) == 240

    with pytest.raises(ValueError, match="config keys do not match"):
        write_project_input_from_dictconfig(OmegaConf.create({**raw, "unknown": 1}))
    incomplete = dict(raw)
    del incomplete["force"]
    with pytest.raises(ValueError, match="config keys do not match"):
        write_project_input_from_dictconfig(OmegaConf.create(incomplete))
    with pytest.raises(ValueError, match="config must be a mapping"):
        write_project_input_from_dictconfig(cast(Any, OmegaConf.create([1, 2])))
    with pytest.raises(ValueError, match="config keys must be strings"):
        write_project_input_from_dictconfig(OmegaConf.create({1: "value"}))


def test_cli_composes_overrides_and_publishes_a_loadable_artifact(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    output = tmp_path / "cli" / "input.parquet"
    manifest = tmp_path / "cli" / "manifest.json"

    build_project_input_main(
        [
            f"prepared_dir={prepared}",
            f"output_path={output}",
            f"manifest_path={manifest}",
            "max_concepts=2",
        ]
    )

    frame, _, _ = load_project_input_artifact(
        input_path=output,
        manifest_path=manifest,
        prepared_dir=prepared,
        max_concepts=2,
    )
    assert len(frame) == 2 * 10 * len(FULL_FACTORS)
