"""Operator-attested replay of completed counter points from an immutable snapshot.

A snapshot inventory is a provenance receipt, never a native completion manifest.
Only atomic point.json files are eligible; missing points are recomputed in full.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import re
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from nanochat.atomic_io import atomic_write_text
from scripts.eval.axbench_adv.pressures import COUNTER_PRESSURES, Pressure, magnitude_label
from steering._common.strict_json import reject_nonstandard_json_constant, strict_json_object
from steering.eval.axbench_adv.counter import CounterDirection, mean_counter_fitness
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation

SNAPSHOT_NAME = "counter_snapshot.json"
# This parent was audited against the counter runtime at 0cdb3cb09. This is an
# explicit legacy continuation, not a claim that parent points ran on new main.
QUALIFIED_PARENT = "5fbf442e66c478f2277c369877e5f447753e84f8"
SCIENCE_FIELDS = (
    "master_seed",
    "gepa_seed",
    "worker_max_prompt_tokens",
    "pressures",
    "counter_magnitudes",
    "project_counter_layer",
    "project_counter_fit_seed",
    "factor_overrides",
    "api_base_url",
)
_POINT_FIELDS = {"magnitude", "direction", "evaluations", "mean_fitness"}
_ROW_FIELDS = {
    "input_id",
    "generation",
    "concept_rating",
    "fluency_rating",
    "judge_rationale",
    "paid_judge_cache_misses",
}
_RATING = re.compile(
    r"(?<!\w)Rating:\s*\[\[\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\]\]"
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(
            path.read_text(),
            object_pairs_hook=strict_json_object,
            parse_constant=reject_nonstandard_json_constant,
        )
    except (OSError, ValueError, UnicodeError) as exc:
        raise AxBenchAdvError(f"invalid counter snapshot JSON: {path}") from exc


def _inventory(root: Path) -> dict[str, str]:
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AxBenchAdvError(f"snapshot must not contain symlinks: {path}")
        if path.is_file() and path != root / SNAPSHOT_NAME:
            files[path.relative_to(root).as_posix()] = sha256(path)
    return files


def seal_snapshot(
    root: Path,
    *,
    parent_identity: str,
    methods: Sequence[str],
    concept_ids: Sequence[int],
    parent_tokenizer_path: str,
) -> Path:
    """Seal an already quiescent/copied snapshot; never reads a live job itself."""
    root = root.resolve(strict=True)
    if (
        not parent_identity.strip()
        or not methods
        or not concept_ids
        or not Path(parent_tokenizer_path).is_absolute()
    ):
        raise AxBenchAdvError("snapshot requires parent identity and execution scope")
    if not set(methods).issubset({"SelfConditioning", "SelfConditioningMasked"}):
        raise AxBenchAdvError("qualified counter replay supports the two self-conditioning methods")
    path = root / SNAPSHOT_NAME
    if path.exists():
        raise AxBenchAdvError("snapshot is already sealed")
    files = _inventory(root)
    required = {
        "config.yaml",
        "launch.json",
        "output/source_contract.json",
        "output/selection/best_points.json",
        "judge-cache.sqlite",
        "fit_manifest.json",
    }
    if any(name in files for name in ("judge-cache.sqlite-wal", "judge-cache.sqlite-shm")):
        raise AxBenchAdvError("snapshot cache must be a closed SQLite backup without WAL sidecars")
    if not required.issubset(files):
        raise AxBenchAdvError(
            f"snapshot missing required evidence: {sorted(required - set(files))}"
        )
    if not any(name.startswith("tokenizer/") for name in files):
        raise AxBenchAdvError("snapshot requires the actual parent tokenizer directory")
    if _inventory(Path(parent_tokenizer_path)) != {
        name.removeprefix("tokenizer/"): digest
        for name, digest in files.items()
        if name.startswith("tokenizer/")
    }:
        raise AxBenchAdvError("snapshot tokenizer differs from parent tokenizer path")
    launch = read_json(root / "launch.json")
    if (
        not isinstance(launch, dict)
        or launch.get("source") != QUALIFIED_PARENT
        or launch.get("concept_ids") != list(concept_ids)
    ):
        raise AxBenchAdvError("snapshot launch source/scope differs")
    if "methods" in launch and launch["methods"] != list(methods):
        raise AxBenchAdvError("snapshot launch methods differ")
    if "output/manifest.json" in files:
        manifest = read_json(root / "output/manifest.json")
        implementation = manifest.get("implementation") if isinstance(manifest, dict) else None
        if (
            not isinstance(implementation, dict)
            or implementation.get("repository_git_commit") != QUALIFIED_PARENT
        ):
            raise AxBenchAdvError("snapshot manifest producer commit differs")
    document = {
        "schema_version": 1,
        "kind": "counter_partial_snapshot",
        "parent_source": QUALIFIED_PARENT,
        "parent_identity": parent_identity,
        "methods": list(methods),
        "concept_ids": list(concept_ids),
        "files": files,
        "parent_tokenizer_path": parent_tokenizer_path,
    }
    atomic_write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", path)
    return path


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _without_paths(document: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in document.items() if key != "paths"}


def _point(path: Path, magnitude: float, direction: Mapping[str, Any]) -> tuple[RawEvaluation, ...]:
    doc = read_json(path)
    if not isinstance(doc, dict) or set(doc) != _POINT_FIELDS:
        raise AxBenchAdvError(f"counter point fields differ: {path}")
    if type(doc["magnitude"]) not in (int, float) or doc["magnitude"] != magnitude:
        raise AxBenchAdvError(f"counter point magnitude differs: {path}")
    saved_direction = doc["direction"]
    if (
        not isinstance(saved_direction, dict)
        or (
            _canonical({k: v for k, v in saved_direction.items() if k != "path"})
            != _canonical({k: v for k, v in direction.items() if k != "path"})
        )
        or not isinstance(saved_direction.get("path"), str)
        or not Path(saved_direction["path"]).is_absolute()
    ):
        raise AxBenchAdvError(f"counter point direction differs: {path}")
    rows = doc["evaluations"]
    if not isinstance(rows, list) or len(rows) != 10:
        raise AxBenchAdvError(f"counter point must contain ten complete rows: {path}")
    results = []
    for index, candidate in enumerate(rows):
        if not isinstance(candidate, dict) or set(candidate) != _ROW_FIELDS:
            raise AxBenchAdvError(f"counter point input rows differ: {path}")
        row = cast(dict[str, Any], candidate)
        if type(row["input_id"]) is not int or row["input_id"] != index:
            raise AxBenchAdvError(f"counter point input rows differ: {path}")
        if not isinstance(row["generation"], str) or not isinstance(row["judge_rationale"], str):
            raise AxBenchAdvError(f"counter point text fields differ: {path}")
        if (
            type(row["paid_judge_cache_misses"]) is not int
            or not 0 <= row["paid_judge_cache_misses"] <= 2
        ):
            raise AxBenchAdvError(f"counter point accounting differs: {path}")
        try:
            rationale = json.loads(row["judge_rationale"], object_pairs_hook=strict_json_object)
        except ValueError as exc:
            raise AxBenchAdvError(f"counter point rationale is corrupt: {path}") from exc
        if not isinstance(rationale, dict) or set(rationale) != {"concept", "fluency"}:
            raise AxBenchAdvError(f"counter point rationale fields differ: {path}")
        for kind in ("concept", "fluency"):
            rating = row[f"{kind}_rating"]
            completion = rationale[kind]
            if (
                type(rating) not in (int, float)
                or not math.isfinite(rating)
                or not 0 <= rating <= 2
            ):
                raise AxBenchAdvError(f"counter point rating invalid: {path}")
            matches = _RATING.findall(completion) if isinstance(completion, str) else []
            if len(matches) != 1 or completion.count("Rating:") != 1 or float(matches[0]) != rating:
                raise AxBenchAdvError(f"counter point rationale/rating differs: {path}")
        results.append(RawEvaluation(**{k: v for k, v in row.items() if k != "input_id"}))
    if type(doc["mean_fitness"]) not in (int, float) or doc["mean_fitness"] != mean_counter_fitness(
        results
    ):
        raise AxBenchAdvError(f"counter point fitness differs: {path}")
    return tuple(results)


def _templates() -> dict[str, str]:
    # Read literal constants without importing the vendor package/model stack.
    path = (
        Path(__file__).resolve().parents[3]
        / "vendor/axbench/axbench/evaluators/prompt_templates.py"
    )
    wanted = {
        "concept": "UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE",
        "fluency": "UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE",
    }
    assignments = {
        node.targets[0].id: ast.literal_eval(node.value)
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign)
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in wanted.values()
    }
    return {kind: assignments[name] for kind, name in wanted.items()}


def validate_judge_cache(
    cache_path: Path, retained: Sequence[tuple[str, RawEvaluation]], *, api_base_url: str
) -> None:
    """Prove final judging cannot rebill retained concept/fluency completions."""
    templates = _templates()
    try:
        with sqlite3.connect(f"{cache_path.resolve().as_uri()}?mode=ro", uri=True) as db:
            for concept, raw in retained:
                rationale = json.loads(raw.judge_rationale)
                for kind, template in templates.items():
                    request = {
                        "version": 1,
                        "endpoint": api_base_url.strip().rstrip("/") + "/chat/completions",
                        "model": "openai/gpt-5.6-luna",
                        "temperature": 0.0,
                        "messages": [
                            {
                                "role": "user",
                                "content": template.format(
                                    concept=concept, sentence=raw.generation
                                ),
                            }
                        ],
                    }
                    encoded = json.dumps(
                        request,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    row = db.execute(
                        "SELECT request_json, response FROM judge_responses WHERE request_sha256 = ?",
                        (hashlib.sha256(encoded.encode()).digest(),),
                    ).fetchone()
                    if row != (encoded, rationale[kind]):
                        raise AxBenchAdvError(
                            f"retained {kind} judgment missing/conflicting in child cache"
                        )
    except sqlite3.Error as exc:
        raise AxBenchAdvError("invalid counter replay judge cache") from exc


def validate_cache_copy(parent: Path, child: Path) -> None:
    """Keep every paid response, including incomplete points and final instructions."""
    try:
        with (
            sqlite3.connect(f"{parent.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as source,
            sqlite3.connect(f"{child.resolve().as_uri()}?mode=ro", uri=True) as target,
        ):
            for digest, request, response in source.execute(
                "SELECT request_sha256, request_json, response FROM judge_responses"
            ):
                if (
                    not isinstance(request, str)
                    or hashlib.sha256(request.encode()).digest() != digest
                ):
                    raise AxBenchAdvError("parent judge cache has corrupt request identity")
                row = target.execute(
                    "SELECT request_json, response FROM judge_responses WHERE request_sha256 = ?",
                    (digest,),
                ).fetchone()
                if row != (request, response):
                    raise AxBenchAdvError(
                        "child cache omits or conflicts with a parent paid judgment"
                    )
    except sqlite3.Error as exc:
        raise AxBenchAdvError("invalid counter replay cache copy") from exc


@dataclass
class CounterReplay:
    root: Path
    receipt: dict[str, Any]
    points: dict[tuple[str, str, int, float], tuple[RawEvaluation, ...]]
    controls: dict[tuple[str, int], tuple[RawEvaluation, ...]]

    def revalidate(self) -> None:
        for path, digest in self.receipt.get("child_direction_files", {}).items():
            if sha256(Path(path)) != digest:
                raise AxBenchAdvError("counter replay direction file changed during run")
        if "child_tokenizer_path" in self.receipt:
            expected = {
                name.removeprefix("tokenizer/"): digest
                for name, digest in self.receipt["snapshot"]["files"].items()
                if name.startswith("tokenizer/")
            }
            if _inventory(Path(self.receipt["child_tokenizer_path"])) != expected:
                raise AxBenchAdvError("counter replay tokenizer changed during run")
        if _inventory(self.root) != self.receipt["snapshot"]["files"]:
            raise AxBenchAdvError("parent counter snapshot changed during replay")
        if sha256(self.root / SNAPSHOT_NAME) != self.receipt["snapshot_sha256"]:
            raise AxBenchAdvError("parent counter snapshot index changed during replay")

    def complete(
        self, method: str, concept_id: int, pressures: Sequence[Any], magnitudes: Sequence[Any]
    ) -> bool:
        return (method, concept_id) in self.controls and all(
            (str(p), method, concept_id, float(m)) in self.points
            for p in pressures
            for m in magnitudes
        )


def load_replay(
    *,
    snapshot_path: Path,
    snapshot_sha256: str,
    science: Mapping[str, Any],
    child_contract: Path,
    child_selection: Path,
    directions: Mapping[tuple[Pressure, str, int], CounterDirection],
    methods: Sequence[str],
    concept_ids: Sequence[int],
    concepts: Mapping[int, str],
    judge_cache: Path,
    output_dir: Path,
    fit_manifest_sha256: str,
) -> CounterReplay:
    root = snapshot_path.resolve().parent
    if output_dir.resolve().is_relative_to(root) or root.is_relative_to(output_dir.resolve()):
        raise AxBenchAdvError("counter replay output must be disjoint from parent snapshot")
    if judge_cache.resolve().is_relative_to(root):
        raise AxBenchAdvError("counter replay must use a separate child cache copy")
    if sha256(snapshot_path) != snapshot_sha256:
        raise AxBenchAdvError("counter snapshot digest differs")
    doc = read_json(snapshot_path)
    if (
        not isinstance(doc, dict)
        or set(doc)
        != {
            "schema_version",
            "kind",
            "parent_source",
            "parent_identity",
            "methods",
            "concept_ids",
            "files",
            "parent_tokenizer_path",
        }
        or type(doc["schema_version"]) is not int
        or doc["schema_version"] != 1
        or doc["kind"] != "counter_partial_snapshot"
        or doc["parent_source"] != QUALIFIED_PARENT
    ):
        raise AxBenchAdvError("counter snapshot source/schema is not qualified")
    if not set(methods).issubset({"SelfConditioning", "SelfConditioningMasked"}):
        raise AxBenchAdvError("qualified replay supports only self-conditioning methods")
    if list(methods) != doc["methods"] or list(concept_ids) != doc["concept_ids"]:
        raise AxBenchAdvError("counter replay scope differs from parent snapshot")
    if not set(map(Pressure, science["pressures"])).issubset(COUNTER_PRESSURES):
        raise AxBenchAdvError("counter replay only supports counter-only runs")
    replay = CounterReplay(
        root,
        {
            "snapshot_sha256": snapshot_sha256,
            "snapshot": doc,
            "logical_run": "legacy_counter_continuation",
        },
        {},
        {},
    )
    replay.revalidate()
    if doc["files"].get("fit_manifest.json") != fit_manifest_sha256:
        raise AxBenchAdvError("counter replay fit manifest differs")
    tokenizer_root = (
        Path(
            os.environ.get("NANOCHAT_BASE_DIR")
            or str(Path(__file__).resolve().parents[3] / "nanochat_cache")
        )
        / "tokenizer"
    )
    expected_tokenizer = {
        name.removeprefix("tokenizer/"): digest
        for name, digest in doc["files"].items()
        if name.startswith("tokenizer/")
    }
    if not expected_tokenizer or _inventory(tokenizer_root) != expected_tokenizer:
        raise AxBenchAdvError("counter replay tokenizer assets differ")
    replay.receipt["child_tokenizer_path"] = str(tokenizer_root.resolve())
    replay.receipt["fit_manifest_sha256"] = fit_manifest_sha256
    replay.receipt["child_direction_files"] = {
        str(d.provenance["path"]): d.provenance["sha256"] for d in directions.values()
    }
    parent_cfg = yaml.safe_load((root / "config.yaml").read_text())
    for key in SCIENCE_FIELDS:
        if _canonical(parent_cfg.get(key)) != _canonical(science[key]):
            raise AxBenchAdvError(f"counter replay config differs: {key}")
    if _canonical(_without_paths(read_json(root / "output/source_contract.json"))) != _canonical(
        _without_paths(read_json(child_contract))
    ):
        raise AxBenchAdvError("counter replay source contract differs")
    if _canonical(read_json(root / "output/selection/best_points.json")) != _canonical(
        read_json(child_selection)
    ):
        raise AxBenchAdvError("counter replay selected factors differ")
    expected_paths = set()
    retained = []
    for (pressure, method, concept_id), direction in directions.items():
        for magnitude in (0.0, *science["counter_magnitudes"]):
            relative = f"output/counter/{pressure}/{method}/c{concept_id}/{magnitude_label(magnitude)}/point.json"
            expected_paths.add(relative)
            if relative not in doc["files"]:
                continue
            results = _point(root / relative, magnitude, direction.provenance)
            replay.points[(str(pressure), method, concept_id, float(magnitude))] = results
            if magnitude == 0:
                previous = replay.controls.setdefault((method, concept_id), results)
                if previous != results:
                    raise AxBenchAdvError("counter replay shared zero differs across pressures")
            retained.extend((concepts[concept_id], raw) for raw in results)
    actual_paths = {
        name
        for name in doc["files"]
        if name.endswith("/point.json") and name.startswith("output/counter/")
    }
    if actual_paths - expected_paths:
        raise AxBenchAdvError("counter snapshot contains points outside declared grid")
    for _pressure, method, concept_id, magnitude in replay.points:
        if magnitude and (method, concept_id) not in replay.controls:
            raise AxBenchAdvError("completed counter point has no saved shared zero")
    validate_cache_copy(root / "judge-cache.sqlite", judge_cache)
    validate_judge_cache(judge_cache, retained, api_base_url=science["api_base_url"])
    replay.receipt["reused_point_records"] = len(replay.points)
    replay.receipt["reused_unique_evaluations"] = 10 * (
        len(replay.controls) + sum(k[-1] != 0 for k in replay.points)
    )
    missing = []
    for method in methods:
        for concept_id in concept_ids:
            if (method, concept_id) not in replay.controls:
                missing.append(["shared_zero", method, concept_id, 0.0])
            for pressure in science["pressures"]:
                for magnitude in science["counter_magnitudes"]:
                    key = (str(pressure), method, concept_id, float(magnitude))
                    if key not in replay.points:
                        missing.append(list(key))
    replay.receipt["missing_generation_points"] = missing
    replay.receipt["remaining_generation_batches"] = len(missing)
    replay.receipt["remaining_generation_rows"] = 10 * len(missing)
    atomic_write_text(
        json.dumps(replay.receipt, indent=2, sort_keys=True) + "\n",
        output_dir / "counter_resume.json",
    )
    return replay
