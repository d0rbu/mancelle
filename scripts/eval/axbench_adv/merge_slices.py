"""Merge completed AxBench-Adv concept or method slices into plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq
import torch as t
from safetensors import SafetensorError
from safetensors.torch import load_file

from nanochat.atomic_io import atomic_write_text
from scripts.eval.axbench_adv.pressures import (
    COUNTER_PRESSURES,
    DEFAULT_MAGNITUDES,
    LEGACY_PRESSURES,
    Pressure,
    magnitude_label,
    parse_pressures,
    validate_magnitudes,
)
from steering._common.strict_json import (
    reject_nonstandard_json_constant,
    strict_json_object,
)
from steering.eval.axbench_adv.aggregation import (
    build_aggregate,
    build_counter_aggregate,
    build_gcg_aggregate,
)
from steering.eval.axbench_adv.best_point import load_steering_entries, write_counter_best_points
from steering.eval.axbench_adv.counter import mean_counter_fitness, validate_counter_provenance
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.execution_scope import validate_concept_ids
from steering.eval.axbench_adv.final_judge import (
    validate_final_judge_outputs,
    validate_point_frame,
)
from steering.eval.axbench_adv.gcg_config import parse_gcg_config
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.method_grid import ALL_METHOD_FACTORS
from steering.eval.axbench_adv.plotting import plot_concept_vs_fluency
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix
from steering.eval.axbench_adv.soft_prompt_config import parse_soft_prompt_config
from steering.eval.axbench_adv.soft_prompt_driver import _validate_tensor, validate_embedding_layout
from steering.eval.axbench_adv.types import (
    ALL_METHODS,
    NL_METHOD,
    PLOT_EFFORT_LEVELS,
    PROPOSAL_COUNTS,
    validate_method_subset,
)


def _peek_concept_ids(steering_jsonl: Path) -> tuple[int, ...]:
    """Return the concept identities a judged steering.jsonl carries, in file order."""

    concept_ids: list[int] = []
    for line_number, entry in enumerate(
        _strict_jsonl(steering_jsonl, label="steering.jsonl"),
        start=1,
    ):
        if type(entry.get("concept_id")) is not int:
            raise AxBenchAdvError(
                f"steering.jsonl line {line_number} lacks an integer concept_id: {steering_jsonl}"
            )
        concept_ids.append(entry["concept_id"])
    return tuple(concept_ids)


_SCIENTIFIC_COMPATIBILITY_FIELDS = (
    "schema_version",
    "proposal_counts",
    "max_concepts",
    "master_seed",
    "gepa_seed",
    "source_hashes",
    "runtime_source_hashes",
    "runtime_source_byte_lengths",
    "natural_language",
)


@dataclass(frozen=True)
class _ControlRecord:
    prompt: tuple[Any, ...]
    judged: tuple[Any, ...]
    raw: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _RunSlice:
    root: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    methods: tuple[str, ...]
    concept_ids: tuple[int, ...]
    factors: Mapping[str, float]
    entries: Mapping[float, Sequence[Mapping[str, Any]]]
    fitness: Mapping[tuple[str, float], Sequence[float]] = field(default_factory=dict)
    controls: Mapping[tuple[str, int, int], _ControlRecord] = field(default_factory=dict)
    gcg_receipt_compatibility: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]] = (
        field(default_factory=dict)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise AxBenchAdvError(f"{label} is missing: {path}")
    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_nonstandard_json_constant,
            object_pairs_hook=strict_json_object,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise AxBenchAdvError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(loaded, dict):
        raise AxBenchAdvError(f"{label} must be a JSON object: {path}")
    return loaded


def _strict_jsonl(path: Path, *, label: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AxBenchAdvError(f"{label} is not valid UTF-8: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(
                line,
                parse_constant=reject_nonstandard_json_constant,
                object_pairs_hook=strict_json_object,
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise AxBenchAdvError(
                f"{label} is not strict JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(entry, dict):
            raise AxBenchAdvError(f"{label} line {line_number} must be an object: {path}")
        entries.append(entry)
    return entries


def _validate_recorded_inventory(
    root: Path,
    files: Mapping[str, Any],
    *,
    required: Sequence[Path],
) -> None:
    """Authenticate recorded files without allowing paths outside the run root."""

    resolved_files: dict[Path, str] = {}
    for raw_relative, raw_hash in files.items():
        if not isinstance(raw_relative, str) or not raw_relative:
            raise AxBenchAdvError("slice manifest file paths must be nonblank strings")
        relative = Path(raw_relative)
        raw_parts = raw_relative.split("/")
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in raw_parts):
            raise AxBenchAdvError(f"slice manifest file path is unsafe: {raw_relative!r}")
        if (
            not isinstance(raw_hash, str)
            or len(raw_hash) != 64
            or any(character not in "0123456789abcdef" for character in raw_hash)
        ):
            raise AxBenchAdvError(f"slice manifest file hash is invalid: {raw_relative}")
        path = root / relative
        if not path.is_file():
            raise AxBenchAdvError(f"slice manifest recorded file is missing: {raw_relative}")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise AxBenchAdvError(f"slice manifest file escapes its run root: {raw_relative}")
        if resolved in resolved_files:
            raise AxBenchAdvError(
                "slice manifest has conflicting paths for one file: "
                f"{resolved_files[resolved]!r} and {raw_relative!r}"
            )
        resolved_files[resolved] = raw_relative
        if _sha256(resolved) != raw_hash:
            raise AxBenchAdvError(f"slice recorded file hash differs: {raw_relative}")

    missing = [str(path) for path in required if str(path) not in files]
    if missing:
        raise AxBenchAdvError(f"slice manifest omits required scientific files: {missing}")


def _raw_evaluation(row: Mapping[str, Any], *, relative: Path) -> RawEvaluation:
    fields = (
        "generation",
        "concept_rating",
        "fluency_rating",
        "judge_rationale",
        "paid_judge_cache_misses",
    )
    if any(field not in row for field in fields):
        raise AxBenchAdvError(f"counter point evaluation fields differ: {relative}")
    try:
        return RawEvaluation(**{field: row[field] for field in fields})
    except TypeError as exc:
        raise AxBenchAdvError(f"counter point evaluation fields differ: {relative}") from exc


def _reconstruct_judged_emissions(
    *,
    input_table: Any,
    entries: Sequence[Mapping[str, Any]],
    methods: tuple[str, ...],
    factors: Mapping[str, float],
    concept_ids: tuple[int, ...],
) -> dict[tuple[str, int, int], RawEvaluation]:
    """Reconstruct legacy GEPA emissions for internal judge-output validation.

    Historical GEPA slices did not retain worker point receipts.  Their raw
    concept/fluency ratings and completions are nevertheless present in the
    judged JSONL, so they can validate frame geometry and JSONL/Parquet parity
    without inventing unavailable worker provenance.
    """

    frame = input_table.to_pandas()
    by_concept = {int(entry["concept_id"]): entry for entry in entries}
    emissions: dict[tuple[str, int, int], RawEvaluation] = {}
    required = (
        "raw_relevance_concept_ratings",
        "raw_fluency_ratings",
        "relevance_concept_completions",
        "fluency_completions",
    )
    for concept_id in concept_ids:
        concept_frame = frame.loc[frame["concept_id"].eq(concept_id)].reset_index(drop=True)
        evaluator = _entry_methods(by_concept[concept_id])
        for method in methods:
            block = evaluator.get(method)
            if not isinstance(block, Mapping):
                raise AxBenchAdvError(f"GEPA judge block differs for {method}/c{concept_id}")
            values = {field: block.get(field) for field in required}
            if any(
                not isinstance(value, list) or len(value) != len(concept_frame)
                for value in values.values()
            ):
                raise AxBenchAdvError(f"GEPA raw judge fields differ for {method}/c{concept_id}")
            declared = concept_frame["factor"].eq(factors[method]).tolist()
            for position, selected in enumerate(declared):
                if not selected:
                    continue
                row = concept_frame.iloc[position]
                input_id = int(row["input_id"])
                emissions[(method, concept_id, input_id)] = RawEvaluation(
                    generation=row[f"{method}_steered_generation"],
                    concept_rating=values["raw_relevance_concept_ratings"][position],
                    fluency_rating=values["raw_fluency_ratings"][position],
                    judge_rationale=json.dumps(
                        {
                            "concept": values["relevance_concept_completions"][position],
                            "fluency": values["fluency_completions"][position],
                        },
                        sort_keys=True,
                        allow_nan=False,
                    ),
                    paid_judge_cache_misses=0,
                )
    return emissions


def _control_records(
    *,
    root: Path,
    methods: tuple[str, ...],
    factors: Mapping[str, float],
    concept_ids: tuple[int, ...],
    max_concepts: int,
    entries: Sequence[Mapping[str, Any]],
    raw_records: Mapping[tuple[str, int, int], Mapping[str, Any]],
) -> dict[tuple[str, int, int], _ControlRecord]:
    input_path = root / "points" / "p0" / "input" / "steering_data.parquet"
    table = pq.read_table(input_path)
    validate_point_frame(
        table,
        best_factors=factors,
        max_concepts=max_concepts,
        concept_ids=concept_ids,
    )
    frame = table.to_pandas()
    by_concept = {int(entry["concept_id"]): entry for entry in entries}
    controls: dict[tuple[str, int, int], _ControlRecord] = {}
    for concept_id in concept_ids:
        concept_frame = frame.loc[frame["concept_id"].eq(concept_id)].reset_index(drop=True)
        evaluator = _entry_methods(by_concept[concept_id])
        for method in methods:
            block = evaluator[method]
            if not isinstance(block, Mapping):
                raise AxBenchAdvError(f"control judge block differs for {method}/c{concept_id}")
            declared = concept_frame["factor"].eq(factors[method]).tolist()
            positions = [position for position, selected in enumerate(declared) if selected]
            if len(positions) != 10:
                raise AxBenchAdvError(
                    f"control coordinate coverage differs for {method}/c{concept_id}"
                )
            summaries = tuple(
                (field, tuple(value))
                for field, value in sorted(block.items())
                if isinstance(value, list) and len(value) == 1
            )
            for position in positions:
                row = concept_frame.iloc[position]
                input_id = int(row["input_id"])
                per_input = tuple(
                    (field, value[position])
                    for field, value in sorted(block.items())
                    if isinstance(value, list) and len(value) == len(concept_frame)
                )
                controls[(method, concept_id, input_id)] = _ControlRecord(
                    prompt=(
                        row["dataset_name"],
                        row["input_concept"],
                        row["original_prompt"],
                        row[f"{method}_steered_generation"],
                    ),
                    judged=(summaries, per_input),
                    raw=raw_records.get((method, concept_id, input_id)),
                )
    return controls


def _manifest_methods(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    raw = manifest.get("methods")
    if not isinstance(raw, list) or any(not isinstance(method, str) for method in raw):
        raise AxBenchAdvError("slice manifest methods must be a string list")
    methods = validate_method_subset(raw)
    if list(methods) != raw:
        raise AxBenchAdvError("slice manifest methods are not in canonical order")
    return methods


def _manifest_factors(manifest: Mapping[str, Any], methods: tuple[str, ...]) -> dict[str, float]:
    raw = manifest.get("best_factors")
    if not isinstance(raw, dict) or set(raw) != set(methods):
        raise AxBenchAdvError("slice manifest best_factors must match its effective methods")
    factors: dict[str, float] = {}
    for method in methods:
        value = raw[method]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AxBenchAdvError(f"slice manifest factor for {method} must be numeric")
        factor = float(value)
        if not math.isfinite(factor) or factor not in tuple(
            float(candidate) for candidate in ALL_METHOD_FACTORS[method]
        ):
            raise AxBenchAdvError(f"slice manifest factor for {method} is outside its grid")
        factors[method] = factor
    return factors


def _entry_methods(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    results = entry.get("results")
    evaluator = results.get("LMJudgeEvaluator") if isinstance(results, Mapping) else None
    if not isinstance(evaluator, Mapping):
        raise AxBenchAdvError("slice judged entry has no LMJudgeEvaluator mapping")
    return evaluator


def _nl_source_identity(
    manifest: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any]] | None:
    """Return the one declared NL provenance form without conflating the two."""

    attachment = manifest.get("nl_source_attachment")
    natural_language = manifest.get("natural_language")
    if attachment is not None and not isinstance(attachment, Mapping):
        raise AxBenchAdvError("slice manifest nl_source_attachment must be a mapping")
    if natural_language is not None and not isinstance(natural_language, Mapping):
        raise AxBenchAdvError("slice manifest natural_language must be a mapping")
    if attachment is not None and natural_language is not None:
        raise AxBenchAdvError("slice manifest declares both NL source provenance forms")
    if attachment is not None:
        return "nl_source_attachment", attachment
    if natural_language is not None:
        return "natural_language", natural_language
    return None


def _validate_search_training_source(
    source: object, *, max_concepts: int, label: str
) -> Mapping[str, str]:
    pairs = source.get("pair_sha256") if isinstance(source, Mapping) else None
    if (
        not isinstance(source, Mapping)
        or set(source) != {"train_data", "metadata", "pair_sha256"}
        or not isinstance(pairs, Mapping)
    ):
        raise AxBenchAdvError(f"{label} slice lacks effective config or training source identity")
    if set(pairs) != {str(value) for value in range(max_concepts)}:
        raise AxBenchAdvError(f"{label} training pair identities do not cover the source population")
    source = cast(Mapping[str, Any], source)
    for file_field in ("train_data", "metadata"):
        record = source.get(file_field)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256", "byte_length"}
            or not isinstance(record["path"], str)
            or not Path(record["path"]).is_absolute()
            or not isinstance(record["sha256"], str)
            or len(record["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in record["sha256"])
            or type(record["byte_length"]) is not int
            or record["byte_length"] < 0
        ):
            raise AxBenchAdvError(f"{label} {file_field} training source identity is malformed")
    for identity in pairs.values():
        if (
            not isinstance(identity, str)
            or len(identity) != 64
            or any(character not in "0123456789abcdef" for character in identity)
        ):
            raise AxBenchAdvError(f"{label} training pair identity must be lowercase SHA-256")
    return cast(Mapping[str, str], pairs)


def _validate_search_receipts(
    *,
    root: Path,
    manifest: Mapping[str, Any],
    files: Mapping[str, Any],
    methods: Sequence[str],
    concept_ids: Sequence[int],
    factors: Mapping[str, float],
    token_counts: Sequence[int],
    pressure: Pressure,
    compatibility: dict[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]] | None = None,
) -> list[Path]:
    soft_prompt = pressure == Pressure.SOFT_PROMPT
    label = "soft-prompt" if soft_prompt else "GCG"
    effective = manifest.get(pressure)
    if not isinstance(effective, Mapping):
        raise AxBenchAdvError(f"{label} slice lacks its effective config")
    raw = dict(effective)
    objective = raw.get("objective")
    if objective == "activation_l2":
        if raw.pop("direction_factor_policy", None) != "selected_method_factor":
            raise AxBenchAdvError(
                "activation-L2 direction_factor_policy must be selected_method_factor; "
                "a fresh L2 run in a new output directory is required"
            )
        raw["alpha"] = 0.0
    if soft_prompt:
        for name in ("master_seed", "optimizer", "native_cast_policy"):
            raw.pop(name, None)
        parsed_soft = parse_soft_prompt_config(raw)
        master_seed = manifest.get("master_seed")
        if type(master_seed) is not int:
            raise AxBenchAdvError("soft-prompt manifest master_seed must be an integer")
        if (
            json.dumps(effective, sort_keys=True)
            != json.dumps(parsed_soft.effective_identity(master_seed=master_seed), sort_keys=True)
            or tuple(token_counts) != parsed_soft.optimized_token_counts
        ):
            raise AxBenchAdvError("soft-prompt effective config values are malformed")
        objective = parsed_soft.objective.value
        max_steps = parsed_soft.max_steps
    else:
        parsed_gcg = parse_gcg_config(raw)
        if json.loads(
            json.dumps(parsed_gcg.effective_identity())
        ) != effective or parsed_gcg.search.master_seed != manifest.get("master_seed"):
            raise AxBenchAdvError("GCG effective config values are malformed")
        objective = parsed_gcg.objective.value
        max_steps = parsed_gcg.search.max_steps
    pairs = _validate_search_training_source(
        manifest.get(f"{pressure}_training_source"),
        max_concepts=manifest["max_concepts"],
        label=label,
    )
    required: list[Path] = []
    stable: dict[tuple[str, int], tuple[Any, Any, str]] = {}
    for method in methods:
        for concept_id in concept_ids:
            observed_concept: str | None = None
            for count in token_counts:
                relative = Path(pressure) / method / f"c{concept_id}" / f"k{count}" / "search.json"
                path = root / relative
                document = _json_mapping(path, label=f"{label} search receipt")
                required.append(relative)
                if files.get(str(relative)) != _sha256(path):
                    raise AxBenchAdvError(f"{label} search receipt hash differs: {relative}")
                expected_fields = {
                    "schema_version",
                    "objective",
                    "effective_config",
                    "training_pair_identity",
                    "method",
                    "concept_id",
                    "concept",
                    "factor",
                    "token_count",
                    "diagnostics",
                }
                if not soft_prompt:
                    expected_fields |= {"suffix_token_ids", "suffix"}
                if count > 0:
                    expected_fields |= {
                        "model_tokenizer_identity",
                        "layouts_sha256",
                        "direction_calibration",
                        "objective_value",
                    }
                    if soft_prompt:
                        expected_fields |= {"initialization_token_ids", "tensor"}
                if set(document) != expected_fields:
                    raise AxBenchAdvError(f"{label} search receipt fields differ: {relative}")
                receipt_config = document.get("effective_config")
                if objective == "activation_l2" and (
                    not isinstance(receipt_config, Mapping)
                    or receipt_config.get("direction_factor_policy") != "selected_method_factor"
                ):
                    raise AxBenchAdvError(
                        "activation-L2 receipt direction_factor_policy must be "
                        "selected_method_factor; a fresh L2 run in a new output "
                        f"directory is required: {relative}"
                    )
                effective_matches = (
                    json.dumps(receipt_config, sort_keys=True)
                    == json.dumps(effective, sort_keys=True)
                    if soft_prompt
                    else receipt_config == effective
                )
                diagnostics = document.get("diagnostics")
                concept = document.get("concept")
                if (
                    type(document.get("schema_version")) is not int
                    or document["schema_version"] != 1
                    or document.get("objective") != objective
                    or not effective_matches
                    or document.get("training_pair_identity") != pairs[str(concept_id)]
                    or document.get("method") != method
                    or type(document.get("concept_id")) is not int
                    or document["concept_id"] != concept_id
                    or not isinstance(concept, str)
                    or not concept.strip()
                    or type(document.get("factor")) not in (int, float)
                    or document["factor"] != factors[method]
                    or type(document.get("token_count")) is not int
                    or document["token_count"] != count
                    or not isinstance(diagnostics, Mapping)
                    or diagnostics.get("control") is not (count == 0)
                ):
                    raise AxBenchAdvError(f"{label} search receipt identity differs: {relative}")
                if soft_prompt:
                    if (
                        set(diagnostics)
                        != {"control", "updates", "best_step", "full_population_scores"}
                        or any(
                            type(diagnostics[field]) is not int
                            for field in ("updates", "best_step", "full_population_scores")
                        )
                        or diagnostics["updates"] != (max_steps if count > 0 else 0)
                        or not 0 <= diagnostics["best_step"] <= diagnostics["updates"]
                        or diagnostics["full_population_scores"]
                        != (max_steps + 1 if count > 0 else 0)
                    ):
                        raise AxBenchAdvError(
                            f"{label} search receipt identity differs: {relative}"
                        )
                elif (
                    set(diagnostics) != {"control", "accepted_steps", "visited_sequences"}
                    or type(diagnostics["accepted_steps"]) is not int
                    or not 0 <= diagnostics["accepted_steps"] <= max_steps
                    or type(diagnostics["visited_sequences"]) is not int
                    or diagnostics["visited_sequences"] < 0
                ):
                    raise AxBenchAdvError(f"{label} search receipt identity differs: {relative}")
                if observed_concept is not None and concept != observed_concept:
                    raise AxBenchAdvError(
                        f"{label} concept changes across token counts: {relative}"
                    )
                observed_concept = concept
                tensor_relative = relative.with_name("embeddings.safetensors")
                tensor_path = root / tensor_relative
                if count == 0:
                    if soft_prompt:
                        if tensor_path.exists():
                            raise AxBenchAdvError(
                                f"soft-prompt zero control has a tensor: {relative}"
                            )
                    elif (
                        document["suffix_token_ids"] != []
                        or document["suffix"] != ""
                        or diagnostics["accepted_steps"] != 0
                        or diagnostics["visited_sequences"] != 0
                    ):
                        raise AxBenchAdvError(
                            f"GCG zero receipt is not the shared control: {relative}"
                        )
                    continue
                identity = document["model_tokenizer_identity"]
                base_identity = identity
                if soft_prompt:
                    if not isinstance(identity, Mapping):
                        raise AxBenchAdvError(
                            f"soft-prompt model identity is malformed: {relative}"
                        )
                    layout = validate_embedding_layout(identity)
                    base_identity = dict(identity)
                    del base_identity["embedding_layout"], base_identity["generation_conditioning"]
                    if identity.get("runtime") == "project_encoder_decoder":
                        del base_identity["compute_dtype"]
                _validate_gcg_model_identity(
                    base_identity,
                    method=method,
                    concept_id=concept_id,
                    factor=factors[method],
                    relative=relative,
                )
                token_ids = document[
                    "initialization_token_ids" if soft_prompt else "suffix_token_ids"
                ]
                layouts_sha = document["layouts_sha256"]
                objective_value = document["objective_value"]
                if (
                    not isinstance(token_ids, list)
                    or len(token_ids) != count
                    or any(
                        type(token) is not int or not 0 <= token < identity["vocab_size"]
                        for token in token_ids
                    )
                    or not isinstance(layouts_sha, str)
                    or len(layouts_sha) != 64
                    or any(character not in "0123456789abcdef" for character in layouts_sha)
                    or type(objective_value) not in (int, float)
                    or not math.isfinite(objective_value)
                ):
                    raise AxBenchAdvError(
                        f"{label} search receipt values are malformed: {relative}"
                    )
                if not soft_prompt and (
                    diagnostics["visited_sequences"] < 1
                    or not isinstance(document["suffix"], str)
                    or not document["suffix"]
                    or canonicalize_suffix(document["suffix"], forbidden_tokens=())
                    != document["suffix"]
                ):
                    raise AxBenchAdvError(f"GCG search receipt values are malformed: {relative}")
                calibration = document["direction_calibration"]
                _validate_gcg_calibration(
                    calibration,
                    objective=objective,
                    concept_id=concept_id,
                    model_identity=base_identity,
                    relative=relative,
                )
                project = identity.get("runtime") == "project_encoder_decoder"
                if soft_prompt:
                    if (
                        isinstance(calibration, Mapping)
                        and (calibration.get("target_model") == "encdec") != project
                    ):
                        raise AxBenchAdvError(
                            f"soft-prompt calibration runtime differs: {relative}"
                        )
                    descriptor = document["tensor"]
                    shape = (count, sum(width for width, _dtype in layout))
                    if (
                        not isinstance(descriptor, Mapping)
                        or set(descriptor)
                        != {"path", "sha256", "byte_length", "key", "dtype", "shape"}
                        or descriptor.get("path") != "embeddings.safetensors"
                        or descriptor.get("key") != "embeddings"
                        or descriptor.get("dtype") != "torch.float32"
                        or descriptor.get("shape") != list(shape)
                        or not isinstance(descriptor.get("shape"), list)
                        or any(type(value) is not int for value in descriptor["shape"])
                        or type(descriptor.get("byte_length")) is not int
                        or not tensor_path.is_file()
                        or descriptor["byte_length"] != tensor_path.stat().st_size
                        or descriptor.get("sha256") != _sha256(tensor_path)
                        or files.get(str(tensor_relative)) != descriptor["sha256"]
                    ):
                        raise AxBenchAdvError(f"soft-prompt tensor descriptor differs: {relative}")
                    try:
                        tensors = load_file(str(tensor_path), device="cpu")
                    except (SafetensorError, OSError, ValueError) as exc:
                        raise AxBenchAdvError(
                            f"soft-prompt tensor is invalid: {tensor_relative}"
                        ) from exc
                    if set(tensors) != {"embeddings"}:
                        raise AxBenchAdvError(f"soft-prompt tensor keys differ: {tensor_relative}")
                    _validate_tensor(
                        tensors["embeddings"],
                        shape=shape,
                        layout=layout,
                        label=f"tensor {tensor_relative}",
                        compute_dtype=(
                            getattr(t, identity["compute_dtype"].removeprefix("torch."))
                            if project
                            else None
                        ),
                    )
                    required.append(tensor_relative)
                common_model = dict(identity)
                for name in ("concept_id", "method", "factor"):
                    common_model.pop(name, None)
                if compatibility is not None:
                    runtime_key = ("soft_prompt:" if soft_prompt else "") + (
                        "project" if project else "huggingface"
                    )
                    common = (common_model, _gcg_common_calibration(calibration))
                    if compatibility.setdefault(runtime_key, common) != common:
                        raise AxBenchAdvError(
                            f"{label} runtime or calibration differs within slice: {relative}"
                        )
                current = (identity, calibration, layouts_sha)
                if stable.setdefault((method, concept_id), current) != current:
                    raise AxBenchAdvError(
                        f"{label} model, calibration, or layout changes across k: {relative}"
                    )
    return required


def _validate_gcg_model_identity(
    value: object,
    *,
    method: str,
    concept_id: int,
    factor: float,
    relative: Path,
) -> None:
    if not isinstance(value, Mapping):
        raise AxBenchAdvError(f"GCG model identity is malformed: {relative}")
    value = cast(Mapping[str, Any], value)
    runtime = value.get("runtime")
    if runtime == "project_encoder_decoder":
        expected = {
            "runtime",
            "concept_id",
            "rank_sentinel_id",
            "vocab_size",
            "embedding_dim",
            "method",
            "factor",
            "frozen_state",
        }
        valid_runtime = (
            set(value) == expected
            and value.get("frozen_state") == "clean_real_prompt_S0_before_assistant_marker"
            and type(value.get("embedding_dim")) is int
            and value["embedding_dim"] > 0
        )
    else:
        expected = {
            "concept_id",
            "factor",
            "layer",
            "method",
            "model_name",
            "model_revision",
            "rank_sentinel_id",
            "tokenizer_class",
            "vocab_size",
        }
        valid_runtime = (
            set(value) == expected
            and type(value.get("layer")) is int
            and value["layer"] >= 0
            and isinstance(value.get("model_name"), str)
            and bool(value["model_name"].strip())
            and (value.get("model_revision") is None or isinstance(value["model_revision"], str))
            and isinstance(value.get("tokenizer_class"), str)
            and bool(value["tokenizer_class"].strip())
        )
    if (
        not valid_runtime
        or value.get("method") != method
        or type(value.get("concept_id")) is not int
        or value.get("concept_id") != concept_id
        or type(value.get("factor")) not in (int, float)
        or value.get("factor") != factor
        or type(value.get("rank_sentinel_id")) is not int
        or value["rank_sentinel_id"] < 0
        or type(value.get("vocab_size")) is not int
        or value["vocab_size"] <= 0
        or value["rank_sentinel_id"] >= value["vocab_size"]
    ):
        raise AxBenchAdvError(f"GCG model identity is malformed: {relative}")


def _validate_gcg_calibration(
    value: object,
    *,
    objective: str,
    concept_id: int,
    model_identity: Mapping[str, Any],
    relative: Path,
) -> None:
    if objective == "response_margin":
        if value is not None:
            raise AxBenchAdvError(f"response-margin GCG has direction calibration: {relative}")
        return
    if not isinstance(value, Mapping):
        raise AxBenchAdvError(f"activation-L2 GCG lacks direction calibration: {relative}")
    value = cast(Mapping[str, Any], value)
    project = value.get("target_model") == "encdec"
    expected = (
        {
            "objective",
            "family",
            "concept_id",
            "path",
            "sha256",
            "fit_manifest_sha256",
            "layer",
            "site",
            "mode",
            "target_model",
            "direction_norm",
            "max_activation",
            "factor",
            "resolved_scale",
            "coefficient_semantics",
        }
        if project
        else {
            "site",
            "layer",
            "concept_id",
            "direction_path",
            "direction_sha256",
            "latent_path",
            "latent_sha256",
            "raw_max_activation",
            "max_activation",
            "fallback_used",
            "factor",
            "resolved_scale",
            "direction_norm",
            "offset_dtype_semantics",
        }
    )
    sha_fields = (
        ("sha256", "fit_manifest_sha256") if project else ("direction_sha256", "latent_sha256")
    )
    path_fields = ("path",) if project else ("direction_path", "latent_path")
    numeric_fields = ("direction_norm", "max_activation", "resolved_scale", "factor")
    if (
        set(value) != expected
        or type(value.get("concept_id")) is not int
        or value.get("concept_id") != concept_id
        or type(value.get("layer")) is not int
        or value["layer"] < 0
        or value.get("factor") != model_identity.get("factor")
        or any(
            type(value.get(field)) not in (int, float) or not math.isfinite(float(value[field]))
            for field in numeric_fields
        )
        or any(float(value[field]) <= 0.0 for field in ("direction_norm", "max_activation"))
        or float(value["factor"]) < 0.0
        or float(value["max_activation"]) * float(value["factor"]) != float(value["resolved_scale"])
        or any(
            not isinstance(value.get(field), str)
            or len(value[field]) != 64
            or any(character not in "0123456789abcdef" for character in value[field])
            for field in sha_fields
        )
        or any(
            not isinstance(value.get(field), str) or not Path(value[field]).is_absolute()
            for field in path_fields
        )
    ):
        raise AxBenchAdvError(f"GCG direction calibration is malformed: {relative}")
    if project:
        if (
            value.get("objective") != "activation_l2"
            or value.get("family") != "diffmean"
            or value.get("mode") != "positive"
            or value.get("site") != "post_cross_attention"
            or value.get("coefficient_semantics")
            != "saved_diffmean * own_positive_max_activation * selected_method_factor"
        ):
            raise AxBenchAdvError(f"GCG project calibration is malformed: {relative}")
    else:
        raw_max = value.get("raw_max_activation")
        valid_raw = raw_max is None or (
            not isinstance(raw_max, bool)
            and isinstance(raw_max, (int, float))
            and math.isfinite(float(raw_max))
        )
        if not valid_raw:
            raise AxBenchAdvError(f"GCG HF calibration is malformed: {relative}")
        raw_value = None if raw_max is None else float(raw_max)
        expected_fallback = raw_value is None or raw_value <= 0.0
        expected_maximum = 50.0 if expected_fallback else raw_value
        assert expected_maximum is not None
        local_gpt_neox = (
            isinstance(model_identity.get("model_name"), str)
            and Path(model_identity["model_name"]).is_absolute()
            and model_identity.get("model_revision") is None
        )
        if (
            type(value.get("fallback_used")) is not bool
            or value["fallback_used"] is not expected_fallback
            or float(value["max_activation"]) != expected_maximum
            or value.get("site")
            != (
                "block_output.after_actual_intervention"
                if local_gpt_neox
                else f"model.layers[{value['layer']}].output_after_actual_intervention"
            )
            or value.get("offset_dtype_semantics")
            != ("float32 [1] max_act * float32 [1] factor * saved bfloat16 direction -> float32")
        ):
            raise AxBenchAdvError(f"GCG HF calibration is malformed: {relative}")


def _gcg_common_calibration(value: object) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    value = cast(Mapping[str, Any], value)
    if value.get("target_model") == "encdec":
        fields = (
            "objective",
            "family",
            "fit_manifest_sha256",
            "layer",
            "site",
            "mode",
            "target_model",
            "coefficient_semantics",
        )
    else:
        fields = (
            "site",
            "layer",
            "direction_sha256",
            "latent_sha256",
            "offset_dtype_semantics",
        )
    return {field: value[field] for field in fields}


def _load_run_slice(root: Path, pressure: Pressure = Pressure.GEPA) -> _RunSlice:
    root = Path(root).expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _json_mapping(manifest_path, label="completed slice manifest")
    manifest_sha256 = _sha256(manifest_path)
    if manifest.get("schema_version") != 2:
        raise AxBenchAdvError("slice manifest schema_version must be 2")
    proposal_counts = manifest.get("proposal_counts")
    if (
        not isinstance(proposal_counts, list)
        or any(type(value) is not int for value in proposal_counts)
        or proposal_counts != list(PROPOSAL_COUNTS)
    ):
        raise AxBenchAdvError("slice manifest proposal_counts differ from AxBench-Adv")
    for seed_field in ("master_seed", "gepa_seed"):
        if type(manifest.get(seed_field)) is not int or not 0 <= manifest[seed_field] < 1 << 63:
            raise AxBenchAdvError(
                f"slice manifest {seed_field} must be an integer in [0, 2^63 - 1]"
            )
    for source_field in (
        "source_hashes",
        "runtime_source_hashes",
        "runtime_source_byte_lengths",
        "source_project",
    ):
        if not isinstance(manifest.get(source_field), dict) or not manifest[source_field]:
            raise AxBenchAdvError(f"slice manifest {source_field} must be a nonempty mapping")
    source_project = manifest["source_project"]
    masked_policy = source_project.get("self_conditioning_masked_policy")
    if not isinstance(masked_policy, str) or not masked_policy.strip():
        raise AxBenchAdvError("slice manifest source_project lacks self_conditioning_masked_policy")
    methods = _manifest_methods(manifest)
    max_concepts = manifest.get("max_concepts")
    if type(max_concepts) is not int or max_concepts <= 0:
        raise AxBenchAdvError("slice manifest max_concepts must be a positive integer")
    factors = _manifest_factors(manifest, methods)
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise AxBenchAdvError("slice manifest files must be a mapping")
    nl_source_identity = _nl_source_identity(manifest)
    if NL_METHOD in methods:
        if nl_source_identity is None:
            raise AxBenchAdvError("NLSteering slices require source provenance")

    pressures = parse_pressures(manifest.get("pressures", [Pressure.GEPA]), environ={})
    if pressure not in pressures:
        raise AxBenchAdvError(f"slice does not contain pressure {pressure}")
    if pressure in COUNTER_PRESSURES:
        magnitudes = validate_magnitudes(manifest.get("counter_magnitudes", []))
        if not isinstance(manifest.get("counter_directions"), list):
            raise AxBenchAdvError("slice has no counter direction provenance")
        levels: tuple[float | int, ...] = (0.0, *magnitudes)
    elif pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
        label = "GCG" if pressure == Pressure.GCG else "soft-prompt"
        config = manifest.get(pressure)
        search = (
            config.get("search")
            if pressure == Pressure.GCG and isinstance(config, Mapping)
            else config
        )
        token_counts: Any = search.get("optimized_token_counts") if isinstance(search, Mapping) else None
        if (
            not isinstance(token_counts, list)
            or not token_counts
            or any(type(value) is not int or value < 0 for value in token_counts)
            or token_counts != sorted(set(cast(list[int], token_counts)))
            or token_counts[0] != 0
        ):
            raise AxBenchAdvError(
                f"slice {label} effective config must contain an ascending token grid starting at zero"
            )
        levels = tuple(token_counts)
    elif pressure == Pressure.GEPA:
        levels = PLOT_EFFORT_LEVELS
    else:
        raise AxBenchAdvError(f"unsupported pressure {pressure!r}")
    entries: dict[float, Sequence[Mapping[str, Any]]] = {}
    concept_ids: tuple[int, ...] | None = None
    declared_concepts = manifest.get("concept_ids")
    if "concept_ids" in manifest:
        if not isinstance(declared_concepts, list):
            raise AxBenchAdvError("slice manifest concept_ids must be a list")
        concept_ids = validate_concept_ids(declared_concepts, max_concepts=max_concepts)
    required_files: list[Path] = []
    gcg_receipt_compatibility: dict[str, tuple[Mapping[str, Any], Mapping[str, Any] | None]] = {}
    for level in levels:
        if pressure == Pressure.GEPA or level == 0:
            relative = Path("points") / f"p{int(level)}" / "judged" / "steering.jsonl"
        elif pressure in COUNTER_PRESSURES:
            relative = (
                Path("counter")
                / pressure
                / "points"
                / magnitude_label(float(level))
                / "judged"
                / "steering.jsonl"
            )
        elif pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
            relative = Path(pressure) / "points" / f"k{int(level)}" / "judged" / "steering.jsonl"
        else:
            raise AxBenchAdvError(f"unsupported pressure {pressure!r}")
        path = root / relative
        point_root = relative.parents[1]
        required_files.extend(
            (
                point_root / "input" / "steering_data.parquet",
                point_root / "judged" / "steering.jsonl",
                point_root / "judged" / "steering_data.parquet",
            )
        )
        recorded_hash = files.get(str(relative))
        if not isinstance(recorded_hash, str) or _sha256(path) != recorded_hash:
            raise AxBenchAdvError(f"slice judged file hash differs from manifest: {relative}")
        if concept_ids is None:
            # A slice manifest that predates the declared ``concept_ids`` field covers an arbitrary
            # concept range, so its judged file cannot be validated against ``range(n)``: read the
            # identities it carries, validate them as a slice, then load with that expectation.
            concept_ids = validate_concept_ids(_peek_concept_ids(path), max_concepts=max_concepts)
        level_entries = load_steering_entries(
            path,
            expected_concept_ids=concept_ids,
        )
        for entry in level_entries:
            if tuple(_entry_methods(entry)) != methods:
                raise AxBenchAdvError("slice judged method coverage differs from its manifest")
        entries[level] = level_entries
        if pressure in (Pressure.GEPA, Pressure.GCG, Pressure.SOFT_PROMPT):
            input_table = pq.read_table(root / point_root / "input" / "steering_data.parquet")
            reconstructed = _reconstruct_judged_emissions(
                input_table=input_table,
                entries=level_entries,
                methods=methods,
                factors=factors,
                concept_ids=concept_ids,
            )
            validate_final_judge_outputs(
                input_table=input_table,
                output_dir=root / point_root / "judged",
                best_factors=factors,
                emissions=reconstructed,
                max_concepts=max_concepts,
                concept_ids=concept_ids,
            )
    assert concept_ids is not None
    if pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
        required_files.extend(
            _validate_search_receipts(
                root=root,
                manifest=manifest,
                files=files,
                methods=methods,
                concept_ids=concept_ids,
                factors=factors,
                token_counts=tuple(int(value) for value in levels),
                pressure=pressure,
                compatibility=gcg_receipt_compatibility,
            )
        )
    fitness: dict[tuple[str, float], list[float]] = {}
    control_raw: dict[tuple[str, int, int], Mapping[str, Any]] = {}
    emissions_by_level: dict[float, dict[tuple[str, int, int], RawEvaluation]] = {
        level: {} for level in levels
    }
    if pressure in COUNTER_PRESSURES:
        records = validate_counter_provenance(
            manifest["counter_directions"],
            pressure=pressure,
            methods=methods,
            concept_ids=concept_ids,
        )
        for record in records:
            method, concept_id = record["method"], record["concept_id"]
            for magnitude in levels:
                relative = (
                    Path("counter")
                    / pressure
                    / method
                    / f"c{concept_id}"
                    / magnitude_label(float(magnitude))
                    / "point.json"
                )
                path = root / relative
                required_files.append(relative)
                if files.get(str(relative)) != _sha256(path):
                    raise AxBenchAdvError(f"counter point hash differs from manifest: {relative}")
                document = _json_mapping(path, label="counter point")
                document_magnitude = document.get("magnitude")
                if (
                    document.get("direction") != record
                    or isinstance(document_magnitude, bool)
                    or not isinstance(document_magnitude, (int, float))
                    or not math.isfinite(float(document_magnitude))
                    or float(document_magnitude) != magnitude
                ):
                    raise AxBenchAdvError(f"counter point identity differs: {relative}")
                evaluations = document.get("evaluations")
                if (
                    not isinstance(evaluations, list)
                    or any(not isinstance(row, Mapping) for row in evaluations)
                    or any(
                        type(row.get("input_id")) is not int
                        for row in evaluations
                        if isinstance(row, Mapping)
                    )
                    or [row.get("input_id") for row in evaluations if isinstance(row, Mapping)]
                    != list(range(10))
                ):
                    raise AxBenchAdvError(f"counter point must have ten ordered inputs: {relative}")
                evaluation_rows = [row for row in evaluations if isinstance(row, Mapping)]
                raw = [_raw_evaluation(row, relative=relative) for row in evaluation_rows]
                for row, evaluation in zip(evaluation_rows, raw, strict=True):
                    coordinate = (method, concept_id, int(row["input_id"]))
                    emissions_by_level[magnitude][coordinate] = evaluation
                    if magnitude == 0:
                        control_raw[coordinate] = {
                            key: value
                            for key, value in row.items()
                            if key not in {"input_id", "paid_judge_cache_misses"}
                        }
                score = mean_counter_fitness(raw)
                recorded_fitness = document.get("mean_fitness")
                if (
                    isinstance(recorded_fitness, bool)
                    or not isinstance(recorded_fitness, (int, float))
                    or not math.isfinite(float(recorded_fitness))
                    or score != float(recorded_fitness)
                ):
                    raise AxBenchAdvError(f"counter point fitness differs: {relative}")
                fitness.setdefault((method, magnitude), []).append(score)
        for magnitude in levels:
            point_root = (
                root / "points" / "p0"
                if magnitude == 0
                else root / "counter" / pressure / "points" / magnitude_label(float(magnitude))
            )
            validate_final_judge_outputs(
                input_table=pq.read_table(point_root / "input" / "steering_data.parquet"),
                output_dir=point_root / "judged",
                best_factors=factors,
                emissions=emissions_by_level[magnitude],
                max_concepts=max_concepts,
                concept_ids=concept_ids,
            )
    _validate_recorded_inventory(root, files, required=required_files)
    controls = _control_records(
        root=root,
        methods=methods,
        factors=factors,
        concept_ids=concept_ids,
        max_concepts=max_concepts,
        entries=entries[0],
        raw_records=control_raw,
    )
    return _RunSlice(
        root=root,
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        methods=methods,
        concept_ids=concept_ids,
        factors=factors,
        entries=entries,
        fitness=fitness,
        controls=controls,
        gcg_receipt_compatibility=gcg_receipt_compatibility,
    )


def _compatible(first: _RunSlice, other: _RunSlice) -> None:
    mismatches = [
        field
        for field in _SCIENTIFIC_COMPATIBILITY_FIELDS
        if first.manifest.get(field) != other.manifest.get(field)
    ]
    if mismatches:
        raise AxBenchAdvError(f"slice manifests have incompatible fields: {mismatches}")
    first_project = first.manifest["source_project"]
    other_project = other.manifest["source_project"]
    if first_project.get("self_conditioning_masked_policy") != other_project.get(
        "self_conditioning_masked_policy"
    ):
        raise AxBenchAdvError("slice source projects use incompatible masked policies")
    for runtime_kind in set(first.gcg_receipt_compatibility) & set(other.gcg_receipt_compatibility):
        if (
            first.gcg_receipt_compatibility[runtime_kind]
            != other.gcg_receipt_compatibility[runtime_kind]
        ):
            raise AxBenchAdvError(
                f"GCG {runtime_kind} runtime/calibration identities differ across slices"
            )


def _control_records_match(first: _ControlRecord, other: _ControlRecord) -> bool:
    if first.prompt != other.prompt or first.judged != other.judged:
        return False
    if first.raw is None or other.raw is None:
        return True
    # Optional seed/token receipts are compared whenever both sources recorded
    # them.  Their absence in a legacy source remains an absence, not an inferred
    # value.  Operational cache accounting was removed while loading the record.
    shared = set(first.raw) & set(other.raw)
    return all(first.raw[field] == other.raw[field] for field in shared)


def _validate_across_pressures(groups: Mapping[Pressure, Sequence[_RunSlice]]) -> None:
    """Bind factors, controls, and NL provenance across pressure families."""

    factors: dict[str, tuple[float, Pressure]] = {}
    controls: dict[tuple[str, int, int], tuple[_ControlRecord, Pressure]] = {}
    nl_source_identity: tuple[str, Mapping[str, Any]] | None = None
    for pressure, slices in groups.items():
        for run_slice in slices:
            if NL_METHOD in run_slice.methods:
                identity = _nl_source_identity(run_slice.manifest)
                assert identity is not None
                if nl_source_identity is not None and identity != nl_source_identity:
                    raise AxBenchAdvError("NLSteering slices have incompatible source provenance")
                nl_source_identity = identity
            for method, factor in run_slice.factors.items():
                previous = factors.get(method)
                if previous is not None and previous[0] != factor:
                    raise AxBenchAdvError(
                        "selected factor differs across pressures for "
                        f"{method}: {previous[0]} ({previous[1]}) != {factor} ({pressure})"
                    )
                factors[method] = (factor, pressure)
            for coordinate, control in run_slice.controls.items():
                previous = controls.get(coordinate)
                if (
                    previous is not None
                    and previous[1] != pressure
                    and not _control_records_match(previous[0], control)
                ):
                    raise AxBenchAdvError(
                        "shared zero control differs across pressures for "
                        f"{coordinate}: {previous[1]} != {pressure}"
                    )
                controls[coordinate] = (control, pressure)


def _validate_complete_campaign(groups: Mapping[Pressure, Sequence[_RunSlice]]) -> None:
    """Require the exact issue #1673 publication matrix when explicitly requested."""

    if tuple(groups) != LEGACY_PRESSURES:
        raise AxBenchAdvError("complete campaign requires all three pressure families")
    expected_coordinates = {
        (method, concept_id) for method in ALL_METHODS for concept_id in range(50)
    }
    for pressure, slices in groups.items():
        expected_levels = (
            tuple(float(level) for level in PLOT_EFFORT_LEVELS)
            if pressure == Pressure.GEPA
            else (0.0, *DEFAULT_MAGNITUDES)
        )
        observed: set[tuple[str, int]] = set()
        for run_slice in slices:
            if run_slice.manifest.get("max_concepts") != 50:
                raise AxBenchAdvError("complete campaign requires max_concepts=50")
            if tuple(float(level) for level in run_slice.entries) != expected_levels:
                raise AxBenchAdvError(f"complete campaign has wrong five-level grid for {pressure}")
            for method in run_slice.methods:
                for concept_id in run_slice.concept_ids:
                    coordinate = (method, concept_id)
                    if coordinate in observed:
                        raise AxBenchAdvError(
                            f"complete campaign repeats {pressure} coordinate {coordinate}"
                        )
                    observed.add(coordinate)
        if observed != expected_coordinates:
            missing = sorted(expected_coordinates - observed)
            extra = sorted(observed - expected_coordinates)
            raise AxBenchAdvError(
                f"complete campaign coverage differs for {pressure}; "
                f"missing={missing[:5]}, extra={extra[:5]}"
            )


def _fresh_output(output_root: Path, input_roots: Sequence[Path]) -> Path:
    output = Path(output_root).expanduser().resolve()
    for source in input_roots:
        if output == source or output.is_relative_to(source) or source.is_relative_to(output):
            raise AxBenchAdvError("merge output must be path-disjoint from every input root")
    if output.exists():
        if not output.is_dir() or next(output.iterdir(), None) is not None:
            raise AxBenchAdvError(f"merge output directory must be fresh: {output}")
    else:
        output.mkdir(parents=True)
    return output


def _gcg_training_source_compatibility(value: object) -> object:
    """Compare content identities while retaining absolute paths as provenance."""

    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    for source_field in ("train_data", "metadata"):
        record = normalized.get(source_field)
        if isinstance(record, Mapping):
            normalized[source_field] = {key: item for key, item in record.items() if key != "path"}
    return normalized


def _merge_pressure_slices(
    slices: Sequence[_RunSlice],
    output_root: Path,
    pressure: Pressure,
    *,
    manifest_name: str = "manifest.json",
) -> Path:
    """Validate completed roots and publish one aggregate plus plot family."""

    roots = tuple(run_slice.root for run_slice in slices)
    levels = tuple(slices[0].entries)
    if pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
        label = "GCG" if pressure == Pressure.GCG else "soft-prompt"
        effective = slices[0].manifest.get(pressure)
        source_key = f"{pressure}_training_source"
        training_source = slices[0].manifest.get(source_key)
        for run_slice in slices[1:]:
            if run_slice.manifest.get(pressure) != effective:
                raise AxBenchAdvError(f"{label} slices have incompatible effective configs")
            if _gcg_training_source_compatibility(
                run_slice.manifest.get(source_key)
            ) != _gcg_training_source_compatibility(training_source):
                raise AxBenchAdvError(f"{label} slices have incompatible training source identities")
    for index, run_slice in enumerate(slices[1:], start=1):
        for previous in slices[:index]:
            _compatible(previous, run_slice)
        if tuple(run_slice.entries) != levels:
            raise AxBenchAdvError(f"slice magnitude grids differ for {pressure}")
    counter_records = []
    direction_identities = {}
    if pressure in COUNTER_PRESSURES:
        for run_slice in slices:
            records = validate_counter_provenance(
                run_slice.manifest["counter_directions"],
                pressure=pressure,
                methods=run_slice.methods,
                concept_ids=run_slice.concept_ids,
            )
            for record in records:
                coordinate = (record["target_model"], record["concept_id"])
                identity = tuple(
                    record.get(field)
                    for field in (
                        "sha256",
                        "layer",
                        "family",
                        "mode",
                        "coefficient_semantics",
                        "direction_norm",
                        "fit_manifest_sha256",
                        "fit_identity_sha256",
                    )
                )
                if (
                    coordinate in direction_identities
                    and direction_identities[coordinate] != identity
                ):
                    raise AxBenchAdvError(
                        f"incompatible counter directions for {pressure}/{coordinate}"
                    )
                direction_identities[coordinate] = identity
            counter_records.extend(records)
    factors: dict[str, float] = {}
    ownership: dict[tuple[str, int], Path] = {}
    judged_points: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for run_slice in slices:
        for method, factor in run_slice.factors.items():
            if method in factors and factors[method] != factor:
                raise AxBenchAdvError(f"slice factor differs for repeated method {method}")
            factors[method] = factor
            for concept_id in run_slice.concept_ids:
                key = (method, concept_id)
                if key in ownership:
                    raise AxBenchAdvError(
                        "slice roots overlap the same (method, concept): "
                        f"{key} in {ownership[key]} and {run_slice.root}"
                    )
                ownership[key] = run_slice.root
        for level, level_entries in run_slice.entries.items():
            for method in run_slice.methods:
                judged_points.setdefault((method, level), []).extend(level_entries)

    methods = validate_method_subset(factors)
    if tuple(factors) != methods:
        factors = {method: factors[method] for method in methods}
    concept_sets = {
        method: frozenset(
            concept_id for owned_method, concept_id in ownership if owned_method == method
        )
        for method in methods
    }
    populations = set(concept_sets.values())
    if len(populations) != 1:
        detail = {method: sorted(concepts) for method, concepts in concept_sets.items()}
        raise AxBenchAdvError(f"merged method concept sets do not match: {detail}")
    concept_ids = tuple(sorted(next(iter(populations))))
    max_concepts = slices[0].manifest["max_concepts"]
    validate_concept_ids(concept_ids, max_concepts=max_concepts)

    output = _fresh_output(output_root, roots)
    for entries in judged_points.values():
        entries.sort(key=lambda entry: entry["concept_id"])
    aggregate_path = output / "aggregate.parquet"
    if pressure == Pressure.GEPA:
        build_aggregate(
            best_factors=factors,
            judged_points={
                (method, int(level)): entries for (method, level), entries in judged_points.items()
            },
            expected_n_concepts=len(concept_ids),
            expected_concept_ids=concept_ids,
            output_path=aggregate_path,
        )
        plot_paths = plot_concept_vs_fluency(aggregate_path, output / "plots")
    elif pressure in COUNTER_PRESSURES:
        aggregate = build_counter_aggregate(
            best_factors=factors,
            judged_points=judged_points,
            magnitudes=levels[1:],
            fitness={
                (method, level): sum(scores) / len(scores)
                for method in methods
                for level in levels
                if (
                    scores := [
                        score
                        for run_slice in slices
                        for score in run_slice.fitness.get((method, level), ())
                    ]
                )
            },
            concept_ids=concept_ids,
            output_path=aggregate_path,
        )
        best_path = write_counter_best_points(aggregate, output / "selection")
        plot_paths = [
            *plot_concept_vs_fluency(aggregate, output / "plots", pressure=pressure),
            best_path,
        ]
    elif pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
        token_counts = tuple(int(level) for level in levels)
        build_gcg_aggregate(
            best_factors=factors,
            judged_points={
                (method, int(level)): entries for (method, level), entries in judged_points.items()
            },
            token_counts=token_counts,
            concept_ids=concept_ids,
            output_path=aggregate_path,
        )
        plot_paths = plot_concept_vs_fluency(
            aggregate_path,
            output / "plots",
            pressure=pressure,
        )
    else:
        raise AxBenchAdvError(f"unsupported pressure {pressure!r}")
    files = {str(path.relative_to(output)): _sha256(path) for path in (aggregate_path, *plot_paths)}
    manifest_path = output / manifest_name
    atomic_write_text(
        json.dumps(
            {
                **(
                    {
                        "pressure": pressure,
                        "counter_magnitudes": list(levels[1:]),
                        "counter_directions": counter_records,
                    }
                    if pressure in COUNTER_PRESSURES
                    else (
                        {
                            "pressure": pressure,
                            pressure: slices[0].manifest[pressure],
                            f"{pressure}_training_source": slices[0].manifest[f"{pressure}_training_source"],
                        }
                        if pressure in (Pressure.GCG, Pressure.SOFT_PROMPT)
                        else {}
                    )
                ),
                "schema_version": 1,
                "methods": list(methods),
                "concept_ids": list(concept_ids),
                "best_factors": factors,
                "input_roots": [
                    {
                        "path": str(run_slice.root),
                        "manifest_sha256": run_slice.manifest_sha256,
                    }
                    for run_slice in slices
                ],
                "files": files,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        manifest_path,
    )
    return manifest_path


def merge_slices(
    input_roots: Sequence[Path],
    output_root: Path,
    *,
    require_complete_campaign: bool = False,
) -> Path:
    """Merge validated pressure slices, optionally enforcing the full #1673 matrix."""
    roots = tuple(Path(root).expanduser().resolve() for root in input_roots)
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise AxBenchAdvError("merge requires at least two distinct input roots")
    inventories = {
        root: parse_pressures(
            _json_mapping(root / "manifest.json", label="completed slice manifest").get(
                "pressures", [Pressure.GEPA]
            ),
            environ={},
        )
        for root in roots
    }
    pressures = tuple(
        p for p in Pressure if any(p in inventory for inventory in inventories.values())
    )
    # Validate all source point hashes before creating output directories.
    groups = {
        p: tuple(_load_run_slice(root, p) for root in roots if p in inventories[root])
        for p in pressures
    }
    _validate_across_pressures(groups)
    if require_complete_campaign:
        _validate_complete_campaign(groups)
    if pressures == (Pressure.GEPA,):
        return _merge_pressure_slices(groups[Pressure.GEPA], output_root, Pressure.GEPA)
    all_slices = [item for group in groups.values() for item in group]
    for index, item in enumerate(all_slices[1:], start=1):
        for previous in all_slices[:index]:
            _compatible(previous, item)
    output = _fresh_output(output_root, roots)
    for pressure, group in groups.items():
        # GEPA keeps its original root layout. Its inner manifest is replaced
        # by the merged inventory only after every pressure has succeeded.
        if pressure == Pressure.GEPA:
            destination = output
        elif pressure in COUNTER_PRESSURES:
            destination = output / "counter" / pressure
        elif pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
            destination = output / pressure
        else:
            raise AxBenchAdvError(f"unsupported pressure {pressure!r}")
        _merge_pressure_slices(
            group,
            destination,
            pressure,
            manifest_name="gepa_manifest.json" if pressure == Pressure.GEPA else "manifest.json",
        )
    document = {
        "schema_version": 1,
        "pressures": list(pressures),
        "files": {
            str(path.relative_to(output)): _sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    if require_complete_campaign:
        document["campaign_completion"] = {
            "schema_version": 1,
            "campaign": "issue-1673-qualified-counters",
            "pressures": list(LEGACY_PRESSURES),
            "methods": list(ALL_METHODS),
            "concept_ids": list(range(50)),
            "inputs_per_concept": 10,
            "levels": {
                Pressure.GEPA: [float(level) for level in PLOT_EFFORT_LEVELS],
                Pressure.LINEAR_PROBE_COUNTER: [0.0, *DEFAULT_MAGNITUDES],
                Pressure.DIFFMEAN_COUNTER: [0.0, *DEFAULT_MAGNITUDES],
            },
        }
    document["input_roots"] = [
        {
            "path": str(root),
            "manifest_sha256": next(
                item.manifest_sha256 for item in all_slices if item.root == root
            ),
        }
        for root in roots
    ]
    manifest = output / "manifest.json"
    atomic_write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", manifest)
    return manifest


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--require-complete-campaign",
        action="store_true",
        help="Require the exact 3-pressure x 7-method x 50-concept issue #1673 matrix",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    merge_slices(
        args.input_root,
        args.output_root,
        require_complete_campaign=args.require_complete_campaign,
    )


if __name__ == "__main__":
    main()


__all__ = ["merge_slices"]
