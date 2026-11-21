"""Frozen negative-source evaluation with paired controls and explicit coverage."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from steering.eval.axbench_negative.api import (
    ApiResponse,
    QualityAssessment,
    canonical_json,
    parse_presence_response,
    strict_object,
)
from steering.eval.axbench_negative.source import FeatureTarget, SourceCandidate
from steering.eval.axbench_negative.targets import digest, file_sha256, validate_methods
from steering.eval.axbench_upstream.native_decoder import canonical_bytes
from steering.immutable_io import install_bytes_once

FORMAT = "axbench-natural-chat-negative-evaluation-v1"
SEED_DOMAIN = "axbench-natural-chat-negative-paired-v1"


@dataclass(frozen=True)
class EvaluationConfig:
    output_dir: Path
    split: str = "test"
    magnitudes: tuple[float, ...] = (1.0, 3.0, 6.0)
    replicates: int = 1
    seed: int = 42
    max_new_tokens: int = 128
    max_cells: int | None = None

    def validate(self, family: str) -> None:
        if self.split not in {"test", "validation"}:
            raise ValueError("negative evaluation requires frozen validation or test data")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("evaluation seed must be a nonnegative integer")
        for name in ("replicates", "max_new_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_cells is not None and (type(self.max_cells) is not int or self.max_cells <= 0):
            raise ValueError("max_cells must be a positive integer or null")
        if not self.magnitudes or len(set(self.magnitudes)) != len(self.magnitudes):
            raise ValueError("declare a nonempty unique magnitude grid; control zero is added separately")
        for value in self.magnitudes:
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError("suppression magnitudes must be positive finite numbers")
            if family == "encdec" and value > 6:
                raise ValueError("SelfConditioning magnitude cannot exceed 6")


def paired_seed(base_seed: int, family: str, messages_sha256: str, replicate: int, kind: str) -> int:
    """Method, concept and strength do not affect paired sampling randomness."""
    value = digest([SEED_DOMAIN, base_seed, family, messages_sha256, replicate, kind])
    return int(value[:16], 16) % (2**31 - 1)


def _candidate(group: Any) -> SourceCandidate:
    return SourceCandidate(**{key: getattr(group, key) for key in (
        "source_id", "group_id", "split", "source_row", "cut_message_count", "messages", "messages_sha256", "candidate_id",
    )})


def build_grid(groups: Sequence[Any], target: Any, config: EvaluationConfig, bank_sha: str) -> tuple[dict[str, Any], ...]:
    config.validate(target.family)
    validate_methods(target.family, target.methods)
    target_sha = digest(target.identity)
    cells = []
    seen_groups = set()
    for group in groups:
        group_key = (group.candidate_id, group.feature_id)
        if group_key in seen_groups:
            raise ValueError("duplicate candidate/feature group in evaluation population")
        seen_groups.add(group_key)
        for replicate in range(config.replicates):
            for method, magnitude in (("Control", 0.0), *((method, float(value))
                for method in target.methods for value in config.magnitudes)):
                cell = {"format": FORMAT, "source_manifest_sha256": bank_sha, "target_sha256": target_sha,
                        "family": target.family, "candidate_id": group.candidate_id, "feature_id": group.feature_id,
                        "messages_sha256": group.messages_sha256, "method": method,
                        "polarity": "control" if method == "Control" else "negative", "magnitude_hex": magnitude.hex(),
                        "replicate": replicate, "max_new_tokens": config.max_new_tokens,
                        "seed": paired_seed(config.seed, target.family, group.messages_sha256, replicate, "tokens"),
                        "selection_seed": paired_seed(config.seed, target.family, group.messages_sha256, replicate, "features")}
                cells.append({**cell, "cell_id": digest(cell)})
    if not cells or len({cell["cell_id"] for cell in cells}) != len(cells):
        raise ValueError("empty or colliding negative evaluation cell grid")
    return tuple(cells)


def _install(path: Path, value: Any) -> None:
    install_bytes_once(canonical_bytes(value), path, surface="negative AxBench evaluation")


def _save_record(path: Path, cell: Mapping[str, Any], payload: Mapping[str, Any]) -> dict[str, Any]:
    record = {"cell": dict(cell), "payload": dict(payload), "payload_sha256": digest(payload)}
    _install(path, record)
    return dict(payload)


def _load_record(path: Path, cell: Mapping[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = strict_object(path.read_text())
    if set(value) != {"cell", "payload", "payload_sha256"} or value["cell"] != dict(cell):
        raise ValueError("negative evaluation cell identity differs")
    payload = value["payload"]
    if not isinstance(payload, dict) or digest(payload) != value["payload_sha256"]:
        raise ValueError("negative evaluation cell payload hash differs")
    return payload


def _error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:2000]}


def _assess(group: Any, generated: Mapping[str, Any], scorer: Any, quality_judge: Any, presence_judge: Any, key: str) -> dict[str, Any]:
    result: dict[str, Any] = {"sae": None, "quality": None, "presence": None, "errors": {}}
    if generated["status"] == "failed":
        return result
    response = generated["text"]
    try:
        result["sae"] = scorer.assess_completion(_candidate(group), response,
            FeatureTarget(group.feature_id, group.member_ids, group.description)).to_dict()
    except Exception as exc:
        result["errors"]["sae"] = _error(exc)
    try:
        quality_key = digest({"namespace": "negative-target-response-quality-v1", "messages_sha256": group.messages_sha256,
                              "response": response, "judge": quality_judge.identity})
        result["quality"] = quality_judge.assess(group.messages, response, cache_key=quality_key).to_dict()
    except Exception as exc:
        result["errors"]["quality"] = _error(exc)
    try:
        result["presence"] = presence_judge.assess(group.messages, response, group.description, cache_key=f"{key}:presence")
    except Exception as exc:
        result["errors"]["presence"] = _error(exc)
    return result


def summarize(cells: Sequence[Mapping[str, Any]], records: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Keep unknowns and invalid generations out of observed-effect denominators."""
    arms: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    by_pair = {}
    for cell in cells:
        arms[(cell["method"], cell["magnitude_hex"])].append(cell)
        if cell["method"] == "Control":
            by_pair[(cell["candidate_id"], cell["feature_id"], cell["replicate"])] = cell["cell_id"]
    summaries = []
    for (method, magnitude), rows in arms.items():
        counts: Counter[str] = Counter()
        activity_deltas, quality_deltas, presence_deltas, lengths, tokens = [], [], [], [], []
        for cell in rows:
            record = records.get(cell["cell_id"])
            if record is None:
                counts["pending"] += 1
                continue
            generation, assessment = record["generation"], record["assessment"]
            counts["attempted"] += 1
            counts[generation["status"]] += 1
            if generation["status"] != "failed":
                lengths.append(len(generation["text"]))
                tokens.append(generation["token_count"])
            sae, quality, presence = assessment["sae"], assessment["quality"], assessment["presence"]
            valid = generation["status"] == "complete"
            measured = sae is not None and sae.get("status") in {"active", "inactive"}
            judged = quality is not None and quality.get("status") == "assessed"
            present = presence is not None and presence.get("status") == "assessed"
            counts["sae_measured"] += int(measured)
            counts["sae_no_tokens"] += int(sae is not None and sae.get("status") == "no_tokens")
            counts["sae_unknown"] += int(sae is None)
            counts["quality_judged"] += int(judged)
            counts["quality_unknown"] += int(not judged)
            counts["presence_judged"] += int(present)
            counts["presence_unknown"] += int(not present)
            if measured and valid:
                counts["valid_sae_observations"] += 1
                counts["active"] += int(sae["status"] == "active")
            if judged:
                counts["refusals"] += int(quality["refusal"])
                counts["task_fulfilled"] += int(quality["task_fulfilled"] and valid)
            if present and valid:
                counts["valid_presence_observations"] += 1
                counts["concept_present"] += int(presence["concept_present"])
            counts["useful_strict_zero"] += int(valid and measured and sae["status"] == "inactive"
                                                and sae.get("max_activation") == 0 and judged and quality["useful"])
            control_id = by_pair[(cell["candidate_id"], cell["feature_id"], cell["replicate"])]
            control = records.get(control_id)
            if method == "Control" or control is None or not valid or control["generation"]["status"] != "complete":
                continue
            ca = control["assessment"]
            if measured and ca["sae"] is not None and ca["sae"].get("status") in {"active", "inactive"}:
                activity_deltas.append(int(sae["status"] == "active") - int(ca["sae"]["status"] == "active"))
            if judged and ca["quality"] is not None and ca["quality"].get("status") == "assessed":
                quality_deltas.append(int(quality["useful"]) - int(ca["quality"]["useful"]))
            if present and ca["presence"] is not None and ca["presence"].get("status") == "assessed":
                presence_deltas.append(int(presence["concept_present"]) - int(ca["presence"]["concept_present"]))
        def mean(values: Sequence[float]) -> float | None:
            return sum(values) / len(values) if values else None
        summaries.append({"method": method, "magnitude_hex": magnitude, "declared": len(rows), "counts": dict(counts),
                          "mean_characters": mean(lengths), "mean_tokens": mean(tokens),
                          "paired_active_change": {"n": len(activity_deltas), "mean": mean(activity_deltas)},
                          "paired_useful_change": {"n": len(quality_deltas), "mean": mean(quality_deltas)},
                          "paired_concept_presence_change": {"n": len(presence_deltas), "mean": mean(presence_deltas)}})
    return {"declared_cells": len(cells), "recorded_cells": len(records), "pending_cells": len(cells) - len(records),
            "arms": summaries, "effect_denominators": "complete generated responses with observed paired assessments only"}


def evaluate_negative_source(bank: Any, target: Any, scorer: Any, quality_judge: Any,
                             presence_judge: Any, config: EvaluationConfig) -> dict[str, Any]:
    """Evaluate every declared cell, persisting outputs before any judge call."""
    config.validate(target.family)
    root = Path(config.output_dir).resolve()
    if root == bank.root.resolve() or root.is_relative_to(bank.root.resolve()) or bank.root.resolve().is_relative_to(root):
        raise ValueError("evaluation output and frozen bank must be disjoint")
    if dict(scorer.identity) != bank.identity.get("scorer"):
        raise ValueError("evaluation SAE assessment identity differs from the frozen source")
    groups = tuple(group for group in bank.groups if group.split == config.split)
    cells = build_grid(groups, target, config, bank.manifest_sha256)
    population = {(group.candidate_id, group.feature_id): group for group in groups}
    settings = {key: value for key, value in asdict(config).items() if key not in {"output_dir", "max_cells"}}
    identity = {"format": FORMAT, "source_manifest_sha256": bank.manifest_sha256, "target": target.identity,
                "scorer": dict(scorer.identity), "quality_judge": quality_judge.identity,
                "presence_judge": presence_judge.identity, "configuration": settings,
                "grid_sha256": digest(cells), "declared_cells": len(cells)}
    _install(root / "identity.json", identity)
    _install(root / "grid.json", list(cells))
    final_path = root / "manifest.json"
    if final_path.exists():
        previous = strict_object(final_path.read_text())
        if previous.get("identity_sha256") != digest(identity):
            raise ValueError("completed negative evaluation identity differs")
        for item in previous.get("files", []):
            if file_sha256(root / item["path"]) != item["sha256"]:
                raise ValueError("completed negative evaluation artifact hash differs")
    records, new_cells = {}, 0
    control_generations: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        key = cell["cell_id"]
        generation_path, assessment_path = root / "cells" / f"{key}.generation.json", root / "cells" / f"{key}.assessment.json"
        generation = _load_record(generation_path, cell)
        assessment = _load_record(assessment_path, cell)
        control_key = (cell["messages_sha256"], cell["replicate"])
        if cell["method"] == "Control" and generation is not None:
            previous_control = control_generations.setdefault(control_key, generation)
            if previous_control != generation:
                raise ValueError("one prompt/replicate has conflicting control generations")
        if assessment is not None and generation is None:
            raise ValueError("negative cell assessment has no preceding generation")
        if assessment is not None:
            if assessment.pop("generation_sha256", None) != digest(generation):
                raise ValueError("assessment refers to different generated bytes")
            if assessment["quality"] is not None:
                QualityAssessment.from_dict(assessment["quality"])
            presence = assessment["presence"]
            if presence is not None:
                if not isinstance(presence, dict) or not isinstance(presence.get("api_response"), dict):
                    raise ValueError("persisted presence assessment lacks provider evidence")
                observed = parse_presence_response(ApiResponse.from_dict(presence["api_response"]))
                if canonical_json(presence) != canonical_json(observed):
                    raise ValueError("persisted presence contradicts its provider evidence")
            records[key] = {"generation": generation, "assessment": assessment}
            continue
        if config.max_cells is not None and new_cells >= config.max_cells:
            continue
        new_cells += 1
        group = population[(cell["candidate_id"], cell["feature_id"])]
        if generation is None and cell["method"] == "Control":
            generation = control_generations.get(control_key)
            if generation is not None:
                _save_record(generation_path, cell, generation)
        if generation is None:
            try:
                result = target.generate(group, method=cell["method"], magnitude=float.fromhex(cell["magnitude_hex"]),
                    seed=cell["seed"], selection_seed=cell["selection_seed"], max_new_tokens=config.max_new_tokens)
                status = "empty" if not result.text.strip() else "truncated" if result.budget_exhausted else "complete"
                generation = {"status": status, "text": result.text, "token_ids": list(result.token_ids),
                              "token_count": len(result.token_ids), "budget_exhausted": result.budget_exhausted, "error": None}
            except Exception as exc:
                generation = {"status": "failed", "text": None, "token_ids": None, "token_count": None,
                              "budget_exhausted": None, "error": _error(exc)}
            _save_record(generation_path, cell, generation)
            if cell["method"] == "Control":
                control_generations[control_key] = generation
        assessment = _assess(group, generation, scorer, quality_judge, presence_judge, key)
        _save_record(assessment_path, cell, {**assessment, "generation_sha256": digest(generation)})
        records[key] = {"generation": generation, "assessment": assessment}
    summary = summarize(cells, records)
    if len(records) == len(cells):
        _install(root / "summary.json", summary)
        paths = [root / "identity.json", root / "grid.json", root / "summary.json"]
        paths.extend(root / "cells" / f"{cell['cell_id']}.{stage}.json" for cell in cells for stage in ("generation", "assessment"))
        manifest = {"format": FORMAT, "complete": True, "identity_sha256": digest(identity), "summary": summary,
                    "files": [{"path": str(path.relative_to(root)), "sha256": file_sha256(path)} for path in paths]}
        _install(final_path, manifest)
    return {"complete": len(records) == len(cells), "summary": summary, "output_dir": str(root)}
