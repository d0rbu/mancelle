"""Frozen Campaign 2 DiffMean pairs and their CPU admission contract."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import hydra
from datasets import load_dataset
from loguru import logger
from omegaconf import DictConfig

from steering._common.strict_json import reject_nonstandard_json_constant, strict_json_object
from steering.eval.chat_benchmark_targets import normalize_input
from steering.eval.dataset_revision import validate_dataset_revision
from steering.eval.harmfulness_safety_targets import (
    FIT_SHUFFLE_SEED,
    SAFETY_EVALUATION_SOURCE_NAMES,
    SafetySourceName,
    SafetySourceSpec,
    load_populations,
    safety_evaluation_input_keys,
)
from steering.immutable_io import install_bytes_once

MAX_PAIRS = 128
TRAINING_SPEC = SafetySourceSpec(
    "PKU-Alignment/BeaverTails", "default", "330k_train",
    "8401fe609d288129cc684a9b3be6a93e41cfe678",
)


PAIR_FORMAT = "campaign2_diffmean_pairs_v1"
PAIR_RULE = "first_native_responses_exact_prompt_normalized_dedupe_python_shuffle_v1"
PAIR_JSONL = "pairs.jsonl"
PAIR_MANIFEST = "pairs.manifest.json"
COUNT_KEYS = ("rows_seen", "prompts_seen", "excluded_eval", "unpaired",
              "identical_responses", "duplicate_prompts", "paired", "capped", "selected")

class SafetyPair(TypedDict):
    prompt: str
    safe_response: str
    unsafe_response: str

def validate_cap(cap: int) -> None:
    if type(cap) is not int or not 1 <= cap <= MAX_PAIRS:
        raise ValueError(f"pair cap must be an integer in [1, {MAX_PAIRS}]")

def validate_pairs(rows: Any) -> list[SafetyPair]:
    if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_PAIRS:
        raise ValueError(f"safety training pairs require 1..{MAX_PAIRS} rows")
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(SafetyPair.__annotations__):
            raise ValueError("safety pairs require exactly prompt, safe_response, unsafe_response")
        if any(not isinstance(value, str) or not value.strip() for value in row.values()):
            raise ValueError("safety pair fields must be nonempty strings")
        if row["safe_response"].strip() == row["unsafe_response"].strip():
            raise ValueError("safe and unsafe responses must differ")
        key = normalize_input(row["prompt"])
        if key in seen:
            raise ValueError("safety pairs contain duplicate normalized prompts")
        seen.add(key)
    return rows

def decode_pairs(payload: bytes) -> list[SafetyPair]:
    # Blank lines are malformed JSONL, not silently dropped training records.
    return validate_pairs([json.loads(line, object_pairs_hook=strict_json_object, parse_constant=reject_nonstandard_json_constant)
                           for line in payload.decode("utf-8").splitlines()])

def safety_pairs(path: Path) -> list[SafetyPair]:
    return decode_pairs(path.read_bytes())

def pair_rows(training: Sequence[Mapping], evaluation: Sequence[Mapping], *, cap: int = MAX_PAIRS
              ) -> tuple[list[SafetyPair], dict[str, int]]:
    """Never combine responses from differently spelled raw prompts."""
    validate_cap(cap)
    forbidden = safety_evaluation_input_keys(evaluation)
    groups: dict[str, dict[bool, str]] = {}
    counts = dict.fromkeys(COUNT_KEYS, 0)
    for index, row in enumerate(training):
        prompt, response, safe = row.get("prompt"), row.get("response"), row.get("is_safe")
        if any(not isinstance(value, str) or not value.strip() for value in (prompt, response)):
            raise ValueError(f"training row {index}: prompt and response must be nonblank strings")
        if type(safe) is not bool:
            raise ValueError(f"training row {index}: is_safe must be a boolean")
        counts["rows_seen"] += 1
        groups.setdefault(prompt, {}).setdefault(safe, response)
    counts["prompts_seen"] = len(groups)
    candidates: list[SafetyPair] = []
    seen = set()
    for prompt, responses in groups.items():
        key = normalize_input(prompt)
        if key in forbidden:
            counts["excluded_eval"] += 1
            continue
        if len(responses) != 2:
            counts["unpaired"] += 1
            continue
        if responses[True].strip() == responses[False].strip():
            counts["identical_responses"] += 1
            continue
        if key in seen:
            counts["duplicate_prompts"] += 1
            continue
        pair = SafetyPair(prompt=prompt, safe_response=responses[True], unsafe_response=responses[False])
        validate_pairs([pair])
        candidates.append(pair)
        seen.add(key)
    random.Random(FIT_SHUFFLE_SEED).shuffle(candidates)
    counts["paired"] = len(candidates)
    selected = validate_pairs(candidates[:cap])
    counts["selected"] = len(selected)
    counts["capped"] = counts["paired"] - counts["selected"]
    return selected, counts

def load_source(spec: SafetySourceSpec) -> Any:
    rows = load_dataset(spec.dataset, spec.config, split=spec.split, revision=spec.revision)
    validate_dataset_revision(rows, spec.revision)
    return rows

def load_evaluation() -> tuple[list[dict], dict[str, dict]]:
    loaded, provenance = load_populations(SAFETY_EVALUATION_SOURCE_NAMES)
    for population in (loaded[name] for name in SAFETY_EVALUATION_SOURCE_NAMES):
        # Validate each population independently: an empty split must fail.
        safety_evaluation_input_keys(population)
    rows = [row for name in SAFETY_EVALUATION_SOURCE_NAMES for row in loaded[name]]
    return rows, provenance

def manifest_path(path: Path) -> Path:
    return path.with_suffix(".manifest.json")

def write_pairs(output: Path, training: Sequence[Mapping], evaluation: Sequence[Mapping],
                sources: dict[str, dict], *, cap: int = MAX_PAIRS) -> dict:
    pairs, counts = pair_rows(training, evaluation, cap=cap)
    payload = b"".join((json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
                       for row in pairs)
    manifest = {
        "format": PAIR_FORMAT, "rule": PAIR_RULE, "seed": FIT_SHUFFLE_SEED, "cap": cap,
        "sources": sources, "counts": counts, "row_count": len(pairs),
        "evaluation_prompt_count": len(safety_evaluation_input_keys(evaluation)),
        "evaluation_overlap": 0, "jsonl_sha256": hashlib.sha256(payload).hexdigest(),
    }
    manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
    artifacts = ((output / PAIR_JSONL, payload), (output / PAIR_MANIFEST, manifest_bytes))
    # Detect a stale artifact before installing either half; exact reruns are idempotent.
    for path, content in artifacts:
        if path.exists() and path.read_bytes() != content:
            raise ValueError(f"frozen training-pair artifact differs: {path}")
    for path, content in artifacts:
        install_bytes_once(content, path, surface="campaign2 training pairs")
    return manifest

def validate_training_pairs(path: Path) -> dict[str, str]:
    """Authenticate bytes, counts and pinned sources; recompute full eval exclusion."""
    payload, evidence = path.read_bytes(), manifest_path(path).read_bytes()
    pairs = decode_pairs(payload)
    manifest = json.loads(evidence, object_pairs_hook=strict_json_object,
                          parse_constant=reject_nonstandard_json_constant)
    if not isinstance(manifest, dict):
        raise ValueError("training-pair manifest must be an object")
    digest = hashlib.sha256(payload).hexdigest()
    if manifest.get("jsonl_sha256") != digest:
        raise ValueError("training-pair manifest hash mismatch")
    if (manifest.get("format") != PAIR_FORMAT or manifest.get("rule") != PAIR_RULE
            or type(manifest.get("seed")) is not int or manifest["seed"] != FIT_SHUFFLE_SEED):
        raise ValueError("unsupported training-pair manifest contract")
    cap = manifest.get("cap")
    validate_cap(cap)
    counts = manifest.get("counts")
    if (not isinstance(counts, dict) or set(counts) != set(COUNT_KEYS)
            or any(type(value) is not int or value < 0 for value in counts.values())):
        raise ValueError("invalid training-pair manifest counts")
    if (type(manifest.get("row_count")) is not int or manifest["row_count"] != len(pairs)
            or counts["selected"] != len(pairs) or len(pairs) != min(cap, counts["paired"])
            or counts["capped"] != counts["paired"] - len(pairs)
            or counts["prompts_seen"] != sum(counts[key] for key in (
                "excluded_eval", "unpaired", "identical_responses", "duplicate_prompts", "paired"))
            or counts["rows_seen"] < counts["prompts_seen"] + counts["paired"]):
        raise ValueError("training-pair manifest row count mismatch")
    evaluation, sources = load_evaluation()
    sources[SafetySourceName.TRAINING] = TRAINING_SPEC.metadata(counts["rows_seen"])
    if manifest.get("sources") != sources:
        raise ValueError("training-pair manifest dataset provenance mismatch")
    forbidden = safety_evaluation_input_keys(evaluation)
    if (type(manifest.get("evaluation_prompt_count")) is not int
            or manifest["evaluation_prompt_count"] != len(forbidden)
            or type(manifest.get("evaluation_overlap")) is not int
            or manifest["evaluation_overlap"] != 0
            or any(normalize_input(pair["prompt"]) in forbidden for pair in pairs)):
        raise ValueError("safety training pairs overlap evaluation or exclusion evidence differs")
    return {"safety_training_pairs": digest,
        "safety_training_pairs_manifest": hashlib.sha256(evidence).hexdigest()}

@hydra.main(version_base=None, config_path="../../configs/eval", config_name="campaign2_training_pairs")
def main(cfg: DictConfig) -> None:
    validate_cap(cfg.cap)
    evaluation, sources = load_evaluation()
    training = load_source(TRAINING_SPEC)
    sources[SafetySourceName.TRAINING] = TRAINING_SPEC.metadata(len(training))
    manifest = write_pairs(Path(cfg.output_dir), training, evaluation, sources, cap=cfg.cap)
    logger.info("Frozen training pairs at {}: {}", cfg.output_dir, manifest)

if __name__ == "__main__":
    main()
