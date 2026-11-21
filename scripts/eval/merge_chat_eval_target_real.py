"""Merge disjoint, identity-matched turn ChatCORE YAMLs with atomic publication."""

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, ListConfig

from scripts.eval.chat_eval import ALL_CHAT_TASKS, BASELINE_ACCURACIES, compute_chatcore_metric
from steering.eval.chat_eval_result_identity import (
    GENERATION_SETTING_KEYS,
    RANGED_GENERATION_SETTING_KEYS,
    ChatEvalIdentityVersion,
    validate_self_conditioned_run_identity,
)
from steering.eval.pooling_op import PoolingOp
from steering.eval.progression_results import SuiteResultKind
from steering.eval.target_conditioning import TargetConditioningMode


class _UniqueKeyLoader(yaml.SafeLoader):
    """Do not silently discard duplicate task or identity keys in input YAML."""

    def construct_mapping(self, node, deep=False):
        mapping = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in mapping:
                raise ValueError(f"duplicate YAML key: {key}")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _require_keys(value, keys: set[str] | frozenset[str], label: str) -> dict:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label} requires exactly these fields: {sorted(keys)}")
    return value


def _validate_run_identity(value) -> None:
    if isinstance(value, dict) and value.get("version") == ChatEvalIdentityVersion.SELF_CONDITIONED:
        validate_self_conditioned_run_identity(value)
        return
    regenerate = " (rerun through the turn entrypoint to regenerate receipts)"
    identity = _require_keys(value, {"version", "checkpoint", "conditioning", "generation", "cache"},
                             "run_identity" + regenerate)
    if type(identity["version"]) is not int:
        raise ValueError("unsupported run_identity version")
    version = ChatEvalIdentityVersion(identity["version"])
    checkpoint = _require_keys(identity["checkpoint"],
        {"metadata_sha256", "model_sha256", "architecture_sha256", "resolved_tag", "step"},
        "checkpoint identity" + regenerate)
    conditioning = _require_keys(identity["conditioning"], {"turn_context_contract_sha256"}, "conditioning identity")
    cache = _require_keys(identity["cache"],
        {"resolved_path", "contexts_fingerprint", "input_provenance", "shards"}, "cache identity")
    if not isinstance(cache["resolved_path"], str) or not Path(cache["resolved_path"]).is_absolute():
        raise ValueError("cache identity requires an absolute resolved_path")
    inputs = _require_keys(cache["input_provenance"],
        {"jsonl_sha256", "parquet_files", "feature_files"}, "cache input provenance")
    digests = [checkpoint["metadata_sha256"], checkpoint["model_sha256"], checkpoint["architecture_sha256"],
               conditioning["turn_context_contract_sha256"], cache["contexts_fingerprint"], inputs["jsonl_sha256"]]
    for shards, name in ((cache["shards"], "file"), (inputs["parquet_files"], "name"), (inputs["feature_files"], "name")):
        if not isinstance(shards, list) or not shards:
            raise ValueError("cache identity requires nonempty ordered shard digests")
        seen = set()
        for shard in shards:
            _require_keys(shard, {name, "sha256"}, "cache shard identity")
            if not isinstance(shard[name], str) or not shard[name] or shard[name] in seen:
                raise ValueError("cache identity requires unique shard names")
            seen.add(shard[name])
            digests.append(shard["sha256"])
    for digest in digests:
        if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ValueError("run_identity requires SHA-256 digests")
    if (type(checkpoint["step"]) is not int or checkpoint["step"] < 0
            or not isinstance(checkpoint["resolved_tag"], str) or not checkpoint["resolved_tag"]):
        raise ValueError("run_identity requires a checkpoint tag and nonnegative step")
    keys = (GENERATION_SETTING_KEYS if version is ChatEvalIdentityVersion.LEGACY
            else RANGED_GENERATION_SETTING_KEYS)
    generation = _require_keys(identity["generation"], keys, "generation settings" + regenerate)
    for key in ("num_samples", "max_new_tokens", "batch_size", "greedy_microbatch_size"):
        if type(generation[key]) is not int or generation[key] <= 0:
            raise ValueError(f"generation settings require a positive integer {key}")
    for key in ("max_problems", "top_k"):
        if generation.get(key) is not None and (type(generation[key]) is not int or generation[key] <= 0):
            raise ValueError(f"generation settings require null or a positive integer {key}")
    if (type(generation["temperature"]) not in (int, float)
            or not math.isfinite(generation["temperature"]) or generation["temperature"] < 0
            or type(generation["greedy_python_tools"]) is not bool):
        raise ValueError("invalid generation settings")
    for key in ("dtype", "compute_dtype", "device_type"):
        if not isinstance(generation[key], str) or not generation[key]:
            raise ValueError(f"generation settings require {key}")


def _read_result(path: Path) -> tuple[dict, dict, dict]:
    with path.open(encoding="utf-8") as handle:
        payload = yaml.load(handle, Loader=_UniqueKeyLoader)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a target-real result mapping")
    kind = SuiteResultKind(payload.get("kind"))
    fields = {"kind", "target_conditioning_mode", "cache_dir", "seeds"}
    if kind is SuiteResultKind.SELF_CONDITIONED:
        fields.remove("target_conditioning_mode")
    elif kind is SuiteResultKind.TARGET_REAL_POOLED:
        fields.add("pooling_op")
        PoolingOp(payload.get("pooling_op"))
    elif kind is not SuiteResultKind.TARGET_REAL_LOCKSTEP:
        raise ValueError("only target-real results can be merged")
    if kind is not SuiteResultKind.SELF_CONDITIONED:
        TargetConditioningMode.parse(payload.get("target_conditioning_mode"))
    _require_keys(payload, fields, "target config")
    if not isinstance(payload["cache_dir"], str) or not payload["cache_dir"]:
        raise ValueError("target config requires cache_dir")
    seeds = _require_keys(payload["seeds"], {"adhoc"}, "seeds")
    body = seeds["adhoc"]
    if not isinstance(body, dict) or not {"results", "run_identity"} <= set(body):
        raise ValueError("result requires results and run_identity; rerun legacy evaluations to record identity")
    if set(body) - {"results", "run_identity", "chatcore_metric", "task_details"}:
        raise ValueError("unsupported ChatCORE result body fields")
    _validate_run_identity(body["run_identity"])
    results = body["results"]
    if not isinstance(results, dict) or not results:
        raise ValueError("results must be a nonempty task table")
    for task, score in results.items():
        if task not in BASELINE_ACCURACIES:
            raise ValueError(f"unknown ChatCORE task: {task}")
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"invalid accuracy for {task}: {score}")
    version = ChatEvalIdentityVersion(body["run_identity"]["version"])
    if (kind is SuiteResultKind.SELF_CONDITIONED) != (version is ChatEvalIdentityVersion.SELF_CONDITIONED):
        raise ValueError("result kind and receipt version disagree")
    if version in (ChatEvalIdentityVersion.PROBLEM_RANGES, ChatEvalIdentityVersion.SELF_CONDITIONED):
        task_keys: set[str] = {str(task) for task in results}
        details = _require_keys(body.get("task_details"), task_keys, "task_details")
        for task, score in results.items():
            row = _require_keys(details[task],
                {"problem_start", "max_problems", "task_size", "num_passed", "total"}, f"{task} range/counts")
            if any(type(value) is not int or value < 0 for value in row.values()):
                raise ValueError(f"{task}: range/counts require nonnegative integers")
            if (row["total"] <= 0 or row["total"] != row["max_problems"]
                    or row["num_passed"] > row["total"]
                    or row["problem_start"] + row["max_problems"] > row["task_size"]):
                raise ValueError(f"{task}: invalid problem range/counts")
            if score != row["num_passed"] / row["total"]:
                raise ValueError(f"{task}: accuracy disagrees with num_passed / total")
    elif "task_details" in body:
        raise ValueError("legacy run_identity cannot contain task_details")
    comparison = {key: value for key, value in payload.items() if key != "seeds"}
    comparison["run_identity"] = body["run_identity"]
    return payload, results, comparison


def merge_target_real_results(
    input_files: Sequence[Path], output_file: Path, *, task_totals: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Merge task jobs or declared problem ranges and publish atomically.

    Config names are file labels; semantic target settings in the YAML must
    match exactly. Old results without a run identity cannot be authenticated.
    Input metrics are ignored and recomputed with the evaluator's arithmetic.
    ``task_totals`` declares the exact tasks and N for each [0, N) union. Without
    it, retain the all-six-tasks, exactly-once contract for whole-task jobs.
    """
    if not input_files:
        raise ValueError("input_files must contain at least one result YAML")
    if any(path.resolve() == output_file.resolve() for path in input_files):
        raise ValueError("output_file must differ from every input file")
    if task_totals is not None and (not isinstance(task_totals, Mapping) or not task_totals
            or any(task not in ALL_CHAT_TASKS or type(count) is not int or count <= 0
                   for task, count in task_totals.items())):
        raise ValueError("task_totals must declare known tasks with positive integer totals")
    merged = None
    expected = None
    results = {}
    ranges: dict[str, list[dict[str, int]]] = {}
    for path in input_files:
        payload, task_results, comparison = _read_result(path)
        # JSON preserves equality-relevant scalar types (e.g. 1116 vs 1116.0)
        # that Python mapping equality otherwise treats as interchangeable.
        canonical = json.dumps(comparison, sort_keys=True, allow_nan=False)
        if expected is None:
            expected = canonical
            merged = deepcopy(payload)
        elif canonical != expected:
            raise ValueError(f"{path}: checkpoint identity / target config / conditioning / generation settings disagree")
        duplicates = set(results) & set(task_results)
        details = payload["seeds"]["adhoc"].get("task_details")
        if task_totals is not None and details is None:
            raise ValueError("problem shard merging requires range/counts receipts; rerun legacy evaluations")
        if duplicates and task_totals is None:
            raise ValueError(f"duplicate tasks: {sorted(duplicates)}")
        if details is not None:
            for task, row in details.items():
                ranges.setdefault(task, []).append(row)
        results.update(task_results)
    expected_tasks = set(ALL_CHAT_TASKS if task_totals is None else task_totals)
    missing = expected_tasks - set(results)
    if missing:
        raise ValueError(f"cannot compute full ChatCORE; missing tasks: {sorted(missing)}")
    if set(results) - expected_tasks:
        raise ValueError("input tasks disagree with declared task_totals")
    merged_details = {}
    for task in ALL_CHAT_TASKS:
        if task not in ranges:
            continue
        rows = sorted(ranges[task], key=lambda row: row["problem_start"])
        cursor = 0
        passed = 0
        for row in rows:
            if row["task_size"] != rows[0]["task_size"]:
                raise ValueError(f"{task}: task sizes disagree")
            if row["problem_start"] != cursor:
                raise ValueError(f"{task}: overlapping or missing problem ranges at {cursor}")
            cursor += row["total"]
            passed += row["num_passed"]
        if task_totals is not None and cursor != task_totals[task]:
            raise ValueError(f"{task}: ranges cover [0, {cursor}), expected [0, {task_totals[task]})")
        merged_details[task] = {
            "problem_start": 0, "max_problems": cursor, "task_size": rows[0]["task_size"],
            "num_passed": passed, "total": cursor,
        }
        results[task] = passed / cursor
    results = {task: results[task] for task in ALL_CHAT_TASKS if task in results}
    metric = compute_chatcore_metric(results)
    if merged is None:
        raise ValueError("cannot publish ChatCORE without a validated input result")
    merged["seeds"]["adhoc"]["results"] = results
    if merged_details:
        merged["seeds"]["adhoc"]["task_details"] = merged_details
    merged["seeds"]["adhoc"].pop("chatcore_metric", None)
    if metric is not None:
        merged["seeds"]["adhoc"]["chatcore_metric"] = metric
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_file.parent,
                prefix=f".{output_file.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            yaml.safe_dump(merged, handle, default_flow_style=False, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_file)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return merged


@hydra.main(config_path="../../configs/eval", config_name="merge_chat_eval_target_real", version_base=None)
def main(cfg: DictConfig) -> None:
    if not isinstance(cfg.input_files, (list, ListConfig)):
        raise ValueError("input_files must be a list of result YAML paths")
    merged = merge_target_real_results([Path(path) for path in cfg.input_files], Path(cfg.output_file),
                                      task_totals=cfg.get("task_totals"))
    logger.info("Merged {} tasks into {}", len(merged["seeds"]["adhoc"]["results"]), cfg.output_file)


if __name__ == "__main__":
    main()
