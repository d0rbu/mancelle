"""Run the complete AxBench-Adv GEPA pipeline from a strict Hydra config."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from scripts.eval.axbench_adv.pressures import DEFAULT_MAGNITUDES, Pressure
from steering._common.strict_json import string_keyed_mapping
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import AxBenchAdvGcgConfig, parse_gcg_config
from steering.eval.axbench_adv.pipeline import AxBenchAdvRunConfig, run_pipeline
from steering.eval.axbench_adv.soft_prompt_config import (
    AxBenchAdvSoftPromptConfig,
    parse_soft_prompt_config,
)
from steering.eval.axbench_adv.source_contract import (
    NLAttachmentExpectations,
    NLAttachmentPaths,
    SourceExpectations,
    SourcePaths,
)
from steering.eval.axbench_base.config import LLMClientConfig


def _mapping(value: object, *, field: str, keys: set[str]) -> dict[str, Any]:
    parsed = string_keyed_mapping(value)
    if parsed is None or set(parsed) != keys:
        observed = sorted(parsed) if parsed is not None else type(value).__name__
        raise AxBenchAdvError(
            f"config {field} fields must be exactly {sorted(keys)}, got {observed}"
        )
    return parsed


def _text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "???":
        raise AxBenchAdvError(f"config {field} must be a nonblank resolved string")
    return value


def _path(value: object, *, field: str) -> Path:
    return Path(_text(value, field=field)).expanduser()


def build_config(cfg: DictConfig) -> AxBenchAdvRunConfig:
    """Resolve Hydra once, reject unknown fields, and construct typed boundaries."""

    raw = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(raw, dict):
        raw.setdefault("pressures", [Pressure.GEPA])
        raw.setdefault("counter_magnitudes", list(DEFAULT_MAGNITUDES))
        raw.setdefault("project_counter_direction_root", None)
        raw.setdefault("project_counter_layer", None)
        raw.setdefault("project_counter_fit_seed", 42)
        raw.setdefault("counter_resume_snapshot", None)
        raw.setdefault("counter_resume_sha256", None)
        raw.setdefault("project_judge_inflight", 1)
        raw.setdefault("project_judge_workers", 1)
        raw.setdefault("project_judge_pair_concurrency", 1)
        raw.setdefault("project_reuse_judge_workers", False)
        raw.setdefault("project_reuse_single_token_mask", False)
        raw.setdefault("project_bf16_decoder_linears", False)
        raw.setdefault("project_reuse_diffmean_texts", False)
        raw.setdefault("project_weight_free_extraction", False)
        raw.setdefault("project_native_extraction", False)
        raw.setdefault("project_fit_selected_concepts_only", False)
        raw.setdefault("reuse_final_judge_worker", False)
        raw.setdefault("gcg", asdict(AxBenchAdvGcgConfig()))
        raw.setdefault("soft_prompt", asdict(AxBenchAdvSoftPromptConfig()))
        source_raw = raw.get("source")
        if isinstance(source_raw, dict):
            source_raw.setdefault("nl_attachment", None)
            source_raw.setdefault("target_receipt", None)
    top = _mapping(
        raw,
        field="root",
        keys={
            "pressures",
            "counter_magnitudes",
            "project_counter_direction_root",
            "project_counter_layer",
            "project_counter_fit_seed",
            "project_judge_inflight",
            "project_judge_workers",
            "project_judge_pair_concurrency",
            "project_reuse_judge_workers",
            "project_reuse_single_token_mask",
            "project_bf16_decoder_linears",
            "project_reuse_diffmean_texts",
            "project_weight_free_extraction",
            "project_native_extraction",
            "project_fit_selected_concepts_only",
            "reuse_final_judge_worker",
            "gcg",
            "soft_prompt",
            "counter_resume_snapshot",
            "counter_resume_sha256",
            "api_base_url",
            "cache_only",
            "device",
            "factor_overrides",
            "gepa_seed",
            "judge_cache",
            "master_seed",
            "output_dir",
            "reflection",
            "reflection_cache",
            "reflection_max_retries",
            "reflection_minibatch_size",
            "seed_suffix",
            "source",
            "worker_max_prompt_tokens",
            "worker_timeout_seconds",
        },
    )
    source = _mapping(
        top["source"],
        field="source",
        keys={
            "aux_parquet",
            "evaluated_jsonl",
            "expectations",
            "hashes",
            "main_parquet",
            "prepared_dir",
            "project_parquet",
            "target_meta",
            "target_receipt",
            "feature_statistics",
            "nl_attachment",
        },
    )
    expectations_raw = source["expectations"]
    if isinstance(expectations_raw, Mapping):
        expectations_raw = dict(expectations_raw)
        expectations_raw.setdefault("include_nl_steering", False)
    expectations = _mapping(
        expectations_raw,
        field="source.expectations",
        keys={
            "include_nl_steering",
            "config_name_prefix",
            "dataset_revision",
            "layer",
            "max_concepts",
            "model_revision",
            "output_length",
            "package_lock_sha256",
            "project_package_lock_sha256",
            "project_repository_git_commit",
            "project_target_step",
            "project_target_model_tag",
            "repository_git_commit",
            "require_clean_worktree",
            "seed",
            "target_model",
            "temperature",
            "vendored_axbench_upstream_commit",
        },
    )
    hashes = _mapping(
        source["hashes"],
        field="source.hashes",
        keys={
            "aux_parquet",
            "evaluated_jsonl",
            "main_parquet",
            "project_manifest",
            "project_parquet",
            "target_receipt",
        },
    )
    reflection = _mapping(
        top["reflection"],
        field="reflection",
        keys={
            "max_concurrent",
            "model",
            "provider",
            "requests_per_minute",
            "transport_retry_budget_seconds",
        },
    )
    overrides = top["factor_overrides"]
    if not isinstance(overrides, Mapping):
        raise AxBenchAdvError("config factor_overrides must be a mapping")
    nl_attachment_raw = source["nl_attachment"]
    nl_paths = None
    nl_expectations = None
    if nl_attachment_raw is not None:
        attachment = _mapping(
            nl_attachment_raw,
            field="source.nl_attachment",
            keys={"paths", "expectations"},
        )
        attachment_paths = _mapping(
            attachment["paths"],
            field="source.nl_attachment.paths",
            keys={
                "project_parquet",
                "project_manifest",
                "evaluated_jsonl",
            },
        )
        attachment_expectations = _mapping(
            attachment["expectations"],
            field="source.nl_attachment.expectations",
            keys={"max_concepts"},
        )
        nl_paths = NLAttachmentPaths(
            **{
                key: _path(value, field=f"source.nl_attachment.paths.{key}")
                for key, value in attachment_paths.items()
            }
        )
        nl_expectations = NLAttachmentExpectations(**attachment_expectations)
    return AxBenchAdvRunConfig(
        source_paths=SourcePaths(
            prepared_dir=_path(source["prepared_dir"], field="source.prepared_dir"),
            main_parquet=_path(source["main_parquet"], field="source.main_parquet"),
            aux_parquet=_path(source["aux_parquet"], field="source.aux_parquet"),
            evaluated_jsonl=_path(source["evaluated_jsonl"], field="source.evaluated_jsonl"),
            project_parquet=_path(source["project_parquet"], field="source.project_parquet"),
            target_meta=_path(source["target_meta"], field="source.target_meta"),
            feature_statistics=_path(
                source["feature_statistics"], field="source.feature_statistics"
            ),
            target_receipt_path=(
                None
                if source["target_receipt"] is None
                else _path(source["target_receipt"], field="source.target_receipt")
            ),
        ),
        source_expectations=SourceExpectations(**expectations),
        source_hashes={
            key: _text(value, field=f"source.hashes.{key}") for key, value in hashes.items()
        },
        output_dir=_path(top["output_dir"], field="output_dir"),
        judge_cache=_path(top["judge_cache"], field="judge_cache"),
        reflection_cache=_path(top["reflection_cache"], field="reflection_cache"),
        reflection=LLMClientConfig(**reflection),
        master_seed=top["master_seed"],
        gepa_seed=top["gepa_seed"],
        device=top["device"],
        api_base_url=top["api_base_url"],
        worker_timeout_seconds=top["worker_timeout_seconds"],
        worker_max_prompt_tokens=top["worker_max_prompt_tokens"],
        project_judge_inflight=top["project_judge_inflight"],
        project_judge_workers=top["project_judge_workers"],
        project_judge_pair_concurrency=top["project_judge_pair_concurrency"],
        project_reuse_judge_workers=top["project_reuse_judge_workers"],
        project_reuse_single_token_mask=top["project_reuse_single_token_mask"],
        project_bf16_decoder_linears=top["project_bf16_decoder_linears"],
        project_reuse_diffmean_texts=top["project_reuse_diffmean_texts"],
        project_weight_free_extraction=top["project_weight_free_extraction"],
        project_native_extraction=top["project_native_extraction"],
        project_fit_selected_concepts_only=top["project_fit_selected_concepts_only"],
        reuse_final_judge_worker=top["reuse_final_judge_worker"],
        reflection_max_retries=top["reflection_max_retries"],
        reflection_minibatch_size=top["reflection_minibatch_size"],
        seed_suffix=top["seed_suffix"],
        factor_overrides=dict(overrides),
        cache_only=top["cache_only"],
        pressures=top["pressures"],
        counter_magnitudes=top["counter_magnitudes"],
        project_counter_direction_root=None
        if top["project_counter_direction_root"] is None
        else _path(top["project_counter_direction_root"], field="project_counter_direction_root"),
        project_counter_layer=top["project_counter_layer"],
        project_counter_fit_seed=top["project_counter_fit_seed"],
        counter_resume_snapshot=None
        if top["counter_resume_snapshot"] is None
        else _path(top["counter_resume_snapshot"], field="counter_resume_snapshot"),
        counter_resume_sha256=top["counter_resume_sha256"],
        gcg=parse_gcg_config(top["gcg"]),
        soft_prompt=parse_soft_prompt_config(top["soft_prompt"]),
        nl_attachment_paths=nl_paths,
        nl_attachment_expectations=nl_expectations,
    )


@hydra.main(
    config_path="../../../configs/eval/axbench_adv",
    config_name="axbench_adv",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    manifest = run_pipeline(build_config(cfg))
    logger.info("AxBench-Adv complete: {}", manifest)


if __name__ == "__main__":
    main()
