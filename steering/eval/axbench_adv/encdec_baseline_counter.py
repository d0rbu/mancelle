"""Counter-only latent Adv against four positive encoder-decoder baselines."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
from loguru import logger
from omegaconf import DictConfig

from nanochat.atomic_io import atomic_write_parquet
from nanochat.tokenizer import get_tokenizer
from scripts.eval.axbench_adv.pressures import Pressure, magnitude_label, validate_magnitudes
from scripts.eval.axbench_upstream.merge_methods import METHOD_FACTORS
from steering.conditional.encoder_decoder.eval_residual import post_cross_attention_residual_counter
from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.best_point import per_concept_ratings
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    mean_counter_fitness,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.encdec_sources import (
    METHOD_FACTORS_ENCDEC,
    EncDecSources,
    _sha,
    validate_encdec_sources,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.execution_scope import validate_concept_ids
from steering.eval.axbench_adv.final_judge import (
    _read_call_accounting,
    build_point_frame,
    judge_generated_point,
    validate_final_judge_outputs,
)
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.native_mailbox import (
    canonical_json,
    checked_raw,
    digest_json,
    read_json,
)
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient
from steering.eval.axbench_base.aggregation import hmean
from steering.eval.axbench_upstream.encdec_baselines.lora import validate_lora_state
from steering.eval.axbench_upstream.encdec_baselines.run import (
    _generate_encdec_cell,
    _prepare_all,
    _reset_compiler_state_before_generation,
    _steered_prompt_frame,
    _validate_all,
)
from steering.eval.axbench_upstream.methods import load_extraction_stack, load_target_model
from steering.eval.axbench_upstream.prompt_prep import build_rendered_prompt_plan
from steering.eval.axbench_upstream.run_methods import (
    _git_commit,
    _repository_source_sha256,
    _runtime_identity,
)
from steering.eval.axbench_upstream.work_grid import CellKey, build_encdec_cell_grid
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.immutable_io import install_bytes_once

FORMAT = "axbench-adv-encdec-baseline-counter-v1"
PRESSURES = (Pressure.DIFFMEAN_COUNTER, Pressure.LINEAR_PROBE_COUNTER)
FAMILY = {Pressure.DIFFMEAN_COUNTER: "diffmean", Pressure.LINEAR_PROBE_COUNTER: "linear_probe"}
CellAddress = tuple[str, int, int, str, float]
Emissions = dict[tuple[str, float], dict[tuple[str, int, int], RawEvaluation]]
POINT_FILES = (
    "input/steering_data.parquet",
    "judged/steering.jsonl",
    "judged/steering_data.parquet",
)


def _concepts(raw: Any) -> tuple[int, ...]:
    values = tuple(range(50)) if raw is None else tuple(raw)
    return validate_concept_ids(values, max_concepts=50)


def _grid(raw: Any) -> tuple[float, ...]:
    values = validate_magnitudes(tuple(raw))
    return tuple(float(value) for value in values)


def _counter_sources(cfg: DictConfig, sources: EncDecSources) -> Any:
    if type(cfg.counter_layer) is not int or cfg.counter_layer != 9:
        raise AxBenchAdvError("encoder-decoder baseline campaign counter layer must be 9")
    if type(cfg.counter_fit_seed) is not int or cfg.counter_fit_seed != 42:
        raise AxBenchAdvError("encoder-decoder baseline campaign counter fit seed must be 42")
    concepts = {
        concept_id: str(
            sources.source_main.loc[
                sources.source_main["concept_id"].eq(concept_id), "input_concept"
            ].iloc[0]
        )
        for concept_id in range(50)
    }
    model = sources.identity["inputs"]["target_model"]
    width = int(sources.setup.meta["model_config"]["n_embd"])
    return preflight_project_direction_fit(
        Path(str(cfg.counter_root)),
        ProjectCounterExpectations(
            target_model_sha256=model["sha256"],
            target_metadata_sha256=_sha(Path(str(cfg.target_meta_path))),
            concept_texts=concepts,
            decoder_width=width,
            layer=int(cfg.counter_layer),
            fit_seed=int(cfg.counter_fit_seed),
        ),
    )


def _runtime(device: str) -> dict[str, Any]:
    if device.startswith("cuda") and not t.cuda.is_available():
        raise AxBenchAdvError("CUDA requested but unavailable")
    result = dict(_runtime_identity(device))
    if result.get("gpu_name") == "unavailable-test-double":
        raise AxBenchAdvError("CUDA runtime identity is unavailable")
    return result


def _identity(
    *,
    cfg: DictConfig,
    sources: EncDecSources,
    counter: Any,
    concepts: Sequence[int],
    magnitudes: Sequence[float],
    phase: str,
    runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "phase": phase,
        "source": dict(sources.source_digest),
        "target_model": sources.identity["inputs"]["target_model"],
        "counter_manifest_sha256": counter.manifest_sha256,
        "counter_artifacts": {
            f"{family}/c{concept_id}": artifact.sha256
            for (family, concept_id), artifact in sorted(counter.artifacts.items())
        },
        "methods": list(ENCDEC_BASELINE_METHODS),
        "best_factors": dict(sources.best_factors),
        "concept_ids": list(concepts),
        "magnitudes": list(magnitudes),
        "generation": {"seed": 42, "max_new_tokens": 128, "temperature": 1.0, "top_k": 50},
        "counter_site": "post_cross_attention_residual",
        "counter_layer": int(cfg.counter_layer),
        "counter_fit_seed": int(cfg.counter_fit_seed),
        "zero_control_source": "runtime",
        "runtime": dict(runtime) if runtime is not None else _runtime(str(cfg.device)),
        "implementation": {
            "git_commit": _git_commit(),
            "source_sha256": _repository_source_sha256(),
        },
    }


def _install_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json(value)
    install_bytes_once((payload + "\n").encode(), path, surface="encoder-decoder Adv artifact")


def _install_identity(output: Path, identity: Mapping[str, Any]) -> None:
    if output.exists() and not (output / "identity.json").is_file() and any(
        path.is_symlink() or not path.is_file() or not path.name.startswith(".identity.json.tmp.")
        for path in output.iterdir()
    ):
        raise AxBenchAdvError("encoder-decoder Adv output exists without a run identity")
    _install_json(output / "identity.json", identity)


def _cells(
    sources: EncDecSources, concepts: Sequence[int], *, pilot: bool
) -> tuple[pd.DataFrame, tuple[CellKey, ...]]:
    # Prompt feature sampling consumes shared RNGs in producer order: prepare
    # all original and then all steered prompts, even for a concept shard.
    frame = sources.source_main.loc[
        sources.source_main["factor"].isin(sources.best_factors.values())
    ]
    grid = build_encdec_cell_grid(frame, seed=42)
    selected = tuple(
        cell
        for cell in grid
        if cell.concept_id in concepts
        and float.fromhex(cell.factor_hex) == sources.best_factors[cell.method]
        and (not pilot or cell.input_id == 0)
    )
    expected_count = len(concepts) * (1 if pilot else 10) * len(ENCDEC_BASELINE_METHODS)
    if len(selected) != expected_count:
        raise AxBenchAdvError(
            "selected positive cells do not cover the exact concept/input product"
        )
    return frame, selected


def _record_path(
    output: Path,
    method: str,
    concept_id: int,
    input_id: int,
    pressure: str,
    magnitude: float,
    *,
    tree: str = "generations",
) -> Path:
    return (
        output
        / tree
        / method
        / f"c{concept_id:02d}"
        / f"i{input_id:02d}"
        / pressure
        / f"{magnitude_label(magnitude)}.json"
    )


def _record(
    *,
    output: Path,
    identity_sha: str,
    cell: CellKey,
    pressure: str,
    magnitude: float,
    generate: Callable[[], str] | None = None,
) -> str:
    path = _record_path(
        output,
        method=cell.method,
        concept_id=cell.concept_id,
        input_id=cell.input_id,
        pressure=pressure,
        magnitude=magnitude,
    )
    key = {
        "identity_sha256": identity_sha,
        "cell": cell.to_dict(),
        "pressure": pressure,
        "magnitude_hex": float(magnitude).hex(),
    }
    if path.exists():
        record = read_json(path)
        if (
            set(record) != {"key", "generation", "generation_sha256"}
            or record["key"] != key
            or not isinstance(record["generation"], str)
            or record["generation_sha256"]
            != hashlib.sha256(record["generation"].encode()).hexdigest()
        ):
            raise AxBenchAdvError(f"saved generation record differs: {path}")
        return record["generation"]
    if generate is None:
        raise AxBenchAdvError(f"missing generation record: {path}")
    generation = generate()
    if not isinstance(generation, str):
        raise AxBenchAdvError("encoder-decoder generation must be text")
    _install_json(
        path,
        {
            "key": key,
            "generation": generation,
            "generation_sha256": hashlib.sha256(generation.encode()).hexdigest(),
        },
    )
    return generation


def generate(
    cfg: DictConfig,
    sources: EncDecSources,
    counter: Any,
    output: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    concepts = tuple(identity["concept_ids"])
    magnitudes = tuple(identity["magnitudes"])
    frame, cells = _cells(sources, concepts, pilot=identity["phase"] == "pilot")
    tokenizer = get_tokenizer()
    prompt_plan = build_rendered_prompt_plan(
        frame,
        tokenizer,
        declared_factors=METHOD_FACTORS["EncDecDiffMean"],
        render=render_prompt_for_chat_completion,
    )
    steered_frame, _ = _steered_prompt_frame(frame, sources.steering_prompts)
    steered_plan = build_rendered_prompt_plan(
        steered_frame, tokenizer, declared_factors=(1.0,), render=render_prompt_for_chat_completion
    )
    extraction_model, sae = load_extraction_stack(sources.setup, str(cfg.device))
    try:
        prepared = _prepare_all(sources.setup, extraction_model, sae, tokenizer, prompt_plan)
        steered_prepared = _prepare_all(
            sources.setup, extraction_model, sae, tokenizer, steered_plan
        )
    finally:
        del extraction_model, sae
        gc.collect()
        if str(cfg.device).startswith("cuda"):
            t.cuda.empty_cache()
    model, target_tokenizer = load_target_model(sources.setup, str(cfg.device))
    _validate_all(sources.setup, model, target_tokenizer, prompt_plan, prepared, max_new_tokens=128)
    _validate_all(
        sources.setup, model, target_tokenizer, steered_plan, steered_prepared, max_new_tokens=128
    )
    lora_layers = tuple(sources.manifest["layer_resolution"]["resolved_lora_layers"])
    positive_layer = int(sources.manifest["layer_resolution"]["resolved_intervention_layer"])
    for concept_id in concepts:
        validate_lora_state(model, sources.lora_states[concept_id], layers=lora_layers)
    _reset_compiler_state_before_generation()
    digest = digest_json(identity)
    for completed, cell in enumerate(cells, 1):

        def produce(cell: CellKey = cell) -> str:
            return _generate_encdec_cell(
                cell,
                rows=prompt_plan.rows,
                prompt_plan=prompt_plan,
                steered_plan=steered_plan,
                prepared_by_tokens=prepared,
                steered_prepared_by_tokens=steered_prepared,
                fitted=sources.fitted,
                lora_states=sources.lora_states,
                intervention_layer=positive_layer,
                lora_layers=lora_layers,
                setup=sources.setup,
                model=model,
                tokenizer=target_tokenizer,
                max_new_tokens=128,
            )

        control = _record(
            output=output,
            identity_sha=digest,
            cell=cell,
            pressure="shared_zero",
            magnitude=0.0,
            generate=produce,
        )
        if sources.identity.get("runtime") == identity["runtime"]:
            row = prompt_plan.rows[cell.source_row_position]
            if getattr(row, f"{cell.method}_steered_generation") != control:
                raise AxBenchAdvError(f"same-runtime archived positive replay differs at {cell}")
        replay_direction = counter.artifacts[("diffmean", cell.concept_id)].loaded.direction
        with post_cross_attention_residual_counter(
            model, layer=int(cfg.counter_layer), direction=replay_direction, magnitude=0.0
        ):
            replay = produce()
        if replay != control:
            raise AxBenchAdvError(f"positive-only zero replay differs at {cell}")
        for pressure in PRESSURES:
            artifact = counter.artifacts[(FAMILY[pressure], cell.concept_id)]
            for magnitude in magnitudes:
                with post_cross_attention_residual_counter(
                    model,
                    layer=int(cfg.counter_layer),
                    direction=artifact.loaded.direction,
                    magnitude=magnitude,
                ):
                    _record(
                        output=output,
                        identity_sha=digest,
                        cell=cell,
                        pressure=str(pressure),
                        magnitude=magnitude,
                        generate=produce,
                    )
        logger.info("Generated {}/{} positive cells and their counter grid", completed, len(cells))
    return {
        "phase": "generation_complete",
        "positive_cells": len(cells),
        "counter_cells": len(cells) * len(PRESSURES) * len(magnitudes),
        "zero_control_source": "runtime",
    }


def _files(root: Path) -> set[Path]:
    if not root.is_dir():
        raise AxBenchAdvError(f"missing artifact tree: {root}")
    if root.is_symlink():
        raise AxBenchAdvError(f"artifact tree contains an unsupported entry: {root}")
    files = set()
    for path in root.rglob("*"):
        if path.is_symlink() or (not path.is_file() and not path.is_dir()):
            raise AxBenchAdvError(f"artifact tree contains an unsupported entry: {path}")
        # A killed atomic writer can leave scratch bytes, never a committed cell.
        if path.is_file() and not (path.name.startswith(".") and ".json.tmp." in path.name):
            files.add(path)
    return files


def _judgment_requests(
    output: Path, identity: Mapping[str, Any], sources: EncDecSources
) -> list[tuple[CellAddress, Path, dict[str, Any], str]]:
    _, cells = _cells(sources, identity["concept_ids"], pilot=False)
    digest = digest_json(identity)
    paths = set()
    requests = []
    for cell in cells:
        for pressure, magnitude in _point_coordinates(identity):
            coordinate = (cell.method, cell.concept_id, cell.input_id, pressure, magnitude)
            generation = _record(
                output=output,
                identity_sha=digest,
                cell=cell,
                pressure=pressure,
                magnitude=magnitude,
            )
            paths.add(_record_path(output, *coordinate))
            key = {
                "identity_sha256": digest,
                "generation_sha256": hashlib.sha256(generation.encode()).hexdigest(),
                "method": cell.method,
                "concept_id": cell.concept_id,
                "input_id": cell.input_id,
                "pressure": pressure,
                "magnitude_hex": float(magnitude).hex(),
            }
            requests.append(
                (coordinate, _record_path(output, *coordinate, tree="judgments"), key, generation)
            )
    if _files(output / "generations") != paths:
        raise AxBenchAdvError("generation record population has missing or extra files")
    return sorted(requests, key=lambda request: request[0])


def _read_judgment(path: Path, key: Mapping[str, Any], generation: str) -> RawEvaluation:
    document = read_json(path)
    if set(document) != {"key", "raw"} or document["key"] != key:
        raise AxBenchAdvError(f"judgment identity differs: {path}")
    return checked_raw(document["raw"], generation)


def _judge_init(
    cfg: DictConfig, sources: EncDecSources, method: str, concept_id: int
) -> dict[str, Any]:
    return {
        "runtime_kind": "judge",
        "method": method,
        "concept_id": concept_id,
        "concept": str(
            sources.source_main.loc[
                sources.source_main["concept_id"].eq(concept_id), "input_concept"
            ].iloc[0]
        ),
        "factor": sources.best_factors[method],
        "judge_model": "openai/gpt-5.6-luna",
        "judge_cache": str(Path(str(cfg.judge_cache)).resolve()),
        "cache_only": cfg.cache_only,
        "api_base_url": str(cfg.api_base_url),
    }


def _judge_cache_only(cfg: DictConfig) -> bool:
    cache_only = cfg.cache_only
    allow_live = cfg.get("allow_live_provider", False)
    if type(cache_only) is not bool or type(allow_live) is not bool:
        raise AxBenchAdvError("cache_only and allow_live_provider must be Boolean")
    if not cache_only and not allow_live:
        raise AxBenchAdvError("live provider judging requires allow_live_provider=true")
    return cache_only


def judge(
    cfg: DictConfig,
    sources: EncDecSources,
    output: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    requests = _judgment_requests(output, identity, sources)
    _judge_cache_only(cfg)
    client: AxBenchAdvJudgeClient | None = None
    bound: tuple[str, int] | None = None
    completed = 0
    try:
        for (method, cid, iid, _, _), path, key, generation in requests:
            if path.exists():
                _read_judgment(path, key, generation)
            else:
                init = _judge_init(cfg, sources, method, cid)
                if client is None:
                    client = AxBenchAdvJudgeClient(init_payload=init, timeout_seconds=600.0)
                elif bound != (method, cid):
                    client.rebind(init)
                bound = (method, cid)
                raw = client.evaluate(generation=generation, input_id=iid)
                _install_json(path, {"key": key, "raw": asdict(raw)})
            completed += 1
            if completed % 100 == 0:
                logger.info("Judged {}/{} generations", completed, len(requests))
    finally:
        if client is not None:
            client.close()
    return {"phase": "inner_judging_complete", "completed": completed, "expected": len(requests)}


def _emissions(output: Path, identity: Mapping[str, Any], sources: EncDecSources) -> Emissions:
    grouped: Emissions = {}
    paths = set()
    for (method, cid, iid, pressure, magnitude), path, key, generation in _judgment_requests(
        output, identity, sources
    ):
        grouped.setdefault((pressure, magnitude), {})[(method, cid, iid)] = _read_judgment(
            path, key, generation
        )
        paths.add(path)
    if _files(output / "judgments") != paths:
        raise AxBenchAdvError("judgment record population has missing or extra files")
    return grouped


def _point_path(output: Path, pressure: str, magnitude: float) -> Path:
    return output / "points" / pressure / magnitude_label(magnitude)


def _point_coordinates(identity: Mapping[str, Any]) -> tuple[tuple[str, float], ...]:
    return (
        ("shared_zero", 0.0),
        *(
            (str(pressure), float(magnitude))
            for pressure in PRESSURES
            for magnitude in identity["magnitudes"]
        ),
    )


def _point_scores(
    judged_jsonl: Path,
    *,
    emissions: Mapping[tuple[str, int, int], RawEvaluation],
    factors: Mapping[str, float],
    concepts: Sequence[int],
    pressure: str,
    magnitude: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    concept_rows = []
    for method in ENCDEC_BASELINE_METHODS:
        ratings = per_concept_ratings(
            judged_jsonl,
            method,
            factors[method],
            expected_concept_ids=concepts,
            rating_fields=(
                "relevance_concept_ratings",
                "fluency_ratings",
                "relevance_instruction_ratings",
            ),
            method_factors=METHOD_FACTORS_ENCDEC,
        )
        means = [sum(values) / len(values) for values in ratings]
        sems = [
            0.0 if len(values) == 1 else statistics.stdev(values) / math.sqrt(len(values))
            for values in ratings
        ]
        for index, concept_id in enumerate(concepts):
            concept_score, fluency_score, instruction_score = (values[index] for values in ratings)
            values = (concept_score, fluency_score, instruction_score)
            concept_rows.append(
                {
                    "method": method,
                    "pressure": pressure,
                    "magnitude": magnitude,
                    "factor": factors[method],
                    "concept_id": concept_id,
                    "concept_score": concept_score,
                    "fluency_score": fluency_score,
                    "instruction_score": instruction_score,
                    "harmonic_quality_2": hmean(concept_score, fluency_score),
                    "harmonic_quality_3": 0.0
                    if min(values) == 0.0
                    else 3.0 / sum(1.0 / value for value in values),
                }
            )
        raw = [emissions[(method, cid, iid)] for cid in concepts for iid in range(10)]
        rows.append(
            {
                "method": method,
                "pressure": pressure,
                "magnitude": magnitude,
                "factor": factors[method],
                "n_concepts": len(concepts),
                "n_inputs": len(raw),
                "concept_score": means[0],
                "fluency_score": means[1],
                "instruction_score": means[2],
                "concept_score_sem": sems[0],
                "fluency_score_sem": sems[1],
                "instruction_score_sem": sems[2],
                "harmonic_quality_2": hmean(means[0], means[1]),
                "harmonic_quality_3": 0.0
                if min(means) == 0.0
                else 3.0 / sum(1.0 / value for value in means),
                "mean_fitness": mean_counter_fitness(raw),
            }
        )
    return rows, concept_rows


def _judge_manifests(output: Path, identity: Mapping[str, Any]) -> dict[str, str]:
    """Authenticate the evaluator receipts required for each completed shard."""
    files = {}
    for pressure, magnitude in _point_coordinates(identity):
        point = _point_path(output, pressure, magnitude)
        judged = point / "judged"
        manifest = read_json(judged / "manifest.json")
        calls, misses = _read_call_accounting(
            judged / "judge_call_accounting.json",
            expected_calls=len(identity["best_factors"]) * len(identity["concept_ids"]) * 10 * 3,
        )
        if (
            manifest.get("schema_version") != 1
            or manifest.get("best_factors") != identity["best_factors"]
            or manifest.get("input_logical_sha256")
            != logical_table_digest(pq.read_table(point / "input/steering_data.parquet"))
            or canonical_json(manifest.get("call_accounting")) != canonical_json(
                {"actual_evaluator_calls": calls, "paid_judge_cache_misses": misses}
            )
            or manifest.get("files")
            != {
                "judge_call_accounting.json": {"sha256": _sha(judged / "judge_call_accounting.json")},
                "steering.jsonl": {"sha256": _sha(judged / "steering.jsonl")},
                "steering_data.parquet": {
                    "logical_sha256": logical_table_digest(
                        pq.read_table(judged / "steering_data.parquet")
                    )
                },
            }
        ):
            raise AxBenchAdvError(f"encoder-decoder Adv final judge manifest differs: {point}")
        path = judged / "manifest.json"
        files[path.relative_to(output).as_posix()] = _sha(path)
    return files


def _tree_digest(root: Path) -> str:
    return digest_json(
        [(path.relative_to(root).as_posix(), _sha(path)) for path in sorted(_files(root))]
    )


def _point_results(
    output: Path, identity: Mapping[str, Any], sources: EncDecSources, grouped: Emissions
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Rebuild derived scores and verify the exact judged inputs once for every consumer."""
    _files(output / "points")
    rows, per_concept, files = [], [], {}
    for pressure, magnitude in _point_coordinates(identity):
        point = _point_path(output, pressure, magnitude)
        emissions = grouped[(pressure, magnitude)]
        table = build_point_frame(
            source_frame=sources.source_main,
            best_factors=identity["best_factors"],
            emissions=emissions,
            max_concepts=50,
            concept_ids=identity["concept_ids"],
            method_factors=METHOD_FACTORS_ENCDEC,
        )
        if logical_table_digest(
            pq.read_table(point / "input/steering_data.parquet")
        ) != logical_table_digest(table):
            raise AxBenchAdvError(f"encoder-decoder Adv judged point input differs: {point}")
        validate_final_judge_outputs(
            input_table=table,
            output_dir=point / "judged",
            emissions=emissions,
            best_factors=identity["best_factors"],
            max_concepts=50,
            concept_ids=identity["concept_ids"],
            method_factors=METHOD_FACTORS_ENCDEC,
        )
        aggregate_rows, concept_rows = _point_scores(
            point / "judged/steering.jsonl",
            emissions=emissions,
            factors=identity["best_factors"],
            concepts=identity["concept_ids"],
            pressure=pressure,
            magnitude=magnitude,
        )
        rows.extend(aggregate_rows)
        per_concept.extend(concept_rows)
        files.update(
            {
                (point / name).relative_to(output).as_posix(): _sha(point / name)
                for name in POINT_FILES
            }
        )
    return rows, per_concept, files


def _manifest(
    output: Path, identity: Mapping[str, Any], points: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "format": FORMAT,
        "status": "complete",
        "zero_control_source": "runtime",
        "identity_sha256": digest_json(identity),
        "concept_ids": list(identity["concept_ids"]),
        "methods": list(ENCDEC_BASELINE_METHODS),
        "pressures": [str(p) for p in PRESSURES],
        "magnitudes": list(identity["magnitudes"]),
        "best_factors": dict(identity["best_factors"]),
        "generation_tree_sha256": _tree_digest(output / "generations"),
        "judgment_tree_sha256": _tree_digest(output / "judgments"),
        "points": dict(points),
        "aggregate_sha256": _sha(output / "aggregate.parquet"),
        "per_concept_sha256": _sha(output / "per_concept.parquet"),
    }


def finalize(
    cfg: DictConfig, sources: EncDecSources, counter: Any, output: Path, identity: Mapping[str, Any]
) -> dict[str, Any]:
    if identity["phase"] != "generate":
        raise AxBenchAdvError("finalization requires ten-input generation, not a one-input pilot")
    if (output / "manifest.json").exists():
        return _validate_finished_shard(output, cfg, sources, counter)[1]
    cache_only = _judge_cache_only(cfg)
    grouped = _emissions(output, identity, sources)
    for pressure, magnitude in _point_coordinates(identity):
        point = _point_path(output, pressure, magnitude)
        if not (point / "judged").exists():
            judge_generated_point(
                source_frame=sources.source_main,
                point_dir=point,
                best_factors=identity["best_factors"],
                emissions=grouped[(pressure, magnitude)],
                max_concepts=50,
                concept_ids=identity["concept_ids"],
                cache_path=Path(str(cfg.judge_cache)),
                cache_only=cache_only,
                api_base_url=str(cfg.api_base_url),
                method_factors=METHOD_FACTORS_ENCDEC,
            )
    rows, per_concept, points = _point_results(output, identity, sources, grouped)
    points.update(_judge_manifests(output, identity))
    atomic_write_parquet(pa.Table.from_pylist(rows), output / "aggregate.parquet")
    atomic_write_parquet(pa.Table.from_pylist(per_concept), output / "per_concept.parquet")
    manifest = _manifest(output, identity, points)
    _install_json(output / "manifest.json", manifest)
    return manifest


def _validate_finished_shard(
    root: Path, cfg: DictConfig, sources: EncDecSources, counter: Any
) -> tuple[dict[str, Any], dict[str, Any], Emissions]:
    identity = read_json(root / "identity.json")
    if not isinstance(identity.get("runtime"), dict):
        raise AxBenchAdvError(f"encoder-decoder Adv shard has no runtime identity: {root}")
    expected_identity = _identity(
        cfg=cfg,
        sources=sources,
        counter=counter,
        concepts=_concepts(identity["concept_ids"]),
        magnitudes=_grid(identity["magnitudes"]),
        phase="generate",
        runtime=identity["runtime"],
    )
    if identity != expected_identity:
        raise AxBenchAdvError(f"encoder-decoder Adv shard identity differs: {root}")
    grouped = _emissions(root, identity, sources)
    rows, per_concept, points = _point_results(root, identity, sources, grouped)
    points.update(_judge_manifests(root, identity))
    if (
        pq.read_table(root / "aggregate.parquet").to_pylist() != rows
        or pq.read_table(root / "per_concept.parquet").to_pylist() != per_concept
    ):
        raise AxBenchAdvError(
            f"encoder-decoder Adv shard score tables differ from judged points: {root}"
        )
    manifest = read_json(root / "manifest.json")
    if manifest != _manifest(root, identity, points):
        raise AxBenchAdvError(f"encoder-decoder Adv shard manifest differs: {root}")
    return identity, manifest, grouped


def merge(cfg: DictConfig, sources: EncDecSources, counter: Any) -> dict[str, Any]:
    roots = tuple(Path(str(path)).resolve() for path in cfg.shard_roots)
    if not roots:
        raise AxBenchAdvError("merge requires at least one completed shard")
    shards = [_validate_finished_shard(root, cfg, sources, counter) for root in roots]
    identities = [identity for identity, _, _ in shards]
    coverage = [cid for identity in identities for cid in identity["concept_ids"]]
    if sorted(coverage) != list(range(50)):
        raise AxBenchAdvError("encoder-decoder Adv shards must cover concepts 0..49 exactly once")
    common = [
        {key: value for key, value in identity.items() if key != "concept_ids"}
        for identity in identities
    ]
    if any(value != common[0] for value in common[1:]):
        raise AxBenchAdvError("encoder-decoder Adv shards have incompatible identity or hardware")
    output = Path(str(cfg.output_dir)).resolve()
    if any(
        output == root or output.is_relative_to(root) or root.is_relative_to(output)
        for root in roots
    ):
        raise AxBenchAdvError("merge output must be disjoint from every shard")
    if output.exists() and any(output.iterdir()):
        raise AxBenchAdvError("encoder-decoder Adv merge output already exists")
    output.mkdir(parents=True, exist_ok=True)
    grouped: Emissions = {}
    for pressure, magnitude in _point_coordinates(identities[0]):
        point = _point_path(output, pressure, magnitude)
        for name in ("input/steering_data.parquet", "judged/steering_data.parquet"):
            tables = [pq.read_table(_point_path(root, pressure, magnitude) / name) for root in roots]
            table = pa.concat_tables(tables).sort_by(
                [("concept_id", "ascending"), ("input_id", "ascending"), ("factor", "ascending")]
            )
            (point / name).parent.mkdir(parents=True, exist_ok=True)
            atomic_write_parquet(table, point / name)
        entries = []
        for root in roots:
            path = _point_path(root, pressure, magnitude) / "judged/steering.jsonl"
            entries.extend(
                json.loads(line) for line in path.read_text().splitlines() if line.strip()
            )
        entries.sort(key=lambda entry: entry["concept_id"])
        (point / "judged/steering.jsonl").write_text(
            "".join(json.dumps(entry, allow_nan=False) + "\n" for entry in entries)
        )
        grouped[(pressure, magnitude)] = {
            key: raw
            for _, _, shard_emissions in shards
            for key, raw in shard_emissions[(pressure, magnitude)].items()
        }
    merged_rows, merged_per_concept, point_hashes = _point_results(
        output, {**identities[0], "concept_ids": list(range(50))}, sources, grouped
    )
    atomic_write_parquet(pa.Table.from_pylist(merged_rows), output / "aggregate.parquet")
    atomic_write_parquet(pa.Table.from_pylist(merged_per_concept), output / "per_concept.parquet")
    result = {
        "format": FORMAT,
        "status": "complete",
        "concept_ids": list(range(50)),
        "zero_control_source": "runtime",
        "shards": {str(root): _sha(root / "manifest.json") for root in roots},
        "points": point_hashes,
        "aggregate_sha256": _sha(output / "aggregate.parquet"),
        "per_concept_sha256": _sha(output / "per_concept.parquet"),
    }
    _install_json(output / "manifest.json", result)
    return result


def run_encdec_baseline_counter(cfg: DictConfig) -> dict[str, Any]:
    phase = str(cfg.phase)
    if phase not in {"preflight", "pilot", "generate", "judge", "finalize", "merge"}:
        raise AxBenchAdvError("phase must be preflight, pilot, generate, judge, finalize, or merge")
    sources = validate_encdec_sources(
        positive_root=Path(str(cfg.positive_root)),
        source_main_path=Path(str(cfg.source_main_path)),
        source_aux_path=Path(str(cfg.source_aux_path)),
        prepared_dir=Path(str(cfg.prepared_dir)),
        target_meta_path=Path(str(cfg.target_meta_path)),
        feature_statistics_path=Path(str(cfg.feature_statistics_path)),
        pretrain_meta_path=None
        if cfg.pretrain_meta_path is None
        else Path(str(cfg.pretrain_meta_path)),
        judged_input_path=Path(str(cfg.judged_input_path)),
        judged_dir=Path(str(cfg.judged_dir)),
    )
    counter = _counter_sources(cfg, sources)
    if phase == "preflight":
        return {
            "phase": "preflight_complete",
            "source": dict(sources.source_digest),
            "counter_manifest_sha256": counter.manifest_sha256,
            "best_factors": dict(sources.best_factors),
        }
    if phase == "merge":
        return merge(cfg, sources, counter)
    concepts = _concepts(cfg.concept_ids)
    magnitudes = _grid(cfg.counter_magnitudes)
    if phase == "pilot" and len(concepts) != 1:
        raise AxBenchAdvError("provider-free pilot requires one concept")
    output = Path(str(cfg.output_dir)).resolve()
    run_phase = "pilot" if phase == "pilot" else "generate"
    retained_runtime = None
    if phase in {"judge", "finalize"}:
        retained = read_json(output / "identity.json")
        retained_runtime = retained.get("runtime")
        if not isinstance(retained_runtime, dict):
            raise AxBenchAdvError("encoder-decoder Adv output has no bound GPU runtime")
    identity = _identity(
        cfg=cfg,
        sources=sources,
        counter=counter,
        concepts=concepts,
        magnitudes=magnitudes,
        phase=run_phase,
        runtime=retained_runtime,
    )
    _install_identity(output, identity)
    if phase in {"pilot", "generate"}:
        return generate(cfg, sources, counter, output, identity)
    if phase == "judge":
        return judge(cfg, sources, output, identity)
    return finalize(cfg, sources, counter, output, identity)
