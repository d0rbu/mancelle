"""Final AxBench judges and aggregates for native standalone-method Adv runs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from omegaconf import DictConfig

from scripts.eval.axbench_adv.pressures import (
    COUNTER_PRESSURES,
    DIRECTION_METHOD,
    Pressure,
    magnitude_label,
    validate_magnitudes,
)
from scripts.eval.axbench_upstream.run_evaluation import DEFAULT_API_BASE_URL
from steering.eval.axbench_adv.aggregation import (
    build_aggregate,
    build_counter_aggregate,
)
from steering.eval.axbench_adv.artifacts import logical_table_digest
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import (
    _CALL_ACCOUNTING_FILE,
    FinalJudgeResult,
    build_point_frame,
    run_final_judge,
    validate_final_judge_outputs,
)
from steering.eval.axbench_adv.fitness import evaluate_fitness
from steering.eval.axbench_adv.native_control_judge import control_lineage, replay_native_control
from steering.eval.axbench_adv.native_mailbox import digest_json, read_json, write_json
from steering.eval.axbench_adv.native_methods import (
    FORMAT,
    METHODS,
    ZERO_CONTROL_ARCHIVED,
    ZERO_CONTROL_RUNTIME,
    _identity_zero_control_source,
    _validate_pressures,
    load_native_method_emissions,
)
from steering.eval.axbench_adv.plotting import plot_concept_vs_fluency
from steering.eval.axbench_upstream.native_contract import sha256


def _identity_grid(identity: Mapping[str, Any]) -> tuple[tuple[Pressure, ...], tuple[float, ...]]:
    raw_pressures = identity.get("pressures")
    raw_magnitudes = identity.get("counter_magnitudes")
    if not isinstance(raw_pressures, list) or any(
        not isinstance(pressure, str) for pressure in raw_pressures
    ):
        raise AxBenchAdvError("native method identity pressures are invalid")
    pressures = _validate_pressures(raw_pressures)
    if raw_pressures != [str(pressure) for pressure in pressures]:
        raise AxBenchAdvError("native method identity pressures are not canonical")
    if not isinstance(raw_magnitudes, list) or any(
        type(magnitude) is not float for magnitude in raw_magnitudes
    ):
        raise AxBenchAdvError("native method identity counter magnitudes are invalid")
    try:
        magnitudes = validate_magnitudes(raw_magnitudes)
    except (TypeError, ValueError) as exc:
        raise AxBenchAdvError("native method identity counter magnitudes are invalid") from exc
    if raw_magnitudes != list(magnitudes):
        raise AxBenchAdvError("native method identity counter magnitudes are not canonical")
    return pressures, magnitudes


def _point_specs(identity: Mapping[str, Any]) -> list[tuple[Path, dict[str, Any]]]:
    pressures, magnitudes = _identity_grid(identity)
    proposals = tuple(identity["proposal_counts"])
    specs = [
        (Path("points") / "p0", {"pressure": "shared_zero", "proposal_count": 0, "magnitude": 0.0})
    ]
    if Pressure.GEPA in pressures:
        specs.extend(
            (
                Path("gepa") / "points" / f"p{proposal}",
                {"pressure": str(Pressure.GEPA), "proposal_count": proposal},
            )
            for proposal in proposals
        )
    specs.extend(
        (
            Path("counter") / pressure / "points" / magnitude_label(magnitude),
            {"pressure": str(pressure), "magnitude": magnitude},
        )
        for pressure in pressures
        if pressure in COUNTER_PRESSURES
        for magnitude in magnitudes
    )
    return specs


def _validate_point(
    *,
    run_dir: Path,
    relative: Path,
    coordinate: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[Any, dict[tuple[str, int, int], Any]]:
    point = run_dir / relative
    point_identity = read_json(point / "point.json")
    best_factors = identity["best_factors"]
    concepts = tuple(identity["concept_ids"])
    if (
        point_identity.get("format") != FORMAT
        or point_identity.get("run_identity_sha256") != digest_json(identity)
        or point_identity.get("coordinate") != dict(coordinate)
        or point_identity.get("best_factors") != best_factors
        or point_identity.get("concept_ids") != list(concepts)
    ):
        raise AxBenchAdvError(f"native method point identity differs: {relative}")
    input_path = point / "input" / "steering_data.parquet"
    emissions_path = point / "emissions.json"
    if sha256(input_path) != point_identity.get("input_sha256") or sha256(
        emissions_path
    ) != point_identity.get("emissions_sha256"):
        raise AxBenchAdvError(f"native method point hashes differ: {relative}")
    emissions = load_native_method_emissions(emissions_path, methods=METHODS, concept_ids=concepts)
    table = pq.read_table(input_path)
    source = table.to_pandas()
    expected = build_point_frame(
        source_frame=source,
        best_factors=best_factors,
        emissions=emissions,
        max_concepts=100,
        concept_ids=concepts,
    )
    if not table.equals(expected):
        raise AxBenchAdvError(f"native method point input differs from emissions: {relative}")
    return table, emissions


def _validated_judge_receipt(
    *,
    output: Path,
    source: Mapping[str, str],
    best_factors: Mapping[str, float],
    relative: Path,
    identity: Mapping[str, Any],
) -> dict[str, Any]:
    manifest = read_json(output / "manifest.json")
    accounting = manifest.get("call_accounting")
    files = manifest.get("files")
    replay = manifest.get("control_replay")
    zero_control_source = _identity_zero_control_source(identity)
    is_control = relative == Path("points/p0")
    if is_control and zero_control_source == ZERO_CONTROL_ARCHIVED:
        if (
            "control_replay" not in manifest
            or replay != control_lineage(read_json(output.parents[2] / "identity.json"))
            or accounting != {"actual_evaluator_calls": 0, "paid_judge_cache_misses": 0}
            or read_json(output / _CALL_ACCOUNTING_FILE)
            != {"total_calls": 0, "total_cache_hits": 0}
        ):
            raise AxBenchAdvError("native control replay lineage/accounting differs")
    elif "control_replay" in manifest:
        raise AxBenchAdvError("native control replay lineage/accounting differs")
    if is_control and zero_control_source == ZERO_CONTROL_RUNTIME:
        expected_calls = len(METHODS) * len(identity["concept_ids"]) * 10 * 3
        call_accounting = read_json(output / _CALL_ACCOUNTING_FILE)
        calls = call_accounting.get("total_calls")
        hits = call_accounting.get("total_cache_hits")
        if (
            not isinstance(accounting, dict)
            or accounting.get("actual_evaluator_calls") != expected_calls
            or set(call_accounting) != {"total_calls", "total_cache_hits"}
            or type(calls) is not int
            or type(hits) is not int
            or calls != expected_calls
            or not 0 <= hits <= calls
        ):
            raise AxBenchAdvError("runtime control final-judge accounting differs")
    if (
        set(manifest)
        != {
            "schema_version",
            "best_factors",
            "call_accounting",
            "files",
            "input_logical_sha256",
        }
        | ({"control_replay"} if replay is not None else set())
        or manifest.get("schema_version") != 1
        or manifest.get("best_factors") != best_factors
        or manifest.get("input_logical_sha256") != source["input_logical_sha256"]
        or not isinstance(accounting, dict)
        or set(accounting) != {"actual_evaluator_calls", "paid_judge_cache_misses"}
        or any(type(value) is not int or value < 0 for value in accounting.values())
        or not isinstance(files, dict)
        or set(files) != {_CALL_ACCOUNTING_FILE, "steering.jsonl", "steering_data.parquet"}
        or files.get(_CALL_ACCOUNTING_FILE) != {"sha256": sha256(output / _CALL_ACCOUNTING_FILE)}
        or files.get("steering.jsonl") != {"sha256": sha256(output / "steering.jsonl")}
        or files.get("steering_data.parquet")
        != {"logical_sha256": logical_table_digest(pq.read_table(output / "steering_data.parquet"))}
    ):
        raise AxBenchAdvError(f"native completed judge manifest differs: {relative}")
    receipt = {
        "source": dict(source),
        "steering_jsonl": str(output / "steering.jsonl"),
        **accounting,
        **({"control_replay": replay} if replay is not None else {}),
    }
    if is_control and zero_control_source == ZERO_CONTROL_RUNTIME:
        receipt.update(
            zero_control_source=ZERO_CONTROL_RUNTIME,
            run_identity_sha256=digest_json(identity),
        )
    return receipt


def _judge_point(
    *,
    run_dir: Path,
    relative: Path,
    table: Any,
    emissions: Mapping[tuple[str, int, int], Any],
    identity: Mapping[str, Any],
    cache: Path,
    cache_only: bool,
    api_base_url: str,
    judge: Callable[..., FinalJudgeResult],
    final_root: Path | None = None,
) -> Path:
    point = run_dir / relative
    output = point / "judged"
    input_path = point / "input" / "steering_data.parquet"
    best_factors = identity["best_factors"]
    concepts = tuple(identity["concept_ids"])
    receipt_path = point / "judge-receipt.json"
    source = {
        "input_sha256": sha256(input_path),
        "input_logical_sha256": logical_table_digest(table),
        "emissions_sha256": sha256(point / "emissions.json"),
    }

    if output.exists():
        validate_final_judge_outputs(
            input_table=table,
            output_dir=output,
            best_factors=best_factors,
            emissions=emissions,
            max_concepts=100,
            concept_ids=concepts,
        )
        expected_receipt = _validated_judge_receipt(
            output=output,
            source=source,
            best_factors=best_factors,
            relative=relative,
            identity=identity,
        )
        if not receipt_path.exists():
            write_json(receipt_path, expected_receipt)
        else:
            actual_receipt = read_json(receipt_path)
            location = actual_receipt.pop("steering_jsonl", None)
            suffix = (relative / "judged" / "steering.jsonl").parts
            # Match the archive reader: only this locator may relocate. All
            # source/output hashes, accounting and replay lineage stay exact.
            if (
                not isinstance(location, str)
                or Path(location).parts[-len(suffix) :] != suffix
                or actual_receipt
                != {
                    key: value for key, value in expected_receipt.items() if key != "steering_jsonl"
                }
            ):
                raise AxBenchAdvError(f"native completed judge receipt differs: {relative}")
        return output / "steering.jsonl"
    zero_control_source = _identity_zero_control_source(identity)
    if relative == Path("points/p0") and zero_control_source == ZERO_CONTROL_ARCHIVED:
        if final_root is None:
            raise AxBenchAdvError(
                "native shared control requires final_root for authenticated judge replay"
            )
        result = replay_native_control(
            final_root=final_root,
            identity=identity,
            input_table=table,
            emissions=emissions,
            output_dir=output,
        )
    else:
        result = judge(
            input_table=table,
            input_path=input_path,
            output_dir=output,
            cache_path=cache,
            best_factors=best_factors,
            emissions=emissions,
            max_concepts=100,
            concept_ids=concepts,
            cache_only=cache_only,
            api_base_url=api_base_url,
        )
    expected_receipt = _validated_judge_receipt(
        output=output,
        source=source,
        best_factors=best_factors,
        relative=relative,
        identity=identity,
    )
    if (
        result.steering_jsonl != output / "steering.jsonl"
        or result.actual_evaluator_calls != expected_receipt["actual_evaluator_calls"]
        or result.paid_judge_cache_misses != expected_receipt["paid_judge_cache_misses"]
    ):
        raise AxBenchAdvError(f"native final judge result differs from manifest: {relative}")
    write_json(receipt_path, expected_receipt)
    return result.steering_jsonl


def run_native_method_final_judge(
    cfg: DictConfig,
    *,
    judge: Callable[..., FinalJudgeResult] = run_final_judge,
) -> dict[str, Any]:
    """Judge every selected pressure point and publish its aggregates."""

    run_dir = Path(str(cfg.run_dir)).resolve()
    cache = Path(str(cfg.judge_cache)).resolve()
    identity = read_json(run_dir / "identity.json")
    if identity.get("format") != FORMAT or identity.get("methods") != list(METHODS):
        raise AxBenchAdvError("native method final judge run identity differs")
    _identity_zero_control_source(identity)
    if run_dir == cache or run_dir.is_relative_to(cache) or cache.is_relative_to(run_dir):
        raise AxBenchAdvError("native method final judge cache/run paths must be disjoint")
    api_base_url = str(cfg.api_base_url)
    if not api_base_url:
        api_base_url = DEFAULT_API_BASE_URL
    judged: dict[Path, Path] = {}
    emissions_by_point: dict[Path, Mapping[tuple[str, int, int], Any]] = {}
    # Reject incomplete or altered generation artifacts before any paid judging.
    validated = {}
    for relative, coordinate in _point_specs(identity):
        validated[relative] = _validate_point(
            run_dir=run_dir,
            relative=relative,
            coordinate=coordinate,
            identity=identity,
        )
    final_root = Path(str(cfg.final_root)).resolve() if cfg.get("final_root") else None
    for relative, (table, emissions) in validated.items():
        judged[relative] = _judge_point(
            run_dir=run_dir,
            relative=relative,
            table=table,
            emissions=emissions,
            identity=identity,
            cache=cache,
            cache_only=bool(cfg.cache_only),
            api_base_url=api_base_url,
            judge=judge,
            final_root=final_root,
        )
        emissions_by_point[relative] = emissions
    factors = identity["best_factors"]
    concepts = tuple(identity["concept_ids"])
    proposals = tuple(identity["proposal_counts"])
    pressures, magnitudes = _identity_grid(identity)
    control = judged[Path("points") / "p0"]
    if Pressure.GEPA in pressures:
        gepa_sources = {(method, 0): control for method in METHODS} | {
            (method, proposal): judged[Path("gepa") / "points" / f"p{proposal}"]
            for method in METHODS
            for proposal in proposals
        }
        aggregate = build_aggregate(
            best_factors=factors,
            judged_points=gepa_sources,
            expected_n_concepts=len(concepts),
            expected_concept_ids=concepts,
            output_path=run_dir / "gepa" / "aggregate.parquet",
        )
        plot_concept_vs_fluency(
            aggregate, run_dir / "gepa" / "plots", pressure=Pressure.GEPA
        )
    for pressure in pressures:
        if pressure not in COUNTER_PRESSURES:
            continue
        judged_points = {(method, 0.0): control for method in METHODS}
        fitness: dict[tuple[str, float], float] = {}
        for method in METHODS:
            zero = [
                raw
                for (candidate, _concept, _input), raw in emissions_by_point[
                    Path("points") / "p0"
                ].items()
                if candidate == method
            ]
            fitness[(method, 0.0)] = sum(
                evaluate_fitness(
                    concept_rating=raw.concept_rating,
                    fluency_rating=raw.fluency_rating,
                    output=raw.generation,
                ).score
                for raw in zero
            ) / len(zero)
        for magnitude in magnitudes:
            relative = Path("counter") / pressure / "points" / magnitude_label(magnitude)
            for method in METHODS:
                judged_points[(method, float(magnitude))] = judged[relative]
                selected = [
                    raw
                    for (candidate, _concept, _input), raw in emissions_by_point[relative].items()
                    if candidate == method
                ]
                score = sum(
                    evaluate_fitness(
                        concept_rating=raw.concept_rating,
                        fluency_rating=raw.fluency_rating,
                        output=raw.generation,
                    ).score
                    for raw in selected
                ) / len(selected)
                if not math.isfinite(score):
                    raise AxBenchAdvError("native method counter fitness is nonfinite")
                fitness[(method, float(magnitude))] = score
        counter = build_counter_aggregate(
            best_factors=factors,
            judged_points=judged_points,
            magnitudes=magnitudes,
            fitness=fitness,
            concept_ids=concepts,
            output_path=run_dir / "counter" / pressure / "aggregate.parquet",
        )
        plot_concept_vs_fluency(
            counter, run_dir / "counter" / pressure / "plots", pressure=pressure
        )
    result = {
        "phase": "complete",
        "methods": list(METHODS),
        "concept_ids": list(concepts),
        "best_factors": factors,
        "pressures": identity["pressures"],
        "counter_direction_families": {
            str(pressure): str(DIRECTION_METHOD[pressure])
            for pressure in pressures
            if pressure in COUNTER_PRESSURES
        },
    }
    write_json(run_dir / "status.json", result)
    return result


__all__ = ["run_native_method_final_judge"]
