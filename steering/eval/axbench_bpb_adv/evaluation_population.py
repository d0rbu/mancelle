"""Universal held-out evaluation population for AxBench-BPB-Adv.

The frontier compares method families only when they are scored on the exact
same class-margin examples. This module owns that method-independent contract:
``prepare_attack_rows`` writes the held-out prefix/pair population once, and
candidate, concept-score, roundtrip-score, and general-val stages require every
row to match it exactly.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import pyarrow as pa
from beartype import beartype
from beartype.vale import Is
from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from steering.immutable_io import ImmutableArtifactError, install_bytes_once
from steering.refined_types import (
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
    StrictInt,
)

EVALUATION_POPULATION_SCHEMA_VERSION = 1


@dataclass(frozen=True, order=True, init=False)
class EvaluationPairKey:
    concept_id: NonNegativeInt
    mode: SteeringMode
    core_prefix_id: NonEmptyStr
    concept_rich_output_digest: NonEmptyStr
    concept_avoiding_output_digest: NonEmptyStr

    @beartype
    def __init__(
        self,
        concept_id: NonNegativeIntLike,
        mode: SteeringMode,
        core_prefix_id: NonEmptyStrLike,
        concept_rich_output_digest: NonEmptyStrLike,
        concept_avoiding_output_digest: NonEmptyStrLike,
    ) -> None:
        object.__setattr__(self, "concept_id", NonNegativeInt.parse(concept_id))
        object.__setattr__(self, "mode", mode)
        object.__setattr__(
            self,
            "core_prefix_id",
            NonEmptyStr.parse(core_prefix_id),
        )
        object.__setattr__(
            self,
            "concept_rich_output_digest",
            NonEmptyStr.parse(concept_rich_output_digest),
        )
        object.__setattr__(
            self,
            "concept_avoiding_output_digest",
            NonEmptyStr.parse(concept_avoiding_output_digest),
        )


def _is_non_empty_evaluation_pair_key_frozenset(value: object) -> bool:
    return (
        isinstance(value, frozenset)
        and bool(value)
        and all(isinstance(item, EvaluationPairKey) for item in value)
    )


def _is_non_empty_evaluation_prefix_group_mapping(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(
            isinstance(group, tuple)
            and len(group) == 2
            and isinstance(group[0], NonNegativeInt)
            and isinstance(group[1], SteeringMode)
            and isinstance(prefix_ids, frozenset)
            and bool(prefix_ids)
            and all(isinstance(prefix_id, NonEmptyStr) for prefix_id in prefix_ids)
            for group, prefix_ids in value.items()
        )
    )


if TYPE_CHECKING:
    _EvaluationPairKeyFrozenSetLike = frozenset[EvaluationPairKey]
    _EvaluationPrefixGroupMappingLike = dict[
        tuple[NonNegativeInt, SteeringMode], frozenset[NonEmptyStr]
    ]
else:
    _EvaluationPairKeyFrozenSetLike = Annotated[
        frozenset,
        Is[_is_non_empty_evaluation_pair_key_frozenset],
    ]
    _EvaluationPrefixGroupMappingLike = Annotated[
        dict,
        Is[_is_non_empty_evaluation_prefix_group_mapping],
    ]


@dataclass(frozen=True, init=False)
class EvaluationPopulation:
    pair_keys: frozenset[EvaluationPairKey]
    prefix_ids_by_group: dict[tuple[NonNegativeInt, SteeringMode], frozenset[NonEmptyStr]]

    @beartype
    def __init__(
        self,
        pair_keys: _EvaluationPairKeyFrozenSetLike,
        prefix_ids_by_group: _EvaluationPrefixGroupMappingLike,
    ) -> None:
        object.__setattr__(self, "pair_keys", pair_keys)
        object.__setattr__(self, "prefix_ids_by_group", prefix_ids_by_group)
        expected: dict[
            tuple[NonNegativeInt, SteeringMode], set[NonEmptyStr]
        ] = defaultdict(set)
        for key in self.pair_keys:
            expected[(key.concept_id, key.mode)].add(key.core_prefix_id)
        if {
            group: frozenset(prefix_ids) for group, prefix_ids in expected.items()
        } != self.prefix_ids_by_group:
            raise AxBenchBpbAdvError(
                "evaluation population prefix groups must exactly match its pair keys"
            )

    @property
    def n_prefixes(self) -> int:
        return sum(len(prefix_ids) for prefix_ids in self.prefix_ids_by_group.values())

    @property
    def n_pairs(self) -> int:
        return len(self.pair_keys)


def _row_pair_key(row: Mapping[str, Any]) -> EvaluationPairKey:
    try:
        return EvaluationPairKey(
            concept_id=NonNegativeInt.parse(row["concept_id"]),
            mode=SteeringMode(row["mode"]),
            core_prefix_id=NonEmptyStr.parse(row["core_prefix_id"]),
            concept_rich_output_digest=NonEmptyStr.parse(
                row["concept_rich_output_digest"]
            ),
            concept_avoiding_output_digest=NonEmptyStr.parse(
                row["concept_avoiding_output_digest"]
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(
            f"invalid universal evaluation-population row key: {exc}"
        ) from exc


def evaluation_prefixes_for_scoring(
    prefixes: Sequence[BpbAttackPrefix],
    *,
    modes: set[SteeringMode],
    ours: OursGcgConfig,
) -> list[BpbAttackPrefix]:
    """Return the universal held-out prefix set for active scoring modes.

    The split is the same seeded optimize/held-out split used by fair
    feature-DiffMean/GCG. Only *modes* that will produce attack rows are split, so
    a source can omit the opposite mode when no configured method scores it.
    """

    groups: dict[
        tuple[NonNegativeInt, SteeringMode], list[BpbAttackPrefix]
    ] = defaultdict(list)
    for prefix in prefixes:
        if prefix.mode in modes:
            groups[(prefix.concept_id, prefix.mode)].append(prefix)
    if not groups:
        raise AxBenchBpbAdvError(
            "universal evaluation population: no prefixes for active scoring "
            f"modes {sorted(mode.value for mode in modes)}"
        )
    too_small = [
        f"concept_id={concept_id} mode={mode} n_prefixes={len(group_prefixes)}"
        for (concept_id, mode), group_prefixes in sorted(groups.items())
        if len(group_prefixes) < 2
    ]
    if too_small:
        raise AxBenchBpbAdvError(
            "universal evaluation population requires >= 2 accepted prefixes per "
            "(concept, mode) so every scored method, including decoder-only "
            "baselines, uses a deterministic held-out split. Offending groups: "
            f"{'; '.join(too_small)}"
        )

    scored_ids: set[NonEmptyStr] = set()
    for key, group_prefixes in sorted(groups.items()):
        _optimize, held_out = split_prefixes_for_concept(
            group_prefixes,
            ratio=ours.split_ratio,
            seed=ours.split_seed,
        )
        if not held_out:
            concept_id, mode = key
            raise AxBenchBpbAdvError(
                "universal evaluation population selected no HELD-OUT prefixes "
                f"for concept_id={concept_id} mode={mode!r} "
                f"(n_prefixes={len(group_prefixes)}, split_ratio={ours.split_ratio})"
            )
        scored_ids.update(prefix.core_prefix_id for prefix in held_out)

    out = [prefix for prefix in prefixes if prefix.core_prefix_id in scored_ids]
    logger.info(
        f"universal evaluation population: groups={len(groups)} "
        f"source_prefixes={sum(len(group) for group in groups.values())} "
        f"scored_prefixes={len(out)} split_ratio={ours.split_ratio} "
        f"split_seed={ours.split_seed}"
    )
    return out


def population_from_rows(rows: Sequence[Mapping[str, Any]]) -> EvaluationPopulation:
    pair_keys = frozenset(_row_pair_key(row) for row in rows)
    if not pair_keys:
        raise AxBenchBpbAdvError(
            "universal evaluation population cannot be built from zero rows"
        )
    grouped: dict[tuple[NonNegativeInt, SteeringMode], set[NonEmptyStr]] = defaultdict(set)
    for key in pair_keys:
        grouped[(key.concept_id, key.mode)].add(key.core_prefix_id)
    return EvaluationPopulation(
        pair_keys=pair_keys,
        prefix_ids_by_group={
            group: frozenset(prefix_ids) for group, prefix_ids in grouped.items()
        },
    )


def write_evaluation_population(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    ours: OursGcgConfig,
    feature_contract_fingerprint: str | None = None,
) -> None:
    population = population_from_rows(rows)
    pairs_by_group: dict[
        tuple[NonNegativeInt, SteeringMode], list[EvaluationPairKey]
    ] = defaultdict(list)
    for key in population.pair_keys:
        pairs_by_group[(key.concept_id, key.mode)].append(key)
    groups_payload: list[dict[str, Any]] = []
    for group in sorted(population.prefix_ids_by_group):
        concept_id, mode = group
        group_pairs = sorted(pairs_by_group[group])
        groups_payload.append(
            {
                "concept_id": concept_id,
                "mode": mode,
                "scored_core_prefix_ids": sorted(population.prefix_ids_by_group[group]),
                "scored_pairs": [
                    {
                        "core_prefix_id": key.core_prefix_id,
                        "concept_rich_output_digest": key.concept_rich_output_digest,
                        "concept_avoiding_output_digest": key.concept_avoiding_output_digest,
                    }
                    for key in group_pairs
                ],
            }
        )
    payload = {
        "schema_version": EVALUATION_POPULATION_SCHEMA_VERSION,
        "split_ratio": repr(float(ours.split_ratio)),
        "split_seed": int(ours.split_seed),
        "n_groups": len(groups_payload),
        "n_prefixes": population.n_prefixes,
        "n_pairs": population.n_pairs,
        "groups": groups_payload,
    }
    if feature_contract_fingerprint is None:
        atomic_json_dump(payload, path)
    else:
        payload["feature_contract_fingerprint"] = feature_contract_fingerprint
        rendered = json.dumps(payload, indent=2).encode("utf-8")
        try:
            install_bytes_once(
                rendered,
                path,
                surface="clustered evaluation population",
            )
        except ImmutableArtifactError as exc:
            raise AxBenchBpbAdvError(
                "clustered evaluation-population content changed on resume; "
                f"use a fresh run_tag instead of overwriting {path}: {exc}"
            ) from exc
    logger.info(
        "universal evaluation population written: groups={} prefixes={} pairs={} -> {}",
        payload["n_groups"],
        payload["n_prefixes"],
        payload["n_pairs"],
        path,
    )


def read_evaluation_population(
    path: Path,
    *,
    expected_feature_contract_fingerprint: str | None = None,
) -> EvaluationPopulation:
    if not path.exists():
        raise AxBenchBpbAdvError(
            f"missing universal evaluation-population artifact: {path}. "
            "Run prepare_attack_rows with the current code before build_candidates "
            "or any score stage; refusing to fall back to a full-prefix population."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        actual_fingerprint = payload.get("feature_contract_fingerprint")
        if expected_feature_contract_fingerprint is None:
            if actual_fingerprint is not None:
                raise AxBenchBpbAdvError(
                    "unclustered reader rejects cluster-stamped universal "
                    f"evaluation-population artifact {path}"
                )
        elif actual_fingerprint != expected_feature_contract_fingerprint:
            raise AxBenchBpbAdvError(
                "universal evaluation-population feature-contract fingerprint "
                f"mismatch in {path}: expected="
                f"{expected_feature_contract_fingerprint}, actual={actual_fingerprint!r}"
            )
        version = StrictInt.parse(payload["schema_version"])
        n_groups = NonNegativeInt.parse(payload["n_groups"])
        n_prefixes = NonNegativeInt.parse(payload["n_prefixes"])
        n_pairs = NonNegativeInt.parse(payload["n_pairs"])
        groups = payload["groups"]
        if version != EVALUATION_POPULATION_SCHEMA_VERSION:
            raise AxBenchBpbAdvError(
                f"unsupported universal evaluation-population schema_version={version} "
                f"in {path}; expected {EVALUATION_POPULATION_SCHEMA_VERSION}"
            )
        if not isinstance(groups, list):
            raise AxBenchBpbAdvError(
                f"universal evaluation-population artifact {path} has non-list groups"
            )

        keys: set[EvaluationPairKey] = set()
        prefix_ids_by_group: dict[
            tuple[NonNegativeInt, SteeringMode], frozenset[NonEmptyStr]
        ] = {}
        for group in groups:
            if not isinstance(group, Mapping):
                raise AxBenchBpbAdvError(
                    f"universal evaluation-population artifact {path} has non-object group"
                )
            concept_id = NonNegativeInt.parse(group["concept_id"])
            mode = SteeringMode(group["mode"])
            core_prefix_ids = group["scored_core_prefix_ids"]
            pairs = group["scored_pairs"]
            if not isinstance(core_prefix_ids, list):
                raise AxBenchBpbAdvError(
                    f"universal evaluation-population artifact {path} has non-list "
                    "scored_core_prefix_ids"
                )
            if not isinstance(pairs, list):
                raise AxBenchBpbAdvError(
                    f"universal evaluation-population artifact {path} has non-list pairs"
                )
            group_key = (concept_id, mode)
            if group_key in prefix_ids_by_group:
                raise AxBenchBpbAdvError(
                    f"universal evaluation-population artifact {path} has duplicate group "
                    f"concept_id={concept_id} mode={mode!r}"
                )
            group_prefixes = frozenset(
                NonEmptyStr.parse(prefix_id) for prefix_id in core_prefix_ids
            )
            pair_prefixes: set[NonEmptyStr] = set()
            for pair in pairs:
                if not isinstance(pair, Mapping):
                    raise AxBenchBpbAdvError(
                        f"universal evaluation-population artifact {path} has "
                        "non-object pair"
                    )
                key = EvaluationPairKey(
                    concept_id=concept_id,
                    mode=mode,
                    core_prefix_id=NonEmptyStr.parse(pair["core_prefix_id"]),
                    concept_rich_output_digest=NonEmptyStr.parse(
                        pair["concept_rich_output_digest"]
                    ),
                    concept_avoiding_output_digest=NonEmptyStr.parse(
                        pair["concept_avoiding_output_digest"]
                    ),
                )
                if key in keys:
                    raise AxBenchBpbAdvError(
                        f"universal evaluation-population artifact {path} has "
                        f"duplicate pair key for core_prefix_id={key.core_prefix_id!r}"
                    )
                keys.add(key)
                pair_prefixes.add(key.core_prefix_id)
            if pair_prefixes != group_prefixes:
                raise AxBenchBpbAdvError(
                    f"universal evaluation-population artifact {path} has mismatched "
                    f"scored_core_prefix_ids for concept_id={concept_id} mode={mode!r}"
                )
            prefix_ids_by_group[group_key] = group_prefixes
        if not keys:
            raise AxBenchBpbAdvError(
                "universal evaluation population cannot be built from zero rows"
            )
        population = EvaluationPopulation(
            pair_keys=frozenset(keys),
            prefix_ids_by_group=prefix_ids_by_group,
        )
        if n_groups != len(prefix_ids_by_group):
            raise AxBenchBpbAdvError(
                f"universal evaluation-population artifact {path} declares n_groups="
                f"{n_groups}, but contains {len(prefix_ids_by_group)} group(s)"
            )
        if n_prefixes != population.n_prefixes:
            raise AxBenchBpbAdvError(
                f"universal evaluation-population artifact {path} declares n_prefixes="
                f"{n_prefixes}, but contains {population.n_prefixes} prefix(es)"
            )
        if n_pairs != population.n_pairs:
            raise AxBenchBpbAdvError(
                f"universal evaluation-population artifact {path} declares n_pairs="
                f"{n_pairs}, but contains {population.n_pairs} pair(s)"
            )
        return population
    except AxBenchBpbAdvError:
        raise
    except Exception as exc:
        raise AxBenchBpbAdvError(
            f"corrupt or unreadable universal evaluation-population artifact {path}: {exc}"
        ) from exc


def _require_evaluation_population_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    population: EvaluationPopulation,
    label: str,
    stats: dict[str, int] | None,
) -> None:
    kept_by_method: dict[str, int] = {}
    dropped_by_method: dict[str, int] = {}
    missing_examples: list[str] = []
    for row in rows:
        key = _row_pair_key(row)
        group = (key.concept_id, key.mode)
        if group not in population.prefix_ids_by_group:
            raise AxBenchBpbAdvError(
                f"{label}: row references concept_id={key.concept_id} mode={key.mode!r}, "
                "which is absent from the universal evaluation population"
            )
        keep = key in population.pair_keys
        method_name = str(row.get("method_name", "<unknown>"))
        bucket = kept_by_method if keep else dropped_by_method
        bucket[method_name] = bucket.get(method_name, 0) + 1
        if not keep and len(missing_examples) < 8:
            missing_examples.append(
                f"method={method_name} concept_id={key.concept_id} mode={key.mode} "
                f"core_prefix_id={key.core_prefix_id} "
                f"rich_digest={key.concept_rich_output_digest} "
                f"avoid_digest={key.concept_avoiding_output_digest}"
            )

    n_kept = sum(kept_by_method.values())
    n_dropped = sum(dropped_by_method.values())
    kept_sorted = dict(sorted(kept_by_method.items()))
    dropped_sorted = dict(sorted(dropped_by_method.items()))
    logger.info(
        f"{label} universal evaluation-population coverage: kept {n_kept} "
        f"dropped {n_dropped} row(s) (kept per method={kept_sorted}, "
        f"dropped per method={dropped_sorted})"
    )
    if stats is not None:
        stats["evaluation_population_coverage_kept"] = n_kept
        stats["evaluation_population_coverage_dropped"] = n_dropped
    if n_dropped:
        examples = "; ".join(missing_examples)
        raise AxBenchBpbAdvError(
            f"{label}: {n_dropped} row(s) were in a universal evaluation-population "
            "group but absent from the exact held-out pair keys; refusing to use "
            f"an incomparable subset (dropped per method={dropped_sorted}; "
            f"examples={examples}). Regenerate this run from prepare_attack_rows "
            "with the current code."
        )


def require_rows_in_evaluation_population(
    rows: list[dict[str, Any]],
    *,
    population: EvaluationPopulation,
    label: str,
) -> list[dict[str, Any]]:
    if not rows:
        return rows
    _require_evaluation_population_coverage(
        rows,
        population=population,
        label=label,
        stats=None,
    )
    return rows


def require_table_in_evaluation_population(
    table: pa.Table,
    *,
    population: EvaluationPopulation,
    label: str,
    stats: dict[str, int] | None = None,
) -> pa.Table:
    if table.num_rows == 0:
        return table
    _require_evaluation_population_coverage(
        table.to_pylist(),
        population=population,
        label=label,
        stats=stats,
    )
    return table


__all__ = [
    "EvaluationPairKey",
    "EvaluationPopulation",
    "evaluation_prefixes_for_scoring",
    "population_from_rows",
    "read_evaluation_population",
    "require_rows_in_evaluation_population",
    "require_table_in_evaluation_population",
    "write_evaluation_population",
]
