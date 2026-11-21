"""Arrow schemas for AxBench-Adv's portable artifacts."""

from __future__ import annotations

import hashlib
import math

import pyarrow as pa

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.method_grid import ALL_METHOD_FACTORS
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix
from steering.eval.axbench_adv.types import ALL_METHODS, PROPOSAL_COUNTS

UNIVERSAL_SUFFIXES_SCHEMA: pa.Schema = pa.schema(
    [
        ("method", pa.string()),
        ("concept_id", pa.int32()),
        ("proposal_count", pa.int32()),
        ("coefficient", pa.float64()),
        ("suffix", pa.string()),
        ("suffix_sha256", pa.string()),
        ("master_seed", pa.int64()),
        ("best_fitness", pa.float64()),
        ("gepa_nominal_metric_calls", pa.int64()),
        ("actual_evaluator_calls", pa.int64()),
        ("actual_generation_calls", pa.int64()),
        ("paid_judge_cache_misses", pa.int64()),
        ("paid_reflection_cache_misses", pa.int64()),
    ]
)

_COUNTER_COLUMNS = (
    "gepa_nominal_metric_calls",
    "actual_evaluator_calls",
    "actual_generation_calls",
    "paid_judge_cache_misses",
    "paid_reflection_cache_misses",
)


def suffix_sha256(suffix: str) -> str:
    """Hash the exact canonical UTF-8 suffix bytes."""

    canonical = canonicalize_suffix(suffix)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_universal_suffixes(table: pa.Table) -> None:
    """Validate schema plus semantic identity for optimized suffix rows."""

    if not isinstance(table, pa.Table):
        raise AxBenchAdvError("universal suffixes must be a PyArrow table")
    if not table.schema.equals(UNIVERSAL_SUFFIXES_SCHEMA):
        raise AxBenchAdvError("table does not conform to UNIVERSAL_SUFFIXES_SCHEMA")
    if table.num_rows == 0:
        raise AxBenchAdvError("universal suffixes table must not be empty")
    keys: set[tuple[str, int, int]] = set()
    for row in table.to_pylist():
        if any(value is None for value in row.values()):
            raise AxBenchAdvError("universal suffix rows must not contain null values")
        method = str(row["method"])
        concept_id = int(row["concept_id"])
        proposal_count = int(row["proposal_count"])
        key = (method, concept_id, proposal_count)
        if key in keys:
            raise AxBenchAdvError(f"duplicate universal suffix key {key}")
        keys.add(key)
        if method not in ALL_METHODS:
            raise AxBenchAdvError(f"unknown universal suffix method {method!r}")
        if concept_id < 0:
            raise AxBenchAdvError("universal suffix concept_id must be non-negative")
        if proposal_count not in PROPOSAL_COUNTS:
            raise AxBenchAdvError(
                f"proposal_count must be one of {PROPOSAL_COUNTS}, got {proposal_count}"
            )
        coefficient = float(row["coefficient"])
        fitness = float(row["best_fitness"])
        if not math.isfinite(coefficient):
            raise AxBenchAdvError("universal suffix coefficient must be finite")
        if coefficient not in tuple(float(value) for value in ALL_METHOD_FACTORS[method]):
            raise AxBenchAdvError(
                "universal suffix coefficient is outside the method's factor grid"
            )
        if not math.isfinite(fitness) or not 0.0 <= fitness <= 1.0:
            raise AxBenchAdvError("universal suffix best_fitness must be finite in [0, 1]")
        suffix = str(row["suffix"])
        if canonicalize_suffix(suffix) != suffix:
            raise AxBenchAdvError("universal suffix text must already be canonical")
        if suffix_sha256(suffix) != row["suffix_sha256"]:
            raise AxBenchAdvError("universal suffix SHA-256 does not match its text")
        if int(row["master_seed"]) < 0:
            raise AxBenchAdvError("master_seed must be non-negative")
        for column in _COUNTER_COLUMNS:
            if int(row[column]) < 0:
                raise AxBenchAdvError(f"{column} must be non-negative")


__all__ = [
    "UNIVERSAL_SUFFIXES_SCHEMA",
    "suffix_sha256",
    "validate_universal_suffixes",
]
