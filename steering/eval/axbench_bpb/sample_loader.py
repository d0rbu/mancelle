# steering/eval/axbench_bpb/sample_loader.py
"""Load synth parquet rows into ScoringInputs. Uses deterministic SHA256
derivatives for sample_id + prefix_hash so the same sample always has
the same identifier across processes and runs."""
from __future__ import annotations

import hashlib
from collections.abc import Collection
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from steering.eval.axbench_base.positive_qc_contract import (
    validate_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.synth_continuation_contract import (
    Stage1ContinuationArtifactRole,
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_bpb.types import SteeringMode


def make_sample_id(
    *,
    concept_id: int,
    mode: SteeringMode,
    content_class: str,
    prefix_text: str,
    output_text: str,
) -> str:
    """16-hex-char SHA256 derivative uniquely identifying a sample.

    CONTRACT: changing the field order, separator, encoding, or truncation
    length changes aggregation identity. Do not modify casually.
    """
    payload = (
        f"{concept_id}|{mode.value}|{content_class}|{prefix_text}|{output_text}"
    ).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def make_prefix_hash(prefix_text: str) -> str:
    """16-hex-char SHA256 derivative of prefix_text. Used as context-cache
    filename + manifest lookup key."""
    return hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()[:16]


def synth_rows_from_table(table: Any, *, filter_unaccepted: bool = True) -> list[dict[str, Any]]:
    """Convert a synth parquet table into row dicts.

    ``load_raw_synth_rows`` and the BPB-Adv source validator both use this
    helper so preloaded parquet tables and path-loaded parquets apply the same
    QC predicate.
    """
    rows = table.to_pylist()
    if filter_unaccepted:
        rows = [r for r in rows if r["qc_accepted"]]
    return rows


def load_raw_synth_rows(
    path: Path,
    *,
    filter_unaccepted: bool = True,
    require_positive_qc_contract: bool = False,
    expected_continuation_role: Stage1ContinuationArtifactRole | None = None,
    concept_ids: Collection[int] | None = None,
) -> list[dict[str, Any]]:
    """Load a Stage 1b/1c synth parquet into a list of row dicts.

    Works for both the positive-mode parquet (Stage 1b output) and the
    negative-mode parquet (Stage 1c output). Negative-mode rows have 4
    extra columns (``prefix_source_shard``, ``prefix_source_doc``,
    ``prefix_source_seg_idx``, ``qc_zero_activation_verified``) that round-
    trip via pyarrow without special handling.

    If ``filter_unaccepted`` (default), drop rows with ``qc_accepted=False``.
    If ``concept_ids`` is provided, also remove rows outside the explicit final
    Stage 1 population. This is a population boundary, not a discovery step.
    """
    table = pq.read_table(path)
    if expected_continuation_role is not None:
        validate_stage1_continuation_contract(
            path,
            consumer="AxBench-BPB synth loader",
            expected_role=expected_continuation_role,
            schema=table.schema,
        )
    if require_positive_qc_contract:
        validate_positive_qc_contract(
            path,
            consumer="AxBench-BPB positive synth loader",
            schema=table.schema,
        )
        validate_positive_selection_contract(
            path,
            consumer="AxBench-BPB positive synth loader",
            schema=table.schema,
        )
    rows = synth_rows_from_table(
        table,
        filter_unaccepted=filter_unaccepted,
    )
    if concept_ids is None:
        return rows
    allowed = set(concept_ids)
    return [row for row in rows if int(row["concept_id"]) in allowed]


def collect_unique_prefixes(
    positive_path: Path,
    negative_path: Path,
    *,
    filter_unaccepted: bool = True,
    concept_ids: Collection[int] | None = None,
) -> dict[str, str]:
    """Return {prefix_text: prefix_hash} for every unique prefix_text
    across both synth parquets."""
    all_rows = load_raw_synth_rows(
        positive_path,
        filter_unaccepted=filter_unaccepted,
        require_positive_qc_contract=True,
        expected_continuation_role="stage1b_positive",
        concept_ids=concept_ids,
    )
    all_rows += load_raw_synth_rows(
        negative_path,
        filter_unaccepted=filter_unaccepted,
        expected_continuation_role="stage1c_negative",
        concept_ids=concept_ids,
    )
    result: dict[str, str] = {}
    for row in all_rows:
        pt = row["prefix_text"]
        if pt not in result:
            result[pt] = make_prefix_hash(pt)
    return result
