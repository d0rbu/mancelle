"""End-to-end check that the mini base fixture yields a FINITE positive-mode
``primary_margin`` (plan §1.4).

Before the per-prefix class-margin refactor, positive-mode ``concept_avoiding``
rows were faked with a shared global-negative pool (``concept_id == -1``), so the
positive ``(concept_id, mode=positive)`` aggregation group held only
``concept_rich`` rows and ``compute_per_tuple_rollup`` produced ``primary_margin
= NaN``. The refactor makes positive mode symmetric: every accepted
``concept_rich`` row owns a paired ``concept_avoiding`` row of the SAME prefix
with ``concept_id == cid``. Both land in the positive group and a real margin is
produced.

This test loads the regenerated fixture's positive parquet, asserts the
per-prefix pairing data model, then drives the production aggregation
(``compute_per_tuple_rollup``) on score rows derived from those samples and
asserts the resulting positive ``primary_margin`` is finite.
"""
from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import pyarrow as pa

from steering.eval.axbench_bpb.bpb_aggregation import compute_per_tuple_rollup
from steering.eval.axbench_bpb.sample_loader import load_raw_synth_rows, make_sample_id
from steering.eval.axbench_bpb.types import SteeringMode

_FIXTURE_ROOT = (
    Path(__file__).resolve().parent.parent
    / "fixtures" / "mini_axbench_base_run"
)


def _positive_rows() -> list[dict]:
    return load_raw_synth_rows(
        _FIXTURE_ROOT / "synth_training_positive.parquet",
        filter_unaccepted=True,
        require_positive_qc_contract=True,
    )


def test_positive_parquet_pairs_rich_and_avoid_on_same_prefix() -> None:
    """Each positive prefix owns BOTH a concept_rich and a concept_avoiding row
    with concept_id == cid (never the legacy global-pool sentinel -1)."""
    rows = _positive_rows()
    assert rows, "fixture positive parquet is empty"

    # No legacy global-negative sentinel anywhere.
    assert all(r["concept_id"] != -1 for r in rows), (
        "positive parquet still contains concept_id == -1 global-pool rows"
    )

    by_prefix: dict[tuple[int, str], set[str]] = defaultdict(set)
    for r in rows:
        by_prefix[(r["concept_id"], r["prefix_text"])].add(r["content_class"])

    for (cid, prefix), classes in by_prefix.items():
        assert classes == {"concept_rich", "concept_avoiding"}, (
            f"prefix (cid={cid!r}, {prefix!r}) is missing a side: got {classes}; "
            "positive mode must pair concept_rich + concept_avoiding per prefix"
        )


def test_positive_mode_primary_margin_is_finite_end_to_end() -> None:
    """Driving production aggregation over the fixture's positive samples yields
    a finite positive-mode primary_margin (was NaN under the global pool)."""
    rows = _positive_rows()

    # Build a deterministic score table: concept_avoiding scores HIGHER BPB
    # than concept_rich, so positive primary_margin = avoid - rich > 0 and finite.
    score_rows: list[dict] = []
    for r in rows:
        cid = int(r["concept_id"])
        content_class = r["content_class"]
        bpb = 2.0 if content_class == "concept_avoiding" else 1.0
        sample_id = make_sample_id(
            concept_id=cid,
            mode=SteeringMode.POSITIVE,
            content_class=content_class,
            prefix_text=r["prefix_text"],
            output_text=r["output_text"],
        )
        score_rows.append(
            {
                "method_name": "m",
                "concept_id": cid,
                "mode": "positive",
                "content_class": content_class,
                "target": "encdec",
                "coefficient_str": "1",
                "sample_id": sample_id,
                "bpb_intervened": bpb,
                "n_output_tokens": 10,
                "n_output_bytes": 40,
                "truncated": False,
            }
        )

    per_tuple = compute_per_tuple_rollup(
        pa.Table.from_pylist(score_rows)
    ).to_pylist()

    assert per_tuple, "aggregation produced no rows"
    positive_rows = [row for row in per_tuple if row["mode"] == "positive"]
    assert positive_rows, "no positive-mode aggregation rows produced"
    for row in positive_rows:
        margin = row["primary_margin"]
        assert margin is not None and math.isfinite(margin), (
            f"positive primary_margin must be finite, got {margin!r} for "
            f"concept_id={row['concept_id']}"
        )
        assert row["n_rich"] > 0 and row["n_avoid"] > 0, (
            "positive group must have both rich and avoid sides scored"
        )
        assert margin == 1.0, f"expected avoid(2.0) - rich(1.0) = 1.0, got {margin}"
