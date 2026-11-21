"""Tests for merge_per_rank_checkpoints (PR-4 Task 4)."""

from __future__ import annotations

import json as _json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.stage0c_partition import (
    _RANK_CHECKPOINT_SCHEMA_VERSION,
    merge_per_rank_checkpoints,
    read_per_rank_checkpoint,
    validate_per_rank_fingerprint,
)
from steering.eval.axbench_base.stage0c_partition import (
    write_per_rank_checkpoint as _write_per_rank_checkpoint,
)
from steering.eval.axbench_base.types import AxBenchBaseError, QcOutcome

_FP = {
    "feature_dir": "/x",
    "stage0c_multi_gpu_block_size": 4,
    "extraction_tokenizer_id": "google/gemma-2-2b",
    "enable_concept_examples_qc": False,
    "n_verified_prefixes_per_concept_target": 1,
}


def write_per_rank_checkpoint(
    json_path: Path,
    parquet_path: Path,
    payload: dict,
    rows: list[dict],
    *,
    extraction_tokenizer_id: str,
) -> None:
    """Write one schema-current fixture with explicit dynamic state."""
    complete_payload = {
        "stopped_by_qc_survivor_cap": False,
        **payload,
    }
    _write_per_rank_checkpoint(
        json_path,
        parquet_path,
        complete_payload,
        rows,
        extraction_tokenizer_id=extraction_tokenizer_id,
    )


def test_rank_checkpoint_schema_version_is_eight_for_prefill_identity() -> None:
    assert _RANK_CHECKPOINT_SCHEMA_VERSION == 8


def _accept(idx: int) -> dict:
    """Build a complete rank_accepts entry with all Concept fields."""
    return {
        "candidate_index": idx,
        "feature_index": 100 + idx,
        "layer": 8,
        "sae_source": "8-gemmascope-res-16k",
        "description": f"f{idx}",
        "corpus_count": 100,
        "description_source": "neuronpedia",
        "qc_score": None,
        "verified_rows_count": 1,
    }


def _row(idx: int) -> dict:
    return {
        "concept_id": -1,  # provisional; merge assigns canonical
        "prefix_text": f"p{idx}",
        "prefix_source_shard": 0,
        "prefix_source_doc": idx,
        "prefix_activating_seg_idx": 0,
        "verification_continuations": ["c"],
        "verification_seeds": [idx],
        "verification_percentiles": [80.0],
        "candidate_index": idx,
    }


def _qc(idx: int) -> dict:
    return {
        "feature_index": 100 + idx,
        "description": f"f{idx}",
        "outcome": QcOutcome.ACCEPTED.value,
        "score": 2,
        "n_examples": 1,
        "error_message": None,
    }


def test_merge_assigns_canonical_concept_ids_in_candidate_index_order(tmp_path: Path):
    """Accepts spread across ranks merge to dense concept_ids 0..N-1
    sorted by candidate_index."""
    # Rank 0 accepts at indices 0, 4
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 5, "fingerprint": _FP,
         "next_block_index": 1, "visited_candidate_indices": [0, 4],
         "rank_accepts": [_accept(0), _accept(4)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [_row(0), _row(4)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    # Rank 1 accepts at indices 1, 3
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_1.json",
        tmp_path / "stage0c_partial_rank_1.parquet",
        {"rank": 1, "world_size": 2, "n_candidates": 5, "fingerprint": _FP,
         "next_block_index": 1, "visited_candidate_indices": [1, 3],
         "rank_accepts": [_accept(1), _accept(3)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [_row(1), _row(3)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    merged = merge_per_rank_checkpoints(
        run_dir=tmp_path, world_size=2, max_concepts=4,
    )
    accepted = merged["accepted_concepts"]
    rows = merged["rows"]

    # Sorted by candidate_index: 0, 1, 3, 4 → concept_ids 0, 1, 2, 3
    assert [c.concept_id for c in accepted] == [0, 1, 2, 3]
    assert [c.feature_index for c in accepted] == [100, 101, 103, 104]

    # Rows mapped to canonical concept_ids and sorted by concept_id
    expected_concept_id_by_text = {"p0": 0, "p1": 1, "p3": 2, "p4": 3}
    for r in rows:
        assert r["concept_id"] == expected_concept_id_by_text[r["prefix_text"]]
        assert "candidate_index" not in r  # stripped


def test_merge_truncates_to_max_concepts(tmp_path: Path):
    """When more accepts than max_concepts, take first max_concepts after sort."""
    for rank, indices in [(0, [0, 2, 4]), (1, [1, 3, 5])]:
        write_per_rank_checkpoint(
            tmp_path / f"stage0c_partial_rank_{rank}.json",
            tmp_path / f"stage0c_partial_rank_{rank}.parquet",
            {"rank": rank, "world_size": 2, "n_candidates": 6,
             "fingerprint": _FP,
             "next_block_index": 0, "visited_candidate_indices": indices,
             "rank_accepts": [_accept(i) for i in indices],
             "rank_rejects": [], "rank_qc_per_feature": []},
            [_row(i) for i in indices],
            extraction_tokenizer_id="google/gemma-2-2b",
        )

    merged = merge_per_rank_checkpoints(
        run_dir=tmp_path, world_size=2, max_concepts=3,
    )
    # Sorted: 0,1,2,3,4,5 → first 3 are 0,1,2
    assert [c.concept_id for c in merged["accepted_concepts"]] == [0, 1, 2]
    assert [c.feature_index for c in merged["accepted_concepts"]] == [100, 101, 102]


def test_merge_canonicalizes_qc_order_and_filters_overvisits(tmp_path: Path):
    qc_fingerprint = {
        **_FP,
        "enable_concept_examples_qc": True,
        "concept_examples_qc_config": {
            "activation_example_judge_min_score": 1,
            "concept_examples_qc_target_surviving": None,
        },
    }
    for rank, indices in ((0, [0, 2]), (1, [1, 3])):
        accepts = [{**_accept(idx), "qc_score": 2} for idx in indices]
        write_per_rank_checkpoint(
            tmp_path / f"stage0c_partial_rank_{rank}.json",
            tmp_path / f"stage0c_partial_rank_{rank}.parquet",
            {
                "rank": rank,
                "world_size": 2,
                "n_candidates": 4,
                "fingerprint": qc_fingerprint,
                "next_block_index": 1,
                "visited_candidate_indices": indices,
                "rank_accepts": accepts,
                "rank_rejects": [],
                "rank_qc_per_feature": [_qc(idx) for idx in indices],
            },
            [_row(idx) for idx in indices],
            extraction_tokenizer_id="google/gemma-2-2b",
        )

    merged = merge_per_rank_checkpoints(
        run_dir=tmp_path,
        world_size=2,
        max_concepts=2,
    )

    assert [row["feature_index"] for row in merged["qc_per_feature"]] == [
        100,
        101,
    ]


def test_merge_validates_fingerprint_consistency_across_ranks(tmp_path: Path):
    """Different fingerprints across ranks raises (corruption indicator)."""
    fp_a = {**_FP, "feature_dir": "/x"}
    fp_b = {**_FP, "feature_dir": "/y"}
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 0,
         "fingerprint": fp_a,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_1.json",
        tmp_path / "stage0c_partial_rank_1.parquet",
        {"rank": 1, "world_size": 2, "n_candidates": 0,
         "fingerprint": fp_b,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="fingerprint"):
        merge_per_rank_checkpoints(run_dir=tmp_path, world_size=2, max_concepts=2)


def test_merge_rejects_sorted_deterministically(tmp_path: Path):
    """Rejects across multiple ranks merge into a deterministic order
    (sorted by candidate_index)."""
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 5, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0, 4],
         "rank_accepts": [],
         "rank_rejects": [
             {"candidate_index": 0, "feature_index": 100, "description": "f0",
              "reason": "not_enough_verified_prefixes",
              "n_verified_found": 0, "llm_judge_score": None},
             {"candidate_index": 4, "feature_index": 104, "description": "f4",
              "reason": "not_enough_verified_prefixes",
              "n_verified_found": 0, "llm_judge_score": None},
         ],
         "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_1.json",
        tmp_path / "stage0c_partial_rank_1.parquet",
        {"rank": 1, "world_size": 2, "n_candidates": 5, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [1, 3],
         "rank_accepts": [],
         "rank_rejects": [
             {"candidate_index": 1, "feature_index": 101, "description": "f1",
              "reason": "not_enough_verified_prefixes",
              "n_verified_found": 0, "llm_judge_score": None},
             {"candidate_index": 3, "feature_index": 103, "description": "f3",
              "reason": "not_enough_verified_prefixes",
              "n_verified_found": 0, "llm_judge_score": None},
         ],
         "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )

    merged = merge_per_rank_checkpoints(
        run_dir=tmp_path, world_size=2, max_concepts=2,
    )
    indices = [r["candidate_index"] for r in merged["rejected_concepts"]]
    assert indices == [0, 1, 3, 4]  # sorted


def test_merge_handles_empty_rank_with_explicit_commit_marker(tmp_path: Path):
    """An empty stripe still contributes a JSON + typed empty parquet."""
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [_row(0)], extraction_tokenizer_id="google/gemma-2-2b",
    )
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_1.json",
        tmp_path / "stage0c_partial_rank_1.parquet",
        {"rank": 1, "world_size": 2, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )

    merged = merge_per_rank_checkpoints(
        run_dir=tmp_path, world_size=2, max_concepts=1,
    )
    assert len(merged["accepted_concepts"]) == 1
    assert merged["accepted_concepts"][0].concept_id == 0


def test_merge_rejects_missing_current_rank_json(tmp_path: Path):
    """A missing current-rank commit marker is incomplete state, not emptiness."""
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [_row(0)], extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="missing required.*rank_1.json"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=2, max_concepts=1,
        )


def test_merge_raises_when_no_files(tmp_path: Path):
    """No per-rank checkpoint files at all → clear error."""
    with pytest.raises(AxBenchBaseError, match="no per-rank checkpoint files"):
        merge_per_rank_checkpoints(run_dir=tmp_path, world_size=2, max_concepts=1)


def test_per_rank_checkpoint_round_trip(tmp_path: Path):
    """Write + read returns identical payload + rows."""
    payload = {
        "rank": 0, "world_size": 2, "n_candidates": 1,
        "fingerprint": {
            **_FP,
            "stage0c_multi_gpu_block_size": 8,
        },
        "next_block_index": 3,
        "visited_candidate_indices": [0],
        "rank_accepts": [_accept(0)],
        "rank_rejects": [],
        "rank_qc_per_feature": [],
        "stopped_by_qc_survivor_cap": False,
    }
    rows: list[dict] = [_row(0)]

    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        payload, rows, extraction_tokenizer_id="google/gemma-2-2b",
    )
    read_payload, read_rows = read_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
    )
    assert read_payload == payload
    # Rows match (parquet round-trip may reorder dict keys, but values match)
    assert len(read_rows) == 1
    for k, v in rows[0].items():
        assert read_rows[0][k] == v, f"Field {k} mismatch: {read_rows[0][k]} vs {v}"


def test_empty_checkpoint_atomically_overwrites_orphan_parquet(
    tmp_path: Path,
):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    parquet_path = tmp_path / "stage0c_partial_rank_0.parquet"
    parquet_path.write_bytes(b"stale orphan from parquet-first crash")

    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        {"rank": 0, "world_size": 1, "n_candidates": 0, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    table = pq.read_table(parquet_path)
    assert table.num_rows == 0
    assert "candidate_index" in table.schema.names
    assert json_path.exists()


def test_merge_rejects_per_accept_row_count_mismatch(tmp_path: Path):
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="row-count mismatch"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


@pytest.mark.parametrize("stored_count", [0, 2])
def test_merge_rejects_accept_count_that_differs_from_fingerprint_target(
    tmp_path: Path,
    stored_count: int,
):
    accept = {**_accept(0), "verified_rows_count": stored_count}
    rows = [
        {**_row(0), "prefix_source_doc": row_idx}
        for row_idx in range(stored_count)
    ]
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [0], "rank_accepts": [accept],
         "rank_rejects": [], "rank_qc_per_feature": []},
        rows,
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(
        AxBenchBaseError,
        match="verified_rows_count=.*must equal fingerprinted",
    ):
        merge_per_rank_checkpoints(
            run_dir=tmp_path,
            world_size=1,
            max_concepts=1,
        )


def test_merge_rejects_row_for_unrecorded_accept(tmp_path: Path):
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 2, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [_row(1)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="not recorded exactly once"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_accept_reject_overlap(tmp_path: Path):
    reject = {
        "candidate_index": 0,
        "feature_index": 100,
        "description": "f0",
        "reason": "not_enough_verified_prefixes",
        "n_verified_found": 0,
        "llm_judge_score": None,
    }
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)],
         "rank_rejects": [reject], "rank_qc_per_feature": []},
        [_row(0)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="both accepted and rejected"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


@pytest.mark.parametrize("schema_version", [None, 7])
def test_merge_rejects_missing_or_stale_schema_version(
    tmp_path: Path,
    schema_version: int | None,
):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    write_per_rank_checkpoint(
        json_path,
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 0, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    payload = _json.loads(json_path.read_text(encoding="utf-8"))
    if schema_version is None:
        payload.pop("schema_version")
    else:
        payload["schema_version"] = schema_version
    json_path.write_text(_json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="stale schema_version"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_non_object_fingerprint(tmp_path: Path):
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 0, "fingerprint": [],
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="fingerprint object"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_json_commit_without_parquet(tmp_path: Path):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    parquet_path = tmp_path / "stage0c_partial_rank_0.parquet"
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        {"rank": 0, "world_size": 1, "n_candidates": 0,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    parquet_path.unlink()

    with pytest.raises(AxBenchBaseError, match="no matching parquet"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_wrong_rank_parquet_schema(tmp_path: Path):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    parquet_path = tmp_path / "stage0c_partial_rank_0.parquet"
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        {"rank": 0, "world_size": 1, "n_candidates": 0,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    pq.write_table(
        pa.table({"candidate_index": pa.array([], type=pa.int64())}),
        parquet_path,
    )

    with pytest.raises(AxBenchBaseError, match="has schema"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


@pytest.mark.parametrize(
    ("metadata_value", "error_pattern"),
    [
        (None, "missing required schema metadata"),
        (b"wrong/tokenizer", "was written for extraction_tokenizer_id"),
    ],
)
def test_merge_rejects_rank_parquet_tokenizer_metadata_corruption(
    tmp_path: Path,
    metadata_value: bytes | None,
    error_pattern: str,
):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    parquet_path = tmp_path / "stage0c_partial_rank_0.parquet"
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        {"rank": 0, "world_size": 1, "n_candidates": 0,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    table = pq.read_table(parquet_path)
    metadata = dict(table.schema.metadata or {})
    metadata_key = b"extraction_tokenizer_id"
    if metadata_value is None:
        metadata.pop(metadata_key, None)
    else:
        metadata[metadata_key] = metadata_value
    pq.write_table(table.replace_schema_metadata(metadata), parquet_path)

    with pytest.raises(AxBenchBaseError, match=error_pattern):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_visited_candidate_without_terminal_outcome(
    tmp_path: Path,
):
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [0], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="one terminal accept/reject"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_candidate_index_outside_declared_pool(tmp_path: Path):
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [1], "rank_accepts": [_accept(1)],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [_row(1)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="outside candidate pool"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


@pytest.mark.parametrize(
    ("field", "value", "error_pattern"),
    [
        ("rank", False, "strict int rank"),
        ("world_size", True, "strict current world_size"),
        ("next_block_index", -1, "next_block_index"),
        ("stopped_by_qc_survivor_cap", "false", "non-boolean"),
    ],
)
def test_merge_rejects_invalid_dynamic_rank_state(
    tmp_path: Path,
    field: str,
    value: object,
    error_pattern: str,
):
    payload = {
        "rank": 0,
        "world_size": 1,
        "n_candidates": 0,
        "fingerprint": _FP,
        "next_block_index": 0,
        "visited_candidate_indices": [],
        "rank_accepts": [],
        "rank_rejects": [],
        "rank_qc_per_feature": [],
        field: value,
    }
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        payload,
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match=error_pattern):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_merge_rejects_qc_cap_stop_flag_when_qc_is_disabled(tmp_path: Path):
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 0,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": [],
         "stopped_by_qc_survivor_cap": True},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="while examples QC is disabled"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path,
            world_size=1,
            max_concepts=1,
        )


def test_merge_rejects_qc_cap_stop_flag_with_wrong_survivor_count(
    tmp_path: Path,
):
    fingerprint = {
        **_FP,
        "enable_concept_examples_qc": True,
        "concept_examples_qc_config": {
            "activation_example_judge_min_score": 1,
            "concept_examples_qc_target_surviving": 2,
        },
    }
    accept = {**_accept(0), "qc_score": 2}
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1,
         "fingerprint": fingerprint, "next_block_index": 0,
         "visited_candidate_indices": [0], "rank_accepts": [accept],
         "rank_rejects": [], "rank_qc_per_feature": [_qc(0)],
         "stopped_by_qc_survivor_cap": True},
        [_row(0)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="terminal QC survivors number 1"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path,
            world_size=1,
            max_concepts=1,
        )


def test_merge_rejects_qc_cap_stop_flag_under_multi_rank(tmp_path: Path):
    fingerprint = {
        **_FP,
        "enable_concept_examples_qc": True,
        "concept_examples_qc_config": {
            "activation_example_judge_min_score": 1,
            "concept_examples_qc_target_surviving": 0,
        },
    }
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 0,
         "fingerprint": fingerprint, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": [],
         "stopped_by_qc_survivor_cap": True},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="supported only at world_size=1"):
        merge_per_rank_checkpoints(
            run_dir=tmp_path,
            world_size=2,
            max_concepts=1,
        )


@pytest.mark.parametrize(
    ("record_update", "error_pattern"),
    [
        ({"unexpected": True}, "has fields"),
        ({"n_verified_found": 1}, "requires 0 <= n_verified_found"),
        ({
            "reason": "llm_judge_below_min_score",
            "n_verified_found": None,
        }, "QC-only reason"),
    ],
)
def test_merge_rejects_semantically_invalid_rejection_state(
    tmp_path: Path,
    record_update: dict,
    error_pattern: str,
):
    reject = {
        "candidate_index": 0,
        "feature_index": 100,
        "description": "f0",
        "reason": "not_enough_verified_prefixes",
        "n_verified_found": 0,
        "llm_judge_score": None,
    }
    reject.update(record_update)
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1,
         "fingerprint": _FP, "next_block_index": 0,
         "visited_candidate_indices": [0], "rank_accepts": [],
         "rank_rejects": [reject], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match=error_pattern):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


@pytest.mark.parametrize(
    ("qc_records", "error_pattern"),
    [
        ([], "exactly one QC record"),
        ([{"feature_index": 100}], "has fields"),
        ([{
            "feature_index": 100,
            "description": "wrong description",
            "outcome": "accepted",
            "score": 2,
            "n_examples": 1,
            "error_message": None,
        }], "description mismatch"),
        ([{
            "feature_index": 100,
            "description": "f0",
            "outcome": "rejected",
            "score": 0,
            "n_examples": 1,
            "error_message": None,
        }], "incompatible with terminal outcome"),
    ],
)
def test_merge_rejects_corrupt_qc_state(
    tmp_path: Path,
    qc_records: list[dict],
    error_pattern: str,
):
    fingerprint = {
        **_FP,
        "enable_concept_examples_qc": True,
        "concept_examples_qc_config": {
            "activation_example_judge_min_score": 1,
            "concept_examples_qc_target_surviving": None,
        },
    }
    accept = _accept(0)
    accept["qc_score"] = 2
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1,
         "fingerprint": fingerprint, "next_block_index": 0,
         "visited_candidate_indices": [0], "rank_accepts": [accept],
         "rank_rejects": [], "rank_qc_per_feature": qc_records},
        [_row(0)],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match=error_pattern):
        merge_per_rank_checkpoints(
            run_dir=tmp_path, world_size=1, max_concepts=1,
        )


def test_validate_per_rank_fingerprint_mismatch_raises(tmp_path: Path):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    write_per_rank_checkpoint(
        json_path,
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 0, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )
    bad_fp = {
        **_FP,
        "feature_dir": "/y",
        "stage0c_multi_gpu_block_size": 8,
    }
    with pytest.raises(AxBenchBaseError, match="fingerprint mismatch"):
        validate_per_rank_fingerprint(json_path, bad_fp)


@pytest.mark.parametrize(
    ("stored_mode", "expected_mode"),
    [("shared", "repeated"), ("repeated", "shared")],
)
def test_validate_per_rank_fingerprint_rejects_cross_prefill_mode_resume(
    tmp_path: Path,
    stored_mode: str,
    expected_mode: str,
) -> None:
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    stored_fingerprint = {
        **_FP,
        "verification_prefill_mode": stored_mode,
    }
    write_per_rank_checkpoint(
        json_path,
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 0,
         "fingerprint": stored_fingerprint, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )
    expected_fingerprint = {
        **_FP,
        "verification_prefill_mode": expected_mode,
    }
    with pytest.raises(
        AxBenchBaseError,
        match=r"verification_prefill_mode",
    ):
        validate_per_rank_fingerprint(json_path, expected_fingerprint)


@pytest.mark.parametrize("stored_value", [True, 1.0])
def test_validate_per_rank_fingerprint_uses_strict_json_types(
    tmp_path: Path,
    stored_value: object,
):
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    stored_fingerprint = {
        **_FP,
        "n_verified_prefixes_per_concept_target": stored_value,
    }
    write_per_rank_checkpoint(
        json_path,
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 0,
         "fingerprint": stored_fingerprint, "next_block_index": 0,
         "visited_candidate_indices": [], "rank_accepts": [],
         "rank_rejects": [], "rank_qc_per_feature": []},
        [],
        extraction_tokenizer_id="google/gemma-2-2b",
    )

    with pytest.raises(AxBenchBaseError, match="fingerprint mismatch"):
        validate_per_rank_fingerprint(json_path, _FP)


def test_validate_per_rank_fingerprint_field_set_mismatch(tmp_path: Path):
    """Different field SET (missing/extra keys) raises distinct error."""
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    write_per_rank_checkpoint(
        json_path,
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 2, "n_candidates": 0, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [],
         "rank_accepts": [], "rank_rejects": [], "rank_qc_per_feature": []},
        [], extraction_tokenizer_id="google/gemma-2-2b",
    )
    extra_fp = dict(_FP)
    extra_fp["new_field"] = "extra"
    with pytest.raises(AxBenchBaseError, match="field set"):
        validate_per_rank_fingerprint(json_path, extra_fp)


# ---------------------------------------------------------------------------
# PR-4 second-round review: Tier 2 high-value tests
# ---------------------------------------------------------------------------


def test_validate_per_rank_fingerprint_missing_fingerprint_field(
    tmp_path: Path,
):
    """A per-rank JSON without a ``fingerprint`` key at all (e.g.,
    written by a buggy runner that forgot to populate it) must fail
    validation with the canonical "missing a fingerprint" error so an
    operator sees the cause without grep-scanning logs."""
    json_path = tmp_path / "stage0c_partial_rank_0.json"
    bogus_payload = {
        "schema_version": _RANK_CHECKPOINT_SCHEMA_VERSION,
        # No fingerprint key here.
        "rank": 0, "world_size": 1, "n_candidates": 0,
        "next_block_index": 0, "visited_candidate_indices": [],
        "rank_accepts": [], "rank_rejects": [],
        "rank_qc_per_feature": [],
    }
    json_path.write_text(_json.dumps(bogus_payload), encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="missing a fingerprint"):
        validate_per_rank_fingerprint(json_path, _FP)


def test_per_rank_parquet_carries_tokenizer_metadata(tmp_path: Path):
    """The per-rank parquet schema metadata must carry the
    ``extraction_tokenizer_id`` so downstream consumers can detect a
    tokenizer mismatch before reading rows."""
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)], "rank_rejects": [],
         "rank_qc_per_feature": []},
        [_row(0)], extraction_tokenizer_id="google/gemma-2-2b",
    )
    schema = pq.read_schema(tmp_path / "stage0c_partial_rank_0.parquet")
    md = schema.metadata or {}
    # pyarrow stores metadata as bytes, so we look up the canonical key
    # name and decode.
    meta_key = b"extraction_tokenizer_id"
    assert meta_key in md, f"tokenizer metadata missing; got keys={list(md)!r}"
    assert md[meta_key].decode("utf-8") == "google/gemma-2-2b"


def test_per_rank_parquet_schema_includes_candidate_index_column(tmp_path: Path):
    """The per-rank parquet's schema must extend the canonical Stage 0c
    schema with a ``candidate_index`` column so the merge can sort
    rows by candidate-index without falling back to a side index."""
    write_per_rank_checkpoint(
        tmp_path / "stage0c_partial_rank_0.json",
        tmp_path / "stage0c_partial_rank_0.parquet",
        {"rank": 0, "world_size": 1, "n_candidates": 1, "fingerprint": _FP,
         "next_block_index": 0, "visited_candidate_indices": [0],
         "rank_accepts": [_accept(0)], "rank_rejects": [],
         "rank_qc_per_feature": []},
        [_row(0)], extraction_tokenizer_id="google/gemma-2-2b",
    )
    schema = pq.read_schema(tmp_path / "stage0c_partial_rank_0.parquet")
    columns = set(schema.names)
    legacy_columns = {
        "concept_id", "prefix_text", "prefix_source_shard",
        "prefix_source_doc", "prefix_activating_seg_idx",
        "verification_continuations", "verification_seeds",
        "verification_percentiles",
    }
    assert legacy_columns.issubset(columns), (
        f"per-rank parquet missing legacy columns: "
        f"{legacy_columns - columns}"
    )
    assert "candidate_index" in columns, (
        f"per-rank parquet must carry candidate_index; got {columns!r}"
    )
