"""Crash, corruption, resume, and ownership tests for the cell-result store."""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import signal
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import steering.eval.axbench_upstream.cell_store as store_module
from steering.eval.axbench_upstream.cell_store import CellResultStore, CellStoreError
from steering.eval.axbench_upstream.work_grid import (
    NL_PROJECT_METHODS,
    CellKey,
    build_encdec_cell_grid,
    build_project_cell_grid,
    owned_encdec_cells,
    owned_project_cells,
)


def _cells() -> tuple[CellKey, ...]:
    frame = pd.DataFrame(
        [
            {"concept_id": 0, "input_id": 0, "factor": 1.0},
            {"concept_id": 1, "input_id": 0, "factor": 1.0},
        ]
    )
    return build_project_cell_grid(frame, seed=42)


def _identity(*, marker: str = "stable") -> dict[str, Any]:
    return {
        "driver": "project",
        "repository_commit": "abc123",
        "input_sha256": "0" * 64,
        "runtime": {"torch": "test", "cuda": "test", "marker": marker},
    }


def _open(
    root: Path,
    *,
    cells: tuple[CellKey, ...] | None = None,
    tripwire: tuple[CellKey, ...] | None = None,
    identity: dict[str, Any] | None = None,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = 2,
) -> CellResultStore:
    expected = _cells() if cells is None else cells
    first = expected[:2] if tripwire is None else tripwire
    return CellResultStore(
        root=root,
        cells=expected,
        tripwire_cells=first,
        semantic_identity=_identity() if identity is None else identity,
        rank=rank,
        world_size=world_size,
        batch_size=batch_size,
    )


def _batch_path(root: Path, *, rank: int = 0, batch: int = 0) -> Path:
    return root / f"rank_{rank:05d}" / f"batch_{batch:06d}.parquet"


def _write_rows(path: Path, rows: list[dict[str, Any]], *, schema: pa.Schema | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=schema or store_module._RESULT_SCHEMA)
    pq.write_table(table, path)


def _row(cell: CellKey, generation: str) -> dict[str, Any]:
    return {**cell.to_dict(), "generation": generation}


def _complete(root: Path) -> tuple[CellKey, ...]:
    cells = _cells()
    with _open(root, batch_size=1) as store:
        for index, cell in enumerate(cells[:2]):
            assert store.append(cell, f"generation-{index}")
        store.mark_tripwire_passed()
        for index, cell in enumerate(cells[2:], start=2):
            assert store.append(cell, f"generation-{index}")
    return cells


def _sigkill_after_one_durable_batch(root: str) -> None:
    with _open(Path(root), batch_size=1) as store:
        store.append(_cells()[0], "durable-before-sigkill")
        os.kill(os.getpid(), signal.SIGKILL)


def test_new_store_persists_grid_bound_semantic_identity(tmp_path: Path) -> None:
    root = tmp_path / "resume"

    with _open(root) as store:
        fingerprint = store.semantic_fingerprint

    payload = json.loads((root / "identity.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["algorithm"] == "rank-striped-cell-store-v1"
    assert payload["semantic_identity"] == _identity()
    assert payload["logical_cell_count"] == 4
    assert len(payload["logical_cell_grid_sha256"]) == 64
    assert payload["semantic_fingerprint"] == fingerprint
    assert "rank" not in payload
    assert "world_size" not in payload


def test_identity_bytes_are_stable_across_rank_and_world_size(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root, rank=0, world_size=1) as first:
        fingerprint = first.semantic_fingerprint
    before = (root / "identity.json").read_bytes()

    with _open(root, rank=2, world_size=3) as resumed:
        assert resumed.semantic_fingerprint == fingerprint

    assert (root / "identity.json").read_bytes() == before


def test_semantic_identity_drift_fails_without_overwrite(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    before = (root / "identity.json").read_bytes()

    with pytest.raises(CellStoreError, match="differs from the current result"):
        _open(root, identity=_identity(marker="changed"))

    assert (root / "identity.json").read_bytes() == before


def test_grid_drift_is_part_of_identity_even_before_any_result(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, cells=cells):
        pass

    with pytest.raises(CellStoreError, match="differs from the current result"):
        _open(root, cells=cells[:-1], tripwire=cells[:2])


@pytest.mark.parametrize("world_size", [0, -1, True, False, 1.5, "2", None])
def test_store_rejects_invalid_world_size(tmp_path: Path, world_size: object) -> None:
    with pytest.raises(CellStoreError, match="world_size must be a positive int excluding bool"):
        _open(tmp_path / "resume", world_size=cast(Any, world_size))


@pytest.mark.parametrize(
    ("rank", "world_size"),
    [(-1, 2), (2, 2), (True, 2), (False, 2), (0.0, 2), ("0", 2), (None, 2)],
)
def test_store_rejects_invalid_rank(tmp_path: Path, rank: object, world_size: int) -> None:
    with pytest.raises(CellStoreError, match=r"rank must be an int excluding bool in \[0, 2\)"):
        _open(tmp_path / "resume", rank=cast(Any, rank), world_size=world_size)


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "2", None])
def test_store_rejects_invalid_batch_size(tmp_path: Path, batch_size: object) -> None:
    with pytest.raises(CellStoreError, match="batch_size must be a positive int excluding bool"):
        _open(tmp_path / "resume", batch_size=cast(Any, batch_size))


def test_store_rejects_empty_expected_grid(tmp_path: Path) -> None:
    with pytest.raises(CellStoreError, match="requires at least one expected cell"):
        _open(tmp_path / "resume", cells=(), tripwire=())


def test_store_rejects_duplicate_expected_cells(tmp_path: Path) -> None:
    cells = _cells()
    with pytest.raises(CellStoreError, match="expected cell grid contains duplicate"):
        _open(tmp_path / "resume", cells=(cells[0], cells[0]), tripwire=(cells[0],))


def test_store_rejects_empty_tripwire_population(tmp_path: Path) -> None:
    with pytest.raises(CellStoreError, match="tripwire cell population must not be empty"):
        _open(tmp_path / "resume", tripwire=())


def test_store_rejects_duplicate_tripwire_cells(tmp_path: Path) -> None:
    cells = _cells()
    with pytest.raises(CellStoreError, match="tripwire cell population contains duplicate"):
        _open(tmp_path / "resume", tripwire=(cells[0], cells[0]))


def test_store_rejects_out_of_grid_tripwire_cell(tmp_path: Path) -> None:
    cells = _cells()
    unknown = replace(cells[0], source_row_position=999)

    with pytest.raises(CellStoreError, match="tripwire population contains out-of-grid cells"):
        _open(tmp_path / "resume", tripwire=(unknown,))


def test_append_flushes_exact_batches_and_reconstructs_canonical_order(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, batch_size=2) as store:
        assert store.completed_count == 0
        assert store.append(cells[1], "one")
        assert not _batch_path(root).exists()
        assert store.append(cells[0], "zero")
        assert _batch_path(root).is_file()
        store.mark_tripwire_passed()
        assert store.append(cells[3], "three")
        assert store.append(cells[2], "two")
        assert store.complete_results() == {
            cells[0]: "zero",
            cells[1]: "one",
            cells[2]: "two",
            cells[3]: "three",
        }

    assert _batch_path(root, batch=1).is_file()
    table = pq.read_table(_batch_path(root))
    assert table.schema.equals(store_module._RESULT_SCHEMA, check_metadata=True)
    assert table.to_pylist() == [_row(cells[1], "one"), _row(cells[0], "zero")]


def test_context_exit_flushes_partial_batch_during_exception(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()

    with pytest.raises(RuntimeError, match="intentional interruption"):
        with _open(root, batch_size=32) as store:
            store.append(cells[0], "durable-on-exit")
            raise RuntimeError("intentional interruption")

    with _open(root, batch_size=32) as resumed:
        assert resumed.completed_count == 1
        assert resumed.missing(cells[:2]) == (cells[1],)


def test_sigkill_restart_recovers_the_last_atomically_published_batch(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    process = multiprocessing.get_context("spawn").Process(
        target=_sigkill_after_one_durable_batch,
        args=(str(root),),
    )
    process.start()
    process.join(timeout=30)

    assert not process.is_alive()
    assert process.exitcode == -signal.SIGKILL
    with _open(root) as resumed:
        assert resumed.completed_count == 1
        assert resumed.results_for((_cells()[0],)) == {_cells()[0]: "durable-before-sigkill"}


def test_world_size_change_reuses_completed_logical_cells(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _complete(root)

    with _open(root, rank=4, world_size=5) as resumed:
        assert resumed.missing() == ()
        assert list(resumed.complete_results()) == list(cells)


def test_encdec_group_ownership_partial_resume_reuses_cells_and_reconstructs_canonically(
    tmp_path: Path,
) -> None:
    root = tmp_path / "encdec-resume"
    frame = pd.DataFrame(
        [
            {"concept_id": concept_id, "input_id": 0, "factor": factor}
            for concept_id in range(3)
            for factor in (0.0, 1.0)
        ]
    )
    cells = build_encdec_cell_grid(frame, seed=42)
    first_pair = tuple(cell for cell in cells if cell.concept_id == 0)
    identity = {**_identity(), "driver": "encdec"}
    old_owned = [owned_encdec_cells(cells, rank=rank, world_size=2) for rank in range(2)]

    with _open(
        root,
        cells=cells,
        tripwire=first_pair,
        identity=identity,
        rank=0,
        world_size=2,
        batch_size=1,
    ) as first:
        for cell in first_pair:
            first.append(cell, f"generation-{cells.index(cell)}")
        first.mark_tripwire_passed()
        for cell in old_owned[0]:
            first.append(cell, f"generation-{cells.index(cell)}")

    completed = set(old_owned[0])
    with _open(
        root,
        cells=cells,
        tripwire=first_pair,
        identity=identity,
        rank=1,
        world_size=2,
        batch_size=1,
    ) as partial:
        for cell in old_owned[1][:2]:
            partial.append(cell, f"generation-{cells.index(cell)}")
            completed.add(cell)

    missing_counts: list[int] = []
    for rank in range(3):
        newly_owned = owned_encdec_cells(cells, rank=rank, world_size=3)
        expected_missing = tuple(cell for cell in newly_owned if cell not in completed)
        with _open(
            root,
            cells=cells,
            tripwire=first_pair,
            identity=identity,
            rank=rank,
            world_size=3,
            batch_size=1,
        ) as resumed:
            assert resumed.missing(newly_owned) == expected_missing
            missing_counts.append(len(expected_missing))
            for cell in expected_missing:
                resumed.append(cell, f"generation-{cells.index(cell)}")
                completed.add(cell)

    assert missing_counts == [0, 4, 0]
    with _open(
        root,
        cells=cells,
        tripwire=first_pair,
        identity=identity,
        world_size=1,
    ) as complete:
        assert complete.complete_results() == {
            cell: f"generation-{index}" for index, cell in enumerate(cells)
        }


def test_project_group_ownership_partial_resume_keeps_singleton_and_canonical_results(
    tmp_path: Path,
) -> None:
    root = tmp_path / "project-group-resume"
    frame = pd.DataFrame(
        [
            {"concept_id": concept_id, "input_id": 0, "factor": factor}
            for concept_id in range(3)
            for factor in (0.0, 1.0)
        ]
    )
    cells = build_project_cell_grid(frame, seed=42)
    first_pair = tuple(cell for cell in cells if cell.concept_id == 0)
    old_owned = [owned_project_cells(cells, rank=rank, world_size=2) for rank in range(2)]

    with _open(
        root,
        cells=cells,
        tripwire=first_pair,
        rank=0,
        world_size=2,
        batch_size=1,
    ) as first:
        for cell in old_owned[0]:
            first.append(cell, f"generation-{cells.index(cell)}")
        first.mark_tripwire_passed()

    completed = set(old_owned[0])
    with _open(
        root,
        cells=cells,
        tripwire=first_pair,
        rank=1,
        world_size=2,
        batch_size=1,
    ) as partial:
        for cell in old_owned[1][:-1]:
            partial.append(cell, f"generation-{cells.index(cell)}")
            completed.add(cell)

    missing_counts: list[int] = []
    for rank in range(3):
        newly_owned = owned_project_cells(cells, rank=rank, world_size=3)
        expected_missing = tuple(cell for cell in newly_owned if cell not in completed)
        with _open(
            root,
            cells=cells,
            tripwire=first_pair,
            rank=rank,
            world_size=3,
            batch_size=1,
        ) as resumed:
            assert resumed.missing(newly_owned) == expected_missing
            missing_counts.append(len(expected_missing))
            for cell in expected_missing:
                resumed.append(cell, f"generation-{cells.index(cell)}")
                completed.add(cell)

    assert missing_counts == [0, 1, 0]
    with _open(root, cells=cells, tripwire=first_pair, world_size=1) as complete:
        assert complete.complete_results() == {
            cell: f"generation-{index}" for index, cell in enumerate(cells)
        }


def test_nl_project_store_resumes_across_world_sizes_with_exact_grid(tmp_path: Path) -> None:
    root = tmp_path / "nl-project-resume"
    frame = pd.DataFrame(
        [
            {"concept_id": concept_id, "input_id": input_id, "factor": factor}
            for concept_id in range(2)
            for input_id in range(3)
            for factor in (0.0, 1.0)
        ]
    )
    cells = build_project_cell_grid(frame, seed=9, methods=NL_PROJECT_METHODS)
    first_concept = tuple(cell for cell in cells if cell.concept_id == 0)
    completed: set[CellKey] = set()
    for rank in range(2):
        owned = owned_project_cells(
            cells,
            rank=rank,
            world_size=2,
            methods=NL_PROJECT_METHODS,
        )
        with _open(
            root,
            cells=cells,
            tripwire=first_concept,
            rank=rank,
            world_size=2,
            batch_size=1,
        ) as store:
            for cell in owned:
                if cell.concept_id == 0:
                    store.append(cell, f"generation-{cells.index(cell)}")
                    completed.add(cell)

    with _open(
        root,
        cells=cells,
        tripwire=first_concept,
        rank=0,
        world_size=2,
        batch_size=1,
    ) as store:
        store.mark_tripwire_passed()
        first_bulk = next(
            cell
            for cell in owned_project_cells(
                cells,
                rank=0,
                world_size=2,
                methods=NL_PROJECT_METHODS,
            )
            if cell.concept_id == 1
        )
        store.append(first_bulk, f"generation-{cells.index(first_bulk)}")
        completed.add(first_bulk)

    for rank in range(4):
        owned = owned_project_cells(
            cells,
            rank=rank,
            world_size=4,
            methods=NL_PROJECT_METHODS,
        )
        with _open(
            root,
            cells=cells,
            tripwire=first_concept,
            rank=rank,
            world_size=4,
            batch_size=1,
        ) as store:
            expected = tuple(cell for cell in owned if cell not in completed)
            assert store.missing(owned) == expected
            for cell in expected:
                store.append(cell, f"generation-{cells.index(cell)}")
                completed.add(cell)

    with _open(root, cells=cells, tripwire=first_concept) as store:
        assert store.complete_results() == {
            cell: f"generation-{index}" for index, cell in enumerate(cells)
        }


def test_two_rank_writers_can_commit_disjoint_batches(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    rank_zero = _open(root, rank=0, world_size=2, batch_size=1)
    rank_one = _open(root, rank=1, world_size=2, batch_size=1)
    rank_zero.append(cells[0], "zero")
    rank_one.append(cells[1], "one")
    rank_zero.close()
    rank_one.close()

    with _open(root, rank=0, world_size=1) as resumed:
        assert resumed.completed_count == 2
        assert resumed.missing(cells[:2]) == ()


def test_exact_duplicate_rows_across_ranks_are_deduplicated(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    first = _open(root, rank=0, world_size=2, batch_size=1)
    stale_peer = _open(root, rank=1, world_size=2, batch_size=1)
    assert first.append(cells[0], "same")
    assert stale_peer.append(cells[0], "same")
    first.close()
    stale_peer.close()

    with _open(root) as resumed:
        assert resumed.completed_count == 1
        assert resumed.missing(cells[:2]) == (cells[1],)


def test_conflicting_duplicate_rows_across_ranks_fail_on_resume(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    first = _open(root, rank=0, world_size=2, batch_size=1)
    stale_peer = _open(root, rank=1, world_size=2, batch_size=1)
    first.append(cells[0], "left")
    stale_peer.append(cells[0], "right")
    first.close()
    stale_peer.close()

    with pytest.raises(CellStoreError, match="conflicting generations"):
        _open(root)


def test_append_exact_replay_is_a_noop_and_conflict_fails(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cell = _cells()[0]
    with _open(root) as store:
        assert store.append(cell, "same")
        assert not store.append(cell, "same")
        with pytest.raises(CellStoreError, match="already has a conflicting generation"):
            store.append(cell, "different")


def test_append_rejects_out_of_grid_cell(tmp_path: Path) -> None:
    cells = _cells()
    unknown = replace(cells[0], input_id=999)
    with _open(tmp_path / "resume") as store:
        with pytest.raises(CellStoreError, match="cannot append out-of-grid cell"):
            store.append(unknown, "generation")


@pytest.mark.parametrize("generation", [None, 1, b"text", ["text"]])
def test_append_rejects_nonstr_generation(tmp_path: Path, generation: object) -> None:
    with _open(tmp_path / "resume") as store:
        with pytest.raises(CellStoreError, match="generation must be a string"):
            store.append(_cells()[0], cast(Any, generation))


def test_closed_store_rejects_append_and_flush(tmp_path: Path) -> None:
    store = _open(tmp_path / "resume")
    store.close()
    store.close()

    with pytest.raises(CellStoreError, match="cannot append to a closed cell store"):
        store.append(_cells()[0], "generation")
    with pytest.raises(CellStoreError, match="cannot flush a closed cell store"):
        store.flush()


def test_empty_flush_is_a_noop(tmp_path: Path) -> None:
    with _open(tmp_path / "resume") as store:
        store.flush()
        assert not _batch_path(tmp_path / "resume").exists()


def test_missing_rejects_out_of_grid_query(tmp_path: Path) -> None:
    cells = _cells()
    unknown = replace(cells[0], concept_id=999)
    with _open(tmp_path / "resume") as store:
        with pytest.raises(CellStoreError, match=r"missing\(\) received out-of-grid cells"):
            store.missing((unknown,))


def test_complete_results_reports_first_missing_cell(tmp_path: Path) -> None:
    with _open(tmp_path / "resume") as store:
        with pytest.raises(CellStoreError, match="4 of 4 cells missing; first=CellKey"):
            store.complete_results()


def test_tripwire_cannot_be_marked_before_every_first_cell(tmp_path: Path) -> None:
    cells = _cells()
    with _open(tmp_path / "resume") as store:
        store.append(cells[0], "only-one")
        with pytest.raises(CellStoreError, match="1 missing cells"):
            store.mark_tripwire_passed()


def test_tripwire_receipt_is_immutable_and_replayable(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, batch_size=1) as store:
        store.append(cells[0], "zero")
        store.append(cells[1], "one")
        store.mark_tripwire_passed()
        receipt = (root / "tripwire.json").read_bytes()
        store.mark_tripwire_passed()
        store.require_tripwire_passed()
        assert (root / "tripwire.json").read_bytes() == receipt

    with _open(root) as resumed:
        resumed.require_tripwire_passed()


def test_mark_tripwire_flushes_supporting_rows_before_receipt(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, batch_size=32) as store:
        store.append(cells[0], "zero")
        store.append(cells[1], "one")
        assert not _batch_path(root).exists()
        store.mark_tripwire_passed()
        assert _batch_path(root).is_file()
        assert (root / "tripwire.json").is_file()

    with _open(root, batch_size=32) as resumed:
        resumed.require_tripwire_passed()
        assert resumed.results_for(cells[:2]) == {cells[0]: "zero", cells[1]: "one"}


def test_require_tripwire_rejects_missing_receipt(tmp_path: Path) -> None:
    with _open(tmp_path / "resume") as store:
        with pytest.raises(CellStoreError, match="tripwire receipt is missing"):
            store.require_tripwire_passed()


def test_tampered_tripwire_receipt_fails_on_resume(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, batch_size=1) as store:
        store.append(cells[0], "zero")
        store.append(cells[1], "one")
        store.mark_tripwire_passed()
    payload = json.loads((root / "tripwire.json").read_text(encoding="utf-8"))
    payload["cell_count"] = 999
    (root / "tripwire.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CellStoreError, match="does not match the current run identity/grid"):
        _open(root)


@pytest.mark.parametrize(
    "invalid_suffix",
    [
        ',"cell_count":2}',
        ',"unexpected":NaN}',
    ],
)
def test_tripwire_receipt_rejects_duplicate_keys_and_nonfinite_json(
    tmp_path: Path,
    invalid_suffix: str,
) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, batch_size=1) as store:
        store.append(cells[0], "zero")
        store.append(cells[1], "one")
        store.mark_tripwire_passed()
    receipt = root / "tripwire.json"
    original = receipt.read_text(encoding="utf-8").rstrip()
    receipt.write_text(original[:-1] + invalid_suffix, encoding="utf-8")

    with pytest.raises(CellStoreError, match="tripwire receipt is unreadable"):
        _open(root)


def test_receipt_without_complete_tripwire_population_fails_on_resume(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root, batch_size=1) as store:
        store.append(cells[0], "zero")
        store.append(cells[1], "one")
        store.mark_tripwire_passed()
    _batch_path(root, batch=1).unlink()

    with pytest.raises(CellStoreError, match="cell population is incomplete"):
        _open(root)


def test_bulk_batch_without_tripwire_receipt_is_invalid(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    _complete(root)
    (root / "tripwire.json").unlink()

    with pytest.raises(CellStoreError, match="durable bulk cells without a tripwire receipt"):
        _open(root)


def test_unknown_root_file_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    (root / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(CellStoreError, match="resume store contains unknown file"):
        _open(root)


def test_unknown_root_directory_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    (root / "scratch").mkdir()

    with pytest.raises(CellStoreError, match="resume store contains unknown directory"):
        _open(root)


def test_unknown_rank_file_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    rank_dir = root / "rank_00000"
    (rank_dir / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(CellStoreError, match="rank directory contains unknown file"):
        _open(root)


def test_nonfile_rank_entry_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    (root / "rank_00000" / "batch_000000.parquet").mkdir()

    with pytest.raises(CellStoreError, match="rank directory contains non-file entry"):
        _open(root)


def test_symlink_in_store_root_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    target = tmp_path / "target"
    target.write_text("target", encoding="utf-8")
    (root / "notes.txt").symlink_to(target)

    with pytest.raises(CellStoreError, match="resume store contains symlink"):
        _open(root)


def test_symlink_in_rank_directory_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    target = tmp_path / "target.parquet"
    target.write_text("target", encoding="utf-8")
    (root / "rank_00000" / "batch_000000.parquet").symlink_to(target)

    with pytest.raises(CellStoreError, match="rank directory contains symlink"):
        _open(root)


def test_noncontiguous_batch_indices_fail_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root):
        pass
    _write_rows(_batch_path(root, batch=1), [_row(cells[0], "generation")])

    with pytest.raises(CellStoreError, match="non-contiguous batch indices"):
        _open(root)


def test_corrupt_parquet_batch_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    _batch_path(root).write_bytes(b"not parquet")

    with pytest.raises(CellStoreError, match="could not read resume batch"):
        _open(root)


def test_empty_parquet_batch_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    with _open(root):
        pass
    _write_rows(_batch_path(root), [])

    with pytest.raises(CellStoreError, match="resume batch must not be empty"):
        _open(root)


def test_schema_metadata_drift_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root):
        pass
    schema = store_module._RESULT_SCHEMA.with_metadata({b"format": b"wrong"})
    _write_rows(_batch_path(root), [_row(cells[0], "generation")], schema=schema)

    with pytest.raises(CellStoreError, match="resume batch schema mismatch"):
        _open(root)


def test_schema_column_drift_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root):
        pass
    row = _row(cells[0], "generation")
    del row["selection_seed"]
    _write_rows(_batch_path(root), [row], schema=pa.Table.from_pylist([row]).schema)

    with pytest.raises(CellStoreError, match="resume batch schema mismatch"):
        _open(root)


def test_out_of_grid_batch_row_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    cells = _cells()
    with _open(root):
        pass
    row = _row(cells[0], "generation")
    row["input_id"] = 999
    _write_rows(_batch_path(root), [row])

    with pytest.raises(CellStoreError, match="out-of-grid cell"):
        _open(root)


def test_malformed_driver_in_batch_row_fails_loudly(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    row = _row(_cells()[0], "generation")
    with _open(root):
        pass
    row["driver"] = "wrong"
    _write_rows(_batch_path(root), [row])

    with pytest.raises(CellStoreError, match="invalid driver"):
        _open(root)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_row_position", -1),
        ("concept_id", -1),
        ("input_id", -1),
        ("selection_seed", -1),
        ("token_seed", -1),
    ],
)
def test_negative_integer_identity_in_batch_row_fails_loudly(
    tmp_path: Path,
    field: str,
    value: int,
) -> None:
    root = tmp_path / "resume"
    row = _row(_cells()[0], "generation")
    with _open(root):
        pass
    row[field] = value
    _write_rows(_batch_path(root), [row])

    with pytest.raises(CellStoreError, match=f"field {field} must be a nonnegative int"):
        _open(root)


def test_generation_null_is_rejected_by_exact_schema_write(tmp_path: Path) -> None:
    root = tmp_path / "resume"
    row = _row(_cells()[0], "generation")
    row["generation"] = None
    with _open(root):
        pass

    with pytest.raises((TypeError, ValueError, pa.ArrowInvalid)):
        _write_rows(_batch_path(root), [row])


def test_completed_store_can_be_copied_and_resumed_byte_for_byte(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    cells = _complete(source)
    shutil.copytree(source, target)

    with _open(source) as reference, _open(target) as candidate:
        assert candidate.complete_results() == reference.complete_results()
        assert list(candidate.complete_results()) == list(cells)
