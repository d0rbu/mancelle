"""Differential oracle for the array-backed splice-sidecar representation."""

from __future__ import annotations

from array import array
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
    SpliceRef,
)
from tests.steering._splice_cf_fixtures import build_derived_env
from tests.steering.test_splice_provenance_index import _sidecar, _write


def _indices(
    tmp_path: Path,
    sidecar: dict[str, Any],
    *,
    capacity: int = 1,
) -> tuple[SpliceProvenanceIndex, SpliceProvenanceIndex]:
    root = _write(tmp_path, sidecar)
    return (
        SpliceProvenanceIndex(root, max_resident_shards=capacity),
        SpliceProvenanceIndex(root, max_resident_shards=capacity, compact=True),
    )


def _mru(index: SpliceProvenanceIndex) -> Any:
    return index._resident[next(reversed(index._resident))]


def _compact_array_bytes(index: SpliceProvenanceIndex) -> int:
    return sum(
        len(value) * value.itemsize
        for resident in index._resident.values()
        for value in vars(resident).values()
        if isinstance(value, array)
    )


def _public_snapshot(
    index: SpliceProvenanceIndex,
    stem: str,
) -> tuple[int, int, tuple[tuple[int, tuple[tuple[int, SpliceRef], ...]], ...]]:
    n_docs = index.shard_n_docs(stem)
    n_splices = index.shard_n_splice_positions(stem)
    docs = tuple(
        (
            index.doc_n_segments(stem, doc_idx),
            tuple(index.doc_splices(stem, doc_idx).items()),
        )
        for doc_idx in range(n_docs)
    )
    return n_docs, n_splices, docs


def _capture_failure(call: Callable[[], object]) -> BaseException:
    try:
        call()
    except BaseException as exc:
        return exc
    raise AssertionError("expected call to fail")


def _large_valid_sidecar(n_docs: int, splice_every: int) -> dict[str, Any]:
    docs: list[dict[str, Any]] = []
    n_splices = 0
    for doc_idx in range(n_docs):
        splices: list[dict[str, Any]] = []
        if doc_idx % splice_every == 0:
            splices.append(
                {
                    "pos": 1,
                    "whf_base": [doc_idx % 7, doc_idx, 1],
                    "whf_derived": [(doc_idx + 1) % n_docs, 0],
                }
            )
            n_splices += 1
        docs.append(
            {
                "doc_id": doc_idx,
                "n_segments": 3,
                "splices": splices,
            }
        )
    return {
        "schema_version": 2,
        "params": {"n_splice_positions": n_splices},
        "n_docs": n_docs,
        "docs": docs,
    }


@pytest.mark.parametrize(
    "sidecar",
    [
        pytest.param(_sidecar(), id="one-splice"),
        pytest.param(_sidecar(multi_splice=True), id="multi-splice"),
        pytest.param(_sidecar(with_splice=False), id="zero-splice"),
        pytest.param(_sidecar(n_docs=17, multi_splice=True), id="seventeen-docs"),
    ],
)
def test_compact_public_values_and_order_equal_reference(
    tmp_path: Path,
    sidecar: dict[str, Any],
) -> None:
    reference, compact = _indices(tmp_path, sidecar)

    assert _public_snapshot(compact, "shard_00000") == _public_snapshot(
        reference, "shard_00000"
    )
    reference_stats = reference.cache_stats.snapshot()
    compact_stats = compact.cache_stats.snapshot()
    reference_stats.pop("load_seconds")
    compact_stats.pop("load_seconds")
    assert compact_stats == reference_stats


def test_compact_preserves_nonempty_mapping_identity_and_mutability(
    tmp_path: Path,
) -> None:
    _reference, compact = _indices(tmp_path, _sidecar(multi_splice=True))
    first = compact.doc_splices("shard_00000", 0)
    second = compact.doc_splices("shard_00000", 0)
    injected = SpliceRef(pos=99, whf_base=(1, 2, 3), whf_derived=(1, 0))

    assert first is second
    first[99] = injected
    assert compact.doc_splices("shard_00000", 0)[99] is injected


def test_reference_nonempty_mapping_has_same_identity_and_mutation_contract(
    tmp_path: Path,
) -> None:
    reference, _compact = _indices(tmp_path, _sidecar(multi_splice=True))
    first = reference.doc_splices("shard_00000", 0)
    second = reference.doc_splices("shard_00000", 0)
    injected = SpliceRef(pos=99, whf_base=(1, 2, 3), whf_derived=(1, 0))

    assert first is second
    first[99] = injected
    assert reference.doc_splices("shard_00000", 0)[99] is injected


def test_compact_preserves_fresh_empty_mapping_contract(tmp_path: Path) -> None:
    reference, compact = _indices(tmp_path, _sidecar())
    reference_first = reference.doc_splices("shard_00000", 1)
    reference_second = reference.doc_splices("shard_00000", 1)
    compact_first = compact.doc_splices("shard_00000", 1)
    compact_second = compact.doc_splices("shard_00000", 1)

    assert reference_first == compact_first == {}
    assert reference_first is not reference_second
    assert compact_first is not compact_second


def test_compact_splice_refs_remain_frozen(tmp_path: Path) -> None:
    _reference, compact = _indices(tmp_path, _sidecar())
    ref = compact.doc_splices("shard_00000", 0)[2]

    with pytest.raises(FrozenInstanceError):
        ref.pos = 1  # ty: ignore[invalid-assignment] -- frozen contract


def test_compact_materializes_only_requested_nonempty_docs(tmp_path: Path) -> None:
    sidecar = _large_valid_sidecar(n_docs=100, splice_every=5)
    _reference, compact = _indices(tmp_path, sidecar)
    assert compact.shard_n_docs("shard_00000") == 100
    resident = _mru(compact)
    assert hasattr(resident, "materialized_docs")
    assert resident.materialized_docs == {}

    assert compact.doc_splices("shard_00000", 1) == {}
    assert resident.materialized_docs == {}
    assert list(compact.doc_splices("shard_00000", 10)) == [1]
    assert set(resident.materialized_docs) == {10}
    assert list(compact.doc_splices("shard_00000", 25)) == [1]
    assert set(resident.materialized_docs) == {10, 25}


@pytest.mark.parametrize(
    ("n_docs", "splice_every"),
    [(1, 1), (10, 2), (101, 7), (10_000, 5)],
)
def test_compact_array_byte_accounting_is_exact(
    tmp_path: Path,
    n_docs: int,
    splice_every: int,
) -> None:
    sidecar = _large_valid_sidecar(n_docs=n_docs, splice_every=splice_every)
    _reference, compact = _indices(tmp_path, sidecar)
    n_splices = int(sidecar["params"]["n_splice_positions"])
    compact.shard_n_docs("shard_00000")

    expected_bytes = (n_docs + (n_docs + 1) + 6 * n_splices) * 8
    assert _compact_array_bytes(compact) == expected_bytes
    assert _compact_array_bytes(compact) > 0


def test_compact_large_sidecar_payload_is_smaller_than_json_bytes(
    tmp_path: Path,
) -> None:
    sidecar = _large_valid_sidecar(n_docs=10_000, splice_every=4)
    _reference, compact = _indices(tmp_path, sidecar)
    json_path = tmp_path / "_splice_provenance" / "shard_00000.json"
    compact.shard_n_docs("shard_00000")

    assert _compact_array_bytes(compact) < json_path.stat().st_size
    assert _compact_array_bytes(compact) < 500_000
    cached = _mru(compact)
    assert cached.materialized_docs == {}


def test_compact_array_bytes_tracks_lru_admission_and_eviction(tmp_path: Path) -> None:
    first = _large_valid_sidecar(n_docs=10, splice_every=2)
    second = _large_valid_sidecar(n_docs=20, splice_every=4)
    third = _large_valid_sidecar(n_docs=30, splice_every=5)
    root = _write(tmp_path, first, stem="shard_00000")
    _write(tmp_path, second, stem="shard_00001")
    _write(tmp_path, third, stem="shard_00002")
    compact = SpliceProvenanceIndex(root, max_resident_shards=2, compact=True)

    compact.shard_n_docs("shard_00000")
    first_bytes = _compact_array_bytes(compact)
    compact.shard_n_docs("shard_00001")
    first_two_bytes = _compact_array_bytes(compact)
    compact.shard_n_docs("shard_00002")
    last_two_bytes = _compact_array_bytes(compact)

    assert first_bytes == (10 + 11 + 6 * 5) * 8
    assert first_two_bytes == first_bytes + (20 + 21 + 6 * 5) * 8
    assert last_two_bytes == (20 + 21 + 6 * 5 + 30 + 31 + 6 * 6) * 8
    assert compact.cache_stats.evictions == 1


def test_compact_and_reference_lru_sequences_are_identical(tmp_path: Path) -> None:
    root = _write(tmp_path, _large_valid_sidecar(7, 2), stem="shard_00000")
    _write(tmp_path, _large_valid_sidecar(8, 3), stem="shard_00001")
    _write(tmp_path, _large_valid_sidecar(9, 4), stem="shard_00002")
    reference = SpliceProvenanceIndex(root, max_resident_shards=2)
    compact = SpliceProvenanceIndex(root, max_resident_shards=2, compact=True)
    sequence = [
        "shard_00000",
        "shard_00001",
        "shard_00000",
        "shard_00002",
        "shard_00001",
        "shard_00002",
    ]

    for stem in sequence:
        assert _public_snapshot(compact, stem) == _public_snapshot(reference, stem)
        assert list(compact._resident)[-1] == list(reference._resident)[-1]
        for field in ("loads", "hits", "evictions", "retain_calls", "bytes_read"):
            assert getattr(compact.cache_stats, field) == getattr(
                reference.cache_stats, field
            )


def test_compact_retain_and_drop_match_reference(tmp_path: Path) -> None:
    root = _write(tmp_path, _large_valid_sidecar(7, 2), stem="shard_00000")
    _write(tmp_path, _large_valid_sidecar(8, 3), stem="shard_00001")
    _write(tmp_path, _large_valid_sidecar(9, 4), stem="shard_00002")
    reference = SpliceProvenanceIndex(root, max_resident_shards=2)
    compact = SpliceProvenanceIndex(root, max_resident_shards=2, compact=True)

    for index in (reference, compact):
        index.retain(["shard_00001", "shard_00000", "shard_00001"])
        index.drop_except(["shard_00000"])
        index.ensure_resident("shard_00002")

    assert list(compact._resident) == list(reference._resident)
    for field in ("loads", "hits", "evictions", "retain_calls", "bytes_read"):
        assert getattr(compact.cache_stats, field) == getattr(
            reference.cache_stats, field
        )


@pytest.mark.parametrize(
    "sidecar",
    [
        pytest.param(_sidecar(version=1), id="old-schema"),
        pytest.param(_sidecar(drop_splices_key=True), id="missing-splices"),
        pytest.param(_sidecar(bad_whf_doc=True), id="bad-ref-doc"),
        pytest.param(_sidecar(bad_whf_pos=True), id="bad-ref-position"),
        pytest.param(_sidecar(bad_pos=0), id="zero-position"),
        pytest.param(_sidecar(bad_pos=3), id="position-at-end"),
        pytest.param(_sidecar(wrong_total=True), id="wrong-total"),
        pytest.param(_sidecar(duplicate_pos=True), id="duplicate-position"),
        pytest.param({"schema_version": 2, "docs": []}, id="missing-n-docs"),
        pytest.param(
            {"schema_version": 2, "n_docs": 0, "docs": [], "params": []},
            id="params-list",
        ),
    ],
)
def test_compact_validation_exception_is_exactly_reference(
    tmp_path: Path,
    sidecar: dict[str, Any],
) -> None:
    reference, compact = _indices(tmp_path, sidecar)
    reference_error = _capture_failure(
        lambda: reference.shard_n_docs("shard_00000")
    )
    compact_error = _capture_failure(lambda: compact.shard_n_docs("shard_00000"))

    assert type(compact_error) is type(reference_error)
    assert str(compact_error) == str(reference_error)
    assert compact_error.__cause__ is reference_error.__cause__ is None
    assert compact._resident == reference._resident == {}


def test_compact_missing_file_exception_is_exactly_reference(tmp_path: Path) -> None:
    root = tmp_path / "_splice_provenance"
    root.mkdir()
    reference = SpliceProvenanceIndex(tmp_path)
    compact = SpliceProvenanceIndex(tmp_path, compact=True)
    reference_error = _capture_failure(lambda: reference.shard_n_docs("missing"))
    compact_error = _capture_failure(lambda: compact.shard_n_docs("missing"))

    assert type(compact_error) is type(reference_error) is SpliceProvenanceError
    assert str(compact_error) == str(reference_error)


def test_compact_out_of_range_exception_is_exactly_reference(tmp_path: Path) -> None:
    reference, compact = _indices(tmp_path, _sidecar())
    for method_name in ("doc_n_segments", "doc_splices"):
        reference_method = getattr(reference, method_name)
        compact_method = getattr(compact, method_name)
        reference_error = _capture_failure(
            lambda method=reference_method: method("shard_00000", 99)
        )
        compact_error = _capture_failure(
            lambda method=compact_method: method("shard_00000", 99)
        )
        assert type(compact_error) is type(reference_error)
        assert str(compact_error) == str(reference_error)


def test_compact_falls_back_for_valid_arbitrary_precision_whf_base(
    tmp_path: Path,
) -> None:
    sidecar = _sidecar()
    huge = 10**100
    sidecar["docs"][0]["splices"][0]["whf_base"][0] = huge
    reference, compact = _indices(tmp_path, sidecar)

    assert _public_snapshot(compact, "shard_00000") == _public_snapshot(
        reference, "shard_00000"
    )
    assert compact.doc_splices("shard_00000", 0)[2].whf_base[0] == huge
    assert _compact_array_bytes(compact) == 0
    assert hasattr(_mru(compact), "splices_by_doc")


def test_compact_rejects_non_bool_mode(tmp_path: Path) -> None:
    _write(tmp_path, _sidecar())
    for invalid in (0, 1, None, "true"):
        with pytest.raises(ValueError, match="compact must be a bool"):
            SpliceProvenanceIndex(
                tmp_path,
                compact=invalid,  # ty: ignore[invalid-argument-type] -- runtime boundary
            )


def test_compact_real_splicer_artifacts_equal_reference(tmp_path: Path) -> None:
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    feature_dir = Path(env["feature_dir"])
    reference = SpliceProvenanceIndex(feature_dir, max_resident_shards=2)
    compact = SpliceProvenanceIndex(
        feature_dir,
        max_resident_shards=2,
        compact=True,
    )

    for stem in ("shard_00000", "shard_00001"):
        assert _public_snapshot(compact, stem) == _public_snapshot(reference, stem)


def test_compact_preserves_sidecar_splice_insertion_order(tmp_path: Path) -> None:
    sidecar = _sidecar(multi_splice=True)
    assert [item["pos"] for item in sidecar["docs"][0]["splices"]] == [2, 1]
    reference, compact = _indices(tmp_path, sidecar)

    assert list(reference.doc_splices("shard_00000", 0)) == [2, 1]
    assert list(compact.doc_splices("shard_00000", 0)) == [2, 1]


def test_compact_cache_stats_bytes_count_source_json_not_arrays(tmp_path: Path) -> None:
    sidecar = _large_valid_sidecar(1_000, 3)
    _reference, compact = _indices(tmp_path, sidecar)
    sidecar_path = tmp_path / "_splice_provenance" / "shard_00000.json"
    compact.shard_n_docs("shard_00000")
    first = compact.cache_stats.snapshot()
    compact.shard_n_docs("shard_00000")
    second = compact.cache_stats.snapshot()

    assert first["bytes_read"] == sidecar_path.stat().st_size
    assert second["bytes_read"] == first["bytes_read"]
    assert first["loads"] == second["loads"] == 1
    assert second["hits"] == first["hits"] + 1
