"""Unit tests for hash-based work partitioning + work-unit enumeration."""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from steering.eval.axbench_bpb.bpb_scoring import enumerate_work_units, is_mine
from steering.eval.axbench_bpb.types import SteeringMode, WorkUnit


def test_is_mine_stable_across_calls():
    """Same (unit, rank, world_size) produces the same answer deterministically."""
    wu = WorkUnit("m", 5, "sid", "0.5")
    for _ in range(10):
        assert is_mine(wu, rank=0, world_size=8) == is_mine(wu, rank=0, world_size=8)


def test_is_mine_partitions_exhaustively():
    """Every unit is owned by exactly one rank — across various world sizes."""
    units = [WorkUnit(f"m{i % 3}", i, f"sid{i}", "0") for i in range(100)]
    for ws in (1, 2, 4, 8):
        counts = [0] * ws
        for wu in units:
            owners = [r for r in range(ws) if is_mine(wu, rank=r, world_size=ws)]
            assert len(owners) == 1, (
                f"unit {wu} owned by {len(owners)} ranks at ws={ws}"
            )
            counts[owners[0]] += 1
        assert sum(counts) == len(units)


def test_is_mine_approximately_uniform():
    """SHA256 distribution should be within 20% of expected per-rank count."""
    units = [
        WorkUnit(f"m{i % 5}", i, f"sid{i}", f"c{i % 7}")
        for i in range(10_000)
    ]
    ws = 8
    counts = [0] * ws
    for wu in units:
        for r in range(ws):
            if is_mine(wu, rank=r, world_size=ws):
                counts[r] += 1
                break
    expected = len(units) / ws
    for c in counts:
        assert abs(c - expected) < 0.2 * expected, (
            f"rank count {c} outside 20% of expected {expected}"
        )


def test_is_mine_world_size_1_owns_all():
    """world_size=1 → rank 0 owns every unit."""
    units = [WorkUnit(f"m{i}", i, f"s{i}", str(i)) for i in range(50)]
    for wu in units:
        assert is_mine(wu, rank=0, world_size=1) is True


def test_is_mine_different_coefficient_different_ranks_possible():
    """Changing coefficient_str can change ownership (payload is part of hash)."""
    base = WorkUnit("m", 1, "sid", "0.0")
    perturbed = WorkUnit("m", 1, "sid", "1.0")
    owners_base = [r for r in range(8) if is_mine(base, rank=r, world_size=8)]
    owners_perturbed = [r for r in range(8) if is_mine(perturbed, rank=r, world_size=8)]
    # They COULD be the same; but since payload differs, hash-mod-8 could differ.
    # The invariant we assert is that each is owned by exactly one rank:
    assert len(owners_base) == 1
    assert len(owners_perturbed) == 1


@dataclass(frozen=True)
class _FakeMethod:
    name: str
    mode: SteeringMode


@dataclass(frozen=True)
class _FakeConcept:
    concept_id: int


@dataclass(frozen=True)
class _FakeSample:
    sample_id: str


def test_enumerate_work_units_full_product():
    """2 methods × 2 concepts × 2 samples × 3 coefficients = 24 units."""
    methods = [
        _FakeMethod(name="m1", mode=SteeringMode.POSITIVE),
        _FakeMethod(name="m2", mode=SteeringMode.POSITIVE),
    ]
    concepts = [_FakeConcept(concept_id=i) for i in (10, 20)]
    samples_by_concept_mode = {
        (10, SteeringMode.POSITIVE): [_FakeSample("s1"), _FakeSample("s2")],
        (20, SteeringMode.POSITIVE): [_FakeSample("s3"), _FakeSample("s4")],
    }
    coefficient_grids = {"m1": [0, 1, 2], "m2": [0.5, 1.5, 2.5]}
    units = list(enumerate_work_units(
        methods=methods, concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
    ))
    assert len(units) == 24
    # Each unit is unique (method_name, concept_id, sample_id, coefficient_str)
    assert len({(u.method_name, u.concept_id, u.sample_id, u.coefficient_str) for u in units}) == 24


def test_enumerate_work_units_mixed_modes_only_emits_matching():
    """A method with mode=NEGATIVE only emits units for samples keyed by (cid, NEGATIVE)."""
    methods = [_FakeMethod(name="m_neg", mode=SteeringMode.NEGATIVE)]
    concepts = [_FakeConcept(concept_id=5)]
    samples_by_concept_mode = {
        (5, SteeringMode.POSITIVE): [_FakeSample("pos_s1")],
        (5, SteeringMode.NEGATIVE): [_FakeSample("neg_s1"), _FakeSample("neg_s2")],
    }
    coefficient_grids = {"m_neg": [1.0]}
    units = list(enumerate_work_units(
        methods=methods, concepts=concepts,
        samples_by_concept_mode=samples_by_concept_mode,
        coefficient_grids=coefficient_grids,
    ))
    assert len(units) == 2
    assert {u.sample_id for u in units} == {"neg_s1", "neg_s2"}


def test_enumerate_work_units_empty_samples_skips():
    """Missing (concept, mode) key produces no units (uses .get(..., []) default)."""
    methods = [_FakeMethod(name="m", mode=SteeringMode.POSITIVE)]
    concepts = [_FakeConcept(concept_id=99)]  # no entry in dict
    coefficient_grids = {"m": [0]}
    units = list(enumerate_work_units(
        methods=methods, concepts=concepts,
        samples_by_concept_mode={},
        coefficient_grids=coefficient_grids,
    ))
    assert units == []


def test_enumerate_work_units_missing_method_in_grid_raises():
    """If a method isn't in coefficient_grids, KeyError propagates — loud failure."""
    methods = [_FakeMethod(name="m_missing", mode=SteeringMode.POSITIVE)]
    concepts = [_FakeConcept(concept_id=1)]
    samples_by_concept_mode = {(1, SteeringMode.POSITIVE): [_FakeSample("s")]}
    coefficient_grids = {}  # m_missing NOT in grid
    with pytest.raises(KeyError, match="m_missing"):
        list(enumerate_work_units(
            methods=methods, concepts=concepts,
            samples_by_concept_mode=samples_by_concept_mode,
            coefficient_grids=coefficient_grids,
        ))
