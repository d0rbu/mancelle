"""Tests for candidate_pool_identity + _fingerprint_with_pool (Task 7, #683)."""

from __future__ import annotations

import pytest

from steering.eval.axbench_base import _stage0c_internals as si
from steering.eval.axbench_base.types import CandidateConcept
from tests.axbench_base.unit.test_config import make_valid_config


def _c(fid, desc="d"):
    return CandidateConcept(
        feature_index=fid, layer=8, sae_source="8-gemmascope-res-16k",
        description=desc, corpus_count=10, description_source="neuronpedia",
    )


@pytest.fixture
def fake_cfg_or_valid():
    return make_valid_config(run_tag="pool_ident")


def test_identity_deterministic():
    pool = [_c(1), _c(2)]
    assert si.candidate_pool_identity(pool) == si.candidate_pool_identity(list(pool))


def test_identity_order_sensitive():
    assert si.candidate_pool_identity([_c(1), _c(2)]) != si.candidate_pool_identity([_c(2), _c(1)])


def test_identity_description_sensitive():
    assert si.candidate_pool_identity([_c(1, "a")]) != si.candidate_pool_identity([_c(1, "b")])


def test_identity_membership_sensitive():
    assert si.candidate_pool_identity([_c(1)]) != si.candidate_pool_identity([_c(1), _c(2)])


def test_fingerprint_with_pool_adds_exactly_one_key(fake_cfg_or_valid):
    cfg = fake_cfg_or_valid
    base = si._compute_checkpoint_fingerprint(cfg)
    withpool = si._fingerprint_with_pool(cfg, candidate_pool_identity="abc")
    assert set(withpool) == set(base) | {"candidate_pool_identity"}
    assert withpool["candidate_pool_identity"] == "abc"
    assert {k: withpool[k] for k in base} == base
