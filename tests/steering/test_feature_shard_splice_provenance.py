"""Tests for the _splice_provenance sidecar structure (issue #590, Task 9;
schema v2 would-have-followed splice records, issue #596).

Covers: params-block schema (cut_fraction key, 8 counters), per-doc record shape,
triples bijection vs the base shard, JSON round-trip, seeding contract, the
new-started-doc property (first triple ordinal > 0), and (schema v2) per-doc
``splices`` records plus the would-have-followed existence-invariant assert.
"""

from __future__ import annotations

import json
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pytest
import torch as t

from nanochat.atomic_io import atomic_write_parquet
from steering.feature_io import glob_features, save_features
from steering.feature_shard_splice import (
    DerivedDoc,
    ShardSpliceResult,
    SourceSegment,
    SpliceInternalError,
    _doc_seg_offsets,
    build_provenance,
    recombination_seed,
    splice_feature_shard,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)
SEG_SIZE = 2
CORPUS = [
    "alpha beta gamma delta",
    "epsilon zeta eta theta",
    "iota kappa lambda mu",
    "nu xi omicron pi rho",
    "sigma tau upsilon phi",
]


def _make_shard_files(tmp_path, texts, *, tok=TOK, seg_size=SEG_SIZE):
    """Build a base shard + parquet and return (feature_path, parquet_path)."""
    shard = make_base_shard(texts, tok, seg_size)
    save_features(shard, tmp_path / "shard_00007")
    parquet_path = tmp_path / "shard_00007.parquet"
    atomic_write_parquet(pa.table({"text": texts}), parquet_path)
    paths = glob_features(tmp_path)
    assert paths, "save_features must produce at least one file"
    return paths[0], parquet_path


def _splice(tmp_path=None, texts=None, *, cut_fraction=0.6, seed=42, shard_index=7):
    """Splice ``texts`` (default ``CORPUS``) and return splice_feature_shard's
    full ``(derived_shard, derived_texts, provenance)`` triple. When ``tmp_path``
    is omitted, a self-contained temporary directory is created and torn down
    internally, so issue #596 tests can call this with only ``cut_fraction=``."""
    if tmp_path is None:
        with tempfile.TemporaryDirectory() as td:
            return _splice(
                Path(td), texts, cut_fraction=cut_fraction, seed=seed, shard_index=shard_index
            )
    if texts is None:
        texts = CORPUS
    shard_path, parquet_path = _make_shard_files(tmp_path, texts)
    return splice_feature_shard(
        shard_path, parquet_path, TOK, model="test-model",
        cut_fraction=cut_fraction, seed=seed, shard_index=shard_index, shard_name="shard_00007",
        segment_size=SEG_SIZE, context_window=2048, sae_max_context=None,
    )


# ---------------------------------------------------------------------------
# Params block: cut_fraction key (not p)
# ---------------------------------------------------------------------------


def test_params_has_cut_fraction_key_not_p(tmp_path):
    """The params block must use 'cut_fraction', not the legacy 'p' key."""
    _, _, prov = _splice(tmp_path, CORPUS)
    assert "cut_fraction" in prov["params"], "params must have 'cut_fraction' key"
    assert "p" not in prov["params"], "params must not retain the legacy 'p' key"


def test_params_cut_fraction_matches_input(tmp_path):
    """The recorded cut_fraction value must equal the input cut_fraction argument."""
    cut_fraction_val = 0.7
    _, _, prov = _splice(tmp_path, CORPUS, cut_fraction=cut_fraction_val)
    assert prov["params"]["cut_fraction"] == cut_fraction_val


# ---------------------------------------------------------------------------
# Params block: 7 engine/load counters
# ---------------------------------------------------------------------------


EXPECTED_COUNTER_KEYS = [
    "n_cuts_final", "n_redrawn", "n_loops", "n_splits",
    "n_trimmed_docs", "n_trimmed_tokens", "n_marker_normalized_docs",
    "n_splice_positions",
]


def test_params_has_all_8_counters(tmp_path):
    """The params block must include all 8 engine + load counters."""
    _, _, prov = _splice(tmp_path, CORPUS)
    params = prov["params"]
    for key in EXPECTED_COUNTER_KEYS:
        assert key in params, f"params missing required counter: {key!r}"


def test_params_counters_are_non_negative_ints(tmp_path):
    """All 8 counter values must be non-negative Python ints."""
    _, _, prov = _splice(tmp_path, CORPUS)
    params = prov["params"]
    for key in EXPECTED_COUNTER_KEYS:
        val = params[key]
        assert isinstance(val, int), f"params[{key!r}] must be int, got {type(val).__name__}"
        assert val >= 0, f"params[{key!r}] must be >= 0, got {val}"


# ---------------------------------------------------------------------------
# Seeding: shard_seed must equal recombination_seed(seed, shard_index, p)
# ---------------------------------------------------------------------------


def test_shard_seed_uses_recombination_seed(tmp_path):
    """The recorded shard_seed must equal recombination_seed(seed, shard_index, cut_fraction)."""
    seed, shard_index, cut_fraction = 42, 7, 0.6
    _, _, prov = _splice(
        tmp_path, CORPUS, cut_fraction=cut_fraction, seed=seed, shard_index=shard_index
    )
    expected = recombination_seed(seed, shard_index, cut_fraction)
    assert prov["shard_seed"] == expected, (
        f"shard_seed={prov['shard_seed']} != recombination_seed={expected}; "
        f"splice_feature_shard must wire recombination_seed(seed, shard_index, cut_fraction)"
    )


# ---------------------------------------------------------------------------
# Per-doc record shape
# ---------------------------------------------------------------------------


def test_per_doc_record_count_matches_n_docs(tmp_path):
    """len(prov['docs']) must equal prov['n_docs']."""
    _, _, prov = _splice(tmp_path, CORPUS)
    assert len(prov["docs"]) == prov["n_docs"]


def test_per_doc_records_have_required_keys(tmp_path):
    """Each doc record must carry the documented set of keys."""
    _, _, prov = _splice(tmp_path, CORPUS)
    required = {"doc_id", "seed_doc_id", "n_segments", "n_switches",
                "provenance", "segment_token_starts", "splices"}
    for doc in prov["docs"]:
        missing = required - set(doc.keys())
        assert not missing, f"doc {doc.get('doc_id')} missing keys: {missing}"


def test_triples_are_length_3(tmp_path):
    """Each entry in a doc's provenance list must be a 3-element list."""
    _, _, prov = _splice(tmp_path, CORPUS)
    for doc in prov["docs"]:
        for i, triple in enumerate(doc["provenance"]):
            assert len(triple) == 3, (
                f"doc {doc['doc_id']} triple[{i}]={triple!r} is not length 3; "
                f"expected [base_shard_idx, doc_idx, seg_ordinal]"
            )


# ---------------------------------------------------------------------------
# Triples bijection vs the base shard
# ---------------------------------------------------------------------------


def test_triples_bijection_vs_base_shard(tmp_path):
    """Every (base_doc_idx, seg_ordinal) pair appears exactly once across all output docs."""
    _, _, prov = _splice(tmp_path, CORPUS)
    shard = make_base_shard(CORPUS, TOK, SEG_SIZE)

    # Build the expected set of (doc_idx, seg_ordinal) pairs from the base shard.
    seg_bounds = _doc_seg_offsets(shard["doc_ids"], int(shard["n_docs"]))
    expected: set[tuple[int, int]] = set()
    for d in range(int(shard["n_docs"])):
        n_segs = seg_bounds[d + 1] - seg_bounds[d]
        for j in range(n_segs):
            expected.add((d, j))

    # Collect all (doc_idx, seg_ordinal) from the provenance triples.
    seen: Counter[tuple[int, int]] = Counter()
    for doc in prov["docs"]:
        for triple in doc["provenance"]:
            seen[(triple[1], triple[2])] += 1

    duplicate = {pair: c for pair, c in seen.items() if c > 1}
    assert not duplicate, f"pairs appear more than once: {duplicate}"

    missing = expected - set(seen.keys())
    extra = set(seen.keys()) - expected
    assert not missing, f"pairs missing from provenance: {missing}"
    assert not extra, f"unexpected pairs in provenance: {extra}"


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------


def test_json_round_trip(tmp_path):
    """Provenance dict must survive json.dumps -> json.loads unchanged."""
    _, _, prov = _splice(tmp_path, CORPUS)
    serialized = json.dumps(prov)
    prov2 = json.loads(serialized)
    assert prov == prov2


# ---------------------------------------------------------------------------
# New-started doc: first triple ordinal > 0
# ---------------------------------------------------------------------------


def test_new_started_doc_first_triple_ordinal_gt_0():
    """A new-started doc's first provenance triple has seg_ordinal > 0.

    Loop-breaking enters the cycle at a stitch target (dp, bp+1) with bp >= 0,
    so the ordinal is always >= 1.  Verified by building a provenance from a
    hand-constructed ShardSpliceResult with a known new-started doc.
    """
    dummy_fids = t.tensor([1, 2], dtype=t.int32)
    dummy_fmags = t.tensor([1.0, 1.5], dtype=t.float32)
    source_segments = [
        [
            SourceSegment(tokens=(1, 2), feature_ids=dummy_fids, feature_mags=dummy_fmags),
            SourceSegment(tokens=(3, 4), feature_ids=dummy_fids, feature_mags=dummy_fmags),
            SourceSegment(tokens=(5, 6), feature_ids=dummy_fids, feature_mags=dummy_fmags),
        ]
    ]
    # new_started doc starts at seg_ordinal=2 (cycle was broken at that stitch entry).
    result = ShardSpliceResult(
        docs=(
            DerivedDoc(segment_refs=((0, 0), (0, 1)), new_started=False),
            DerivedDoc(segment_refs=((0, 2),), new_started=True),
        ),
        n_cuts_final=1, n_redrawn=0, n_loops=1, n_splits=0,
    )
    params = {
        "cut_fraction": 0.5, "seed": 42,
        "n_cuts_final": 1, "n_redrawn": 0, "n_loops": 1, "n_splits": 0,
        "n_trimmed_docs": 0, "n_trimmed_tokens": 0, "n_marker_normalized_docs": 0,
    }

    prov = build_provenance(
        result, source_segments, params=params,
        shard_index=0, shard_name="shard_00000", shard_seed=12345,
    )

    # Doc 1 is new-started; its first provenance triple must have seg_ordinal > 0.
    first_triple = prov["docs"][1]["provenance"][0]
    assert first_triple[2] > 0, (
        f"new-started doc first triple seg_ordinal={first_triple[2]}, expected > 0"
    )


def test_new_started_doc_segment_token_starts_first_is_1():
    """segment_token_starts[0] == 1 for a new-started doc (content-start convention).

    Contract pin for issue #596: build_provenance uses content-token offsets, NOT
    BOS-inclusive positions.  For a new-started doc the synthetic BOS occupies
    position 0 but is not counted in any segment's token tuple; accordingly acc
    is initialised to 1 before iterating over refs (feature_shard_splice.py ~1049):
        acc = 1 if doc.new_started else 0
    This diverges from the shard's segment_positions which IS BOS-inclusive.
    The value 1 is pinned here so that implementers of #596 notice any inadvertent
    change to the convention.
    """
    dummy_fids = t.tensor([1, 2], dtype=t.int32)
    dummy_fmags = t.tensor([1.0, 1.5], dtype=t.float32)
    # new-started doc with a single 3-token segment
    source_segments = [
        [
            SourceSegment(tokens=(10, 20, 30), feature_ids=dummy_fids, feature_mags=dummy_fmags),
        ]
    ]
    result = ShardSpliceResult(
        docs=(
            DerivedDoc(segment_refs=((0, 0),), new_started=True),
        ),
        n_cuts_final=0, n_redrawn=0, n_loops=1, n_splits=0,
    )
    params = {
        "cut_fraction": 0.5, "seed": 42,
        "n_cuts_final": 0, "n_redrawn": 0, "n_loops": 1, "n_splits": 0,
        "n_trimmed_docs": 0, "n_trimmed_tokens": 0, "n_marker_normalized_docs": 0,
    }

    prov = build_provenance(
        result, source_segments, params=params,
        shard_index=0, shard_name="shard_00000", shard_seed=99,
    )

    starts = prov["docs"][0]["segment_token_starts"]
    # The first real-token offset is 1, not 0: the synthetic BOS is position 0
    # but is NOT included in any segment's token tuple (content-start convention).
    # This value is PINNED — do not change without updating docs/pipelines/feature-shard-splice.md
    # and coordinating with issue #596 (counterfactual splice-delta conditioning).
    assert starts[0] == 1, (
        f"segment_token_starts[0]={starts[0]} for new-started doc; "
        f"expected 1 (content-start convention: synthetic BOS not counted)"
    )


# ---------------------------------------------------------------------------
# Issue #596: schema v2 — would-have-followed splice records
# ---------------------------------------------------------------------------


class TestSpliceRecordsV2:
    def test_schema_version_is_2(self) -> None:
        _, _, prov = _splice(cut_fraction=0.6)
        assert prov["schema_version"] == 2

    def test_every_doc_has_splices_key(self) -> None:
        _, _, prov = _splice(cut_fraction=0.6)
        for doc in prov["docs"]:
            assert "splices" in doc
            assert isinstance(doc["splices"], list)

    def test_identity_splice_has_empty_splices_everywhere(self) -> None:
        _, _, prov = _splice(cut_fraction=0.0)
        assert all(doc["splices"] == [] for doc in prov["docs"])
        assert prov["params"]["n_splice_positions"] == 0

    def test_splice_positions_match_non_natural_adjacencies(self) -> None:
        """A record exists at position p iff triples[p] != triples[p-1] + 1."""
        _, _, prov = _splice(cut_fraction=0.7)
        total = 0
        for doc in prov["docs"]:
            triples = doc["provenance"]
            expected = {
                p
                for p in range(1, len(triples))
                if (triples[p][1], triples[p][2])
                != (triples[p - 1][1], triples[p - 1][2] + 1)
            }
            got = {rec["pos"] for rec in doc["splices"]}
            assert got == expected
            total += len(expected)
        assert prov["params"]["n_splice_positions"] == total
        assert total > 0, "cut_fraction=0.7 on CORPUS must produce splices (non-vacuity)"

    def test_whf_base_is_predecessor_plus_one(self) -> None:
        _, _, prov = _splice(cut_fraction=0.7)
        for doc in prov["docs"]:
            triples = doc["provenance"]
            for rec in doc["splices"]:
                p = rec["pos"]
                pred = triples[p - 1]
                assert rec["whf_base"] == [pred[0], pred[1], pred[2] + 1]

    def test_whf_derived_locates_the_whf_base_segment(self) -> None:
        """placement[whf_base] == whf_derived, using only the emitted triples."""
        _, _, prov = _splice(cut_fraction=0.7)
        placement = {}
        for doc in prov["docs"]:
            for p, tri in enumerate(doc["provenance"]):
                placement[(tri[1], tri[2])] = (doc["doc_id"], p)
        checked = 0
        for doc in prov["docs"]:
            for rec in doc["splices"]:
                whf = (rec["whf_base"][1], rec["whf_base"][2])
                assert placement[whf] == tuple(rec["whf_derived"])
                checked += 1
        assert checked > 0

    def test_splices_sorted_by_pos(self) -> None:
        _, _, prov = _splice(cut_fraction=0.7)
        for doc in prov["docs"]:
            positions = [rec["pos"] for rec in doc["splices"]]
            assert positions == sorted(positions)

    def test_new_started_doc_position_zero_never_recorded(self) -> None:
        _, _, prov = _splice(cut_fraction=0.7)
        for doc in prov["docs"]:
            assert all(rec["pos"] >= 1 for rec in doc["splices"])

    def test_json_round_trip_v2(self) -> None:
        _, _, prov = _splice(cut_fraction=0.7)
        assert json.loads(json.dumps(prov)) == prov


class TestExistenceInvariantAssert:
    def test_impossible_whf_raises_splice_internal_error(self) -> None:
        """Hand-built result whose junction predecessor is a LAST segment (no
        successor) must trip the never-happens assert."""
        segs = [
            [SourceSegment(tokens=(500, 97), feature_ids=t.tensor([1], dtype=t.int32),
                           feature_mags=t.tensor([1.0]))
             for _ in range(2)],
            [SourceSegment(tokens=(500, 98), feature_ids=t.tensor([2], dtype=t.int32),
                           feature_mags=t.tensor([1.0]))
             for _ in range(2)],
        ]
        # Junction (0,1) -> (1,1): predecessor (0,1) is doc 0's LAST segment,
        # so whf == (0,2) does not exist. Never produced by the engine.
        bad = ShardSpliceResult(
            docs=(DerivedDoc(((0, 0), (0, 1), (1, 1)), False),
                  DerivedDoc(((1, 0),), False)),
            n_cuts_final=1, n_redrawn=0, n_loops=0, n_splits=0,
        )
        with pytest.raises(SpliceInternalError, match="would-have-followed"):
            build_provenance(bad, segs, params={}, shard_index=7,
                             shard_name="shard_00007", shard_seed=1)
