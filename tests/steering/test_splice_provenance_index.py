"""SpliceProvenanceIndex validation-matrix tests (issue #596, D1/D10/D11)."""

from __future__ import annotations

import json

import pytest
from loguru import logger

from nanochat.atomic_io import atomic_json_dump
from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
    SpliceRef,
)
from steering.feature_shard_splice import (
    build_provenance,
    load_source_segments,
    recombination_seed,
    recombine_segments,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer


def _sidecar(n_docs=3, with_splice=True, version=2, drop_splices_key=False,
             bad_whf_doc=False, bad_whf_pos=False, wrong_total=False,
             bad_pos=None, duplicate_pos=False, multi_splice=False):
    docs = []
    for i in range(n_docs):
        rec = {
            "doc_id": i,
            "seed_doc_id": i,
            "n_segments": 3,
            "n_switches": 0,
            "provenance": [[7, i, o] for o in range(3)],
            "segment_token_starts": [0, 2, 4],
            "splices": [],
        }
        docs.append(rec)
    n_total = 0
    if with_splice:
        docs[0]["splices"] = [{
            "pos": bad_pos if bad_pos is not None else 2,
            "whf_base": [7, 0, 2],
            "whf_derived": [n_docs + 5 if bad_whf_doc else 1,
                            9 if bad_whf_pos else 0],
        }]
        n_total = 1
        if duplicate_pos:
            docs[0]["splices"].append({
                "pos": 2,
                "whf_base": [7, 0, 2],
                "whf_derived": [1, 1],
            })
            n_total = 2
        if multi_splice:
            # A second, DISTINCT valid splice position in the SAME derived
            # doc (docs[0]) -- pos=1 (vs. the default entry's pos=2) with its
            # own valid whf_derived target. Exercises the accumulation path
            # (>1 real splice per doc), which every other fixture doc in this
            # module leaves at 0 or 1.
            docs[0]["splices"].append({
                "pos": 1,
                "whf_base": [7, 0, 1],
                "whf_derived": [1, 2],
            })
            n_total = 2
    if drop_splices_key:
        for rec in docs:
            rec.pop("splices")
    return {
        "schema_version": version,
        "shard_name": "shard_00000",
        "shard_index": 7,
        "shard_seed": 1,
        "params": {"n_splice_positions": 99 if wrong_total else n_total},
        "n_docs": n_docs,
        "docs": docs,
    }


def _write(tmp_path, sidecar, stem="shard_00000"):
    d = tmp_path / "_splice_provenance"
    d.mkdir(exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(sidecar))
    return tmp_path


class TestConstruction:
    def test_missing_dir_raises(self, tmp_path) -> None:
        with pytest.raises(SpliceProvenanceError, match="_splice_provenance"):
            SpliceProvenanceIndex(tmp_path)

    def test_happy_path(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar()))
        assert idx.shard_n_docs("shard_00000") == 3
        assert idx.shard_n_splice_positions("shard_00000") == 1
        assert idx.doc_n_segments("shard_00000", 0) == 3
        sp = idx.doc_splices("shard_00000", 0)
        assert sp == {2: SpliceRef(pos=2, whf_base=(7, 0, 2), whf_derived=(1, 0))}
        assert idx.doc_splices("shard_00000", 1) == {}

    def test_multi_splice_positions_in_one_doc(self, tmp_path) -> None:
        """Issue #740's multi-splice accumulation path: a derived doc whose
        provenance sidecar records TWO distinct valid splice positions
        (every other fixture doc in this module has 0 or 1). Pins that
        doc_splices() accumulates both and shard_n_splice_positions counts
        the running total, not just the last one parsed."""
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(multi_splice=True)))
        assert idx.shard_n_splice_positions("shard_00000") == 2
        sp = idx.doc_splices("shard_00000", 0)
        assert sp == {
            1: SpliceRef(pos=1, whf_base=(7, 0, 1), whf_derived=(1, 2)),
            2: SpliceRef(pos=2, whf_base=(7, 0, 2), whf_derived=(1, 0)),
        }


class TestValidationMatrix:
    def test_missing_shard_file_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar()))
        with pytest.raises(SpliceProvenanceError, match="shard_00099"):
            idx.shard_n_docs("shard_00099")

    def test_v1_sidecar_raises_with_resplice_hint(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(version=1, drop_splices_key=True)))
        with pytest.raises(SpliceProvenanceError, match="skip_existing=false"):
            idx.shard_n_docs("shard_00000")

    def test_v2_missing_splices_key_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(drop_splices_key=True)))
        with pytest.raises(SpliceProvenanceError, match="splices"):
            idx.shard_n_docs("shard_00000")

    def test_whf_derived_doc_out_of_range_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(bad_whf_doc=True)))
        with pytest.raises(SpliceProvenanceError, match="whf_derived"):
            idx.shard_n_docs("shard_00000")

    def test_whf_derived_pos_out_of_range_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(bad_whf_pos=True)))
        with pytest.raises(SpliceProvenanceError, match="whf_derived"):
            idx.shard_n_docs("shard_00000")

    def test_pos_zero_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(bad_pos=0)))
        with pytest.raises(SpliceProvenanceError, match="pos"):
            idx.shard_n_docs("shard_00000")

    def test_pos_at_n_segments_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(bad_pos=3)))
        with pytest.raises(SpliceProvenanceError, match="pos"):
            idx.shard_n_docs("shard_00000")

    def test_n_splice_positions_mismatch_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(wrong_total=True)))
        with pytest.raises(SpliceProvenanceError, match="n_splice_positions"):
            idx.shard_n_docs("shard_00000")

    def test_duplicate_splice_position_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar(duplicate_pos=True)))
        with pytest.raises(SpliceProvenanceError, match="duplicates"):
            idx.shard_n_docs("shard_00000")

    def test_doc_idx_out_of_range_raises(self, tmp_path) -> None:
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar()))
        with pytest.raises(SpliceProvenanceError, match="doc"):
            idx.doc_n_segments("shard_00000", 99)

    def test_doc_splices_out_of_range_raises(self, tmp_path) -> None:
        """doc_splices() has its own out-of-range guard (a physical duplicate
        of doc_n_segments()'s, checked separately above) -- never previously
        exercised by any test in this module, since every doc_splices() call
        site elsewhere uses a valid index."""
        idx = SpliceProvenanceIndex(_write(tmp_path, _sidecar()))
        with pytest.raises(SpliceProvenanceError, match="doc"):
            idx.doc_splices("shard_00000", 99)


class TestLoggingD10:
    def _capture(self):
        records = []
        handler_id = logger.add(lambda m: records.append(m), level="INFO")
        return records, handler_id

    def test_info_logged_with_counts(self, tmp_path) -> None:
        records, hid = self._capture()
        try:
            SpliceProvenanceIndex(_write(tmp_path, _sidecar())).shard_n_docs("shard_00000")
        finally:
            logger.remove(hid)
        assert any("splice position" in str(r) and "INFO" in str(r) for r in records)

    def test_error_logged_when_zero_splices(self, tmp_path) -> None:
        records, hid = self._capture()
        try:
            SpliceProvenanceIndex(
                _write(tmp_path, _sidecar(with_splice=False))
            ).shard_n_docs("shard_00000")
        finally:
            logger.remove(hid)
        assert any("ZERO splice positions" in str(r) for r in records)


class TestLru1:
    def test_switching_shards_reparses(self, tmp_path) -> None:
        root = _write(tmp_path, _sidecar())
        _write(tmp_path, _sidecar(n_docs=5), stem="shard_00001")
        idx = SpliceProvenanceIndex(root)
        assert idx.shard_n_docs("shard_00000") == 3
        assert idx.shard_n_docs("shard_00001") == 5
        assert idx.shard_n_docs("shard_00000") == 3  # reparse after eviction


def test_real_sidecar_from_splicer_loads_clean(tmp_path) -> None:
    """Reader<->writer compatibility pin: build a real v2 sidecar via the actual
    splicer primitives (not the synthetic ``_sidecar()`` fixture above) and
    confirm SpliceProvenanceIndex reads it back cleanly, with every SpliceRef
    round-tripping exactly against the JSON the splicer produced.
    """
    tok = CharMergeTokenizer(bos_id=500)
    seg_size = 2
    corpus = [
        "alpha beta gamma delta",
        "epsilon zeta eta theta",
        "iota kappa lambda mu",
        "nu xi omicron pi rho",
        "sigma tau upsilon phi",
    ]
    cut_fraction, seed, shard_index = 0.6, 42, 0
    shard_name = "shard_00000"
    context_window = 2048
    sae_max_context = None

    base_shard = make_base_shard(corpus, tok, seg_size, context_window=context_window)
    source_segments, _load_stats = load_source_segments(
        base_shard, corpus, tok,
        segment_size=seg_size, context_window=context_window,
        sae_max_context=sae_max_context, shard_name=shard_name,
    )
    shard_seed = recombination_seed(seed, shard_index, cut_fraction)
    limit = (
        context_window if sae_max_context is None
        else min(context_window, sae_max_context)
    )
    result = recombine_segments(
        source_segments, tok,
        cut_fraction=cut_fraction, shard_seed=shard_seed,
        shard_name=shard_name, limit=limit,
    )
    params = {"cut_fraction": cut_fraction, "seed": seed}
    prov = build_provenance(
        result, source_segments, params=params,
        shard_index=shard_index, shard_name=shard_name, shard_seed=shard_seed,
    )

    sidecar_path = tmp_path / "_splice_provenance" / f"{shard_name}.json"
    atomic_json_dump(prov, sidecar_path)

    idx = SpliceProvenanceIndex(tmp_path)  # must not raise
    assert idx.shard_n_docs(shard_name) == prov["n_docs"]
    assert idx.shard_n_splice_positions(shard_name) == prov["params"]["n_splice_positions"]

    total = 0
    for d, doc in enumerate(prov["docs"]):
        splices = idx.doc_splices(shard_name, d)
        expected_by_pos = {rec["pos"]: rec for rec in doc["splices"]}
        assert set(splices) == set(expected_by_pos)
        for pos, ref in splices.items():
            rec = expected_by_pos[pos]
            assert ref.pos == rec["pos"]
            assert ref.whf_base == tuple(rec["whf_base"])
            assert ref.whf_derived == tuple(rec["whf_derived"])
        total += len(splices)
    assert total == prov["params"]["n_splice_positions"]
