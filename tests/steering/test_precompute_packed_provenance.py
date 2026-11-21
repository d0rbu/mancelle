"""Packer-integrated segment provenance (scripts.preprocess.precompute_packed_rows).

The packer records, for every packed segment, the triple
``(base_shard_idx, doc_idx, seg_ordinal)`` into ``{packed_dir}/provenance/``
in the SAME flush that writes the packed shard — the mapping is produced by
the packing loop itself, so there is no replay step that could drift from it.

These tests build REAL packed shards by driving the production packing
function over the tiny extraction corpus from
``test_gradient_attribution_producer_integration`` and pin:

* the content-level JOIN PROOF — every packed segment's fid list equals the
  base segment's fid slice located via provenance (exactly the contract the
  gradient-selection join relies on), across an EXHAUSTIVE GRID of packing
  shapes (context_len × buffer_size × rows_per_shard) so cropping, multi-doc
  rows, partial final shards, and the crop-to-the-T+1-target-slot (a placed
  doc contributing zero packed segments) case are all exercised;
* 1:1 provenance/packed shard coverage, dtypes, and the prefix-ordinal
  invariant;
* byte-determinism of packed AND provenance shards across fresh re-packs
  (regression for the uninitialized-row-tail bug that broke replay
  bit-stability on Linux — the row packer must zero-fill);
* ``resume=true`` regenerating missing provenance by rewriting the
  incomplete shards byte-identically (crash-mid-flush + pre-integration
  migration), skipping the complete prefix untouched;
* the tokenizer-drift resume refusal;
* provenance payloads remaining opaque to packing decisions.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t

from errors import PackedRowInvariantError
from scripts.preprocess.precompute_packed_rows import (
    _assert_base_stems_aligned,
    _assert_resume_packing_identity_unchanged,
    _pack_row_with_provenance,
    _row_provenance,
    _RowCSR,
    precompute_packed_rows,
)
from steering.conditional.encoder_decoder.sorted_doc_buffer import SortedDocBuffer
from steering.conditional.feature_index import SegmentFeatures
from steering.feature_io import glob_features, load_features, shard_stem_from_path
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    load_overlay_shard,
    read_manifest,
    write_manifest,
)
from tests.steering.test_gradient_attribution_producer_integration import (
    CONTEXT_WINDOW,
    TinyTok,
    _corpus_template,  # noqa: F401  (re-exported fixture)
    corpus,  # noqa: F401  (re-exported fixture)
)

CONTEXT_LEN = 24  # < doc lengths => docs get cropped and rows pack several docs
BUFFER_SIZE = 2
ROWS_PER_SHARD = 2
PCT_CHECKSUM = hashlib.sha256(b"tiny-pct").hexdigest()
TOK_CHECKSUM = hashlib.sha256(b"tiny-tok").hexdigest()


class NanochatTinyTok:
    """The nanochat-side tokenizer interface over the SAME id mapping as the
    extraction tokenizer (1 token per char + BOS) — token boundaries align
    1:1, which is all the packing fast path needs."""

    bos_token_id = TinyTok.bos_token_id

    def __init__(self):
        self._tok = TinyTok()

    def encode(self, texts, prepend=None):
        if isinstance(texts, str):
            return self._tok.encode(texts)
        return [self._tok.encode(text) for text in texts]

    def decode(self, token_ids) -> str:
        return self._tok.decode(token_ids)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id


class FakeFeatureIndex:
    """FeatureIndex stand-in backed by the real tiny base shards, with token
    boundaries derived from the (shared) 1-token-per-char mapping."""

    def __init__(self, feature_dir: str | Path, **_kw):
        self.feature_dir = Path(feature_dir)
        self._docs: dict[tuple[str, int], list[SegmentFeatures]] = {}
        self._stems: list[str] = []
        for shard_path in glob_features(self.feature_dir):
            stem = shard_stem_from_path(shard_path)
            self._stems.append(stem)
            data = load_features(shard_path)
            offsets = data["offsets"].to(t.int64)
            doc_ids = data["doc_ids"].tolist()
            seg_positions = data["segment_positions"].tolist()
            n_docs = int(data["n_docs"])
            kept_lens = [
                min(int(c), CONTEXT_WINDOW) for c in data["doc_token_counts"].tolist()
            ]
            per_doc: dict[int, list[int]] = {d: [] for d in range(n_docs)}
            for s, d in enumerate(doc_ids):
                per_doc[d].append(s)
            for d, segs in per_doc.items():
                feats = []
                for ordinal, s in enumerate(segs):
                    start = seg_positions[s]
                    end = (
                        seg_positions[segs[ordinal + 1]]
                        if ordinal + 1 < len(segs)
                        else kept_lens[d]
                    )
                    lo, hi = int(offsets[s]), int(offsets[s + 1])
                    feats.append(
                        SegmentFeatures(
                            segment_idx=ordinal,
                            position=start,
                            feature_ids=data["feature_ids"][lo:hi].to(t.int32),
                            feature_mags=t.zeros(hi - lo, dtype=t.float32),
                            char_start=start,
                            char_end=end,
                            feature_percentiles=t.zeros(hi - lo, dtype=t.uint8),
                            token_start=start,
                            token_end=end,
                        )
                    )
                self._docs[(stem, d)] = feats

    @property
    def segment_size(self) -> int:
        return 8

    @property
    def shard_stems(self) -> list[str]:
        """Sorted stems — defines the base_shard_idx mapping (mirrors the
        real FeatureIndex's sorted-scan order)."""
        return sorted(self._stems)

    def lookup(self, shard_stem: str, doc_idx: int) -> list[SegmentFeatures]:
        return self._docs[(shard_stem, doc_idx)]


def _packed_manifest_meta(stems: list[str], packed_dir: Path) -> dict:
    """The metadata run() would record (the fixture drives the packing
    function directly, so it writes the manifest itself)."""
    return {
        "type": "packed_rows",
        "context_len": CONTEXT_LEN,
        "max_shards": None,
        "buffer_size": BUFFER_SIZE,
        "rows_per_shard": ROWS_PER_SHARD,
        "pct_checksum": PCT_CHECKSUM,
        "tok_checksum": TOK_CHECKSUM,
        "base_shard_stems": stems,
        "provenance_shards": len(
            glob_overlay_shards(provenance_dir(packed_dir))
        ),
    }


def _pack_split(out_dir: Path, paths: list[str], fi: FakeFeatureIndex, **kw) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    kw.setdefault("rows_per_shard", ROWS_PER_SHARD)
    precompute_packed_rows(
        parquet_paths=paths,
        tokenizer=NanochatTinyTok(),
        feature_index=cast(Any, fi),
        context_len=CONTEXT_LEN,
        output_dir=out_dir,
        buffer_size=BUFFER_SIZE,
        **kw,
    )


@pytest.fixture()
def packed(corpus) -> dict:  # noqa: F811  (fixture reuse)
    """Real packed train + val dirs in the canonical layout, built by the
    production packing function over the tiny corpus (provenance included)."""
    feature_dir = corpus["feature_dir"]
    parquets = sorted(str(p) for p in corpus["data_dir"].glob("*.parquet"))
    assert len(parquets) == 2
    fi = FakeFeatureIndex(feature_dir)
    base_paths = glob_features(feature_dir)

    tok_sub = feature_dir / "pct" / PCT_CHECKSUM[:16] / "tok" / TOK_CHECKSUM[:16]
    train_dir = tok_sub / f"c{CONTEXT_LEN}" / "sall"
    val_dir = tok_sub / f"c{CONTEXT_LEN}" / "val_sall"
    for out_dir, paths in [(train_dir, parquets[:-1]), (val_dir, parquets[-1:])]:
        _pack_split(out_dir, paths, fi)
        write_manifest(
            out_dir,
            f"{PCT_CHECKSUM}:{TOK_CHECKSUM}",
            OverlayKind.PACKED_ROW,
            metadata=_packed_manifest_meta(fi.shard_stems, out_dir),
            base_shard_paths=base_paths,
        )
        assert glob_overlay_shards(out_dir), f"no packed shards in {out_dir}"
    return {
        **corpus,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "parquets": parquets,
        "feature_index": fi,
    }


def _assert_join_contract(packed_dir: Path, feature_dir: Path) -> int:
    """THE content-level proof: every packed segment's fid list equals the
    base segment's fid slice located via provenance. Returns segments checked."""
    base = {}
    manifest = read_manifest(packed_dir)
    assert manifest is not None
    stems = manifest["metadata"]["base_shard_stems"]
    for stem in stems:
        data = load_features(feature_dir / f"{stem}_features.safetensors.zst")
        counts = t.bincount(
            data["doc_ids"].to(t.int64), minlength=int(data["n_docs"])
        )
        dso = t.cat([t.zeros(1, dtype=t.int64), counts.cumsum(0)])
        base[stem] = (data["offsets"].to(t.int64), data["feature_ids"].to(t.int64), dso)

    checked = 0
    for shard_path in glob_overlay_shards(packed_dir):
        from steering.packed_batch_io import load_packed_row_shard

        packed_shard = load_packed_row_shard(shard_path)
        prov = load_overlay_shard(provenance_dir(packed_dir) / shard_path.name)
        b_idx = prov[PROVENANCE_BASE_SHARD_IDX_KEY]
        d_idx = prov[PROVENANCE_DOC_IDX_KEY]
        s_ord = prov[PROVENANCE_SEG_ORDINAL_KEY]
        assert (
            len(b_idx) == len(d_idx) == len(s_ord) == packed_shard.total_segments
        )
        sfo = packed_shard.seg_feat_offsets.to(t.int64)
        for seg in range(packed_shard.total_segments):
            packed_fids = packed_shard.feature_ids[
                int(sfo[seg]) : int(sfo[seg + 1])
            ].to(t.int64)
            assert int(b_idx[seg]) >= 0, (
                f"{shard_path.name} seg {seg}: negative provenance (no "
                f"sentinels exist — every packed segment has a base segment)"
            )
            offsets, fids, dso = base[stems[int(b_idx[seg])]]
            gseg = int(dso[int(d_idx[seg])]) + int(s_ord[seg])
            base_fids = fids[int(offsets[gseg]) : int(offsets[gseg + 1])]
            assert t.equal(packed_fids, base_fids), (
                f"{shard_path.name} seg {seg}: provenance points at a base "
                f"segment with different features"
            )
            checked += 1
    return checked


def _shard_bytes(directory: Path) -> dict[str, bytes]:
    return {p.name: p.read_bytes() for p in glob_overlay_shards(directory)}


def _recursive_digest_map(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


class TestPackerProvenance:
    def test_written_alongside_shards_with_join_contract(self, packed):
        for key in ("train_dir", "val_dir"):
            pdir = packed[key]
            prov = provenance_dir(pdir)
            assert prov.is_dir()
            assert {p.name for p in glob_overlay_shards(prov)} == {
                p.name for p in glob_overlay_shards(pdir)
            }
            manifest = read_manifest(pdir)
            assert manifest is not None
            meta = manifest["metadata"]
            assert meta["provenance_shards"] == len(glob_overlay_shards(prov))
            assert meta["base_shard_stems"] == packed["feature_index"].shard_stems
            checked = _assert_join_contract(pdir, packed["feature_dir"])
            assert checked > 0, f"join proof checked no segments in {pdir}"

    def test_dtypes(self, packed):
        shard = glob_overlay_shards(provenance_dir(packed["train_dir"]))[0]
        prov = load_overlay_shard(shard)
        for key in (
            PROVENANCE_BASE_SHARD_IDX_KEY,
            PROVENANCE_DOC_IDX_KEY,
            PROVENANCE_SEG_ORDINAL_KEY,
        ):
            assert prov[key].dtype == t.int32

    def test_no_sentinels_in_featureful_corpus(self, packed):
        """Every doc of the tiny corpus has stored features, so no packed
        segment may carry sentinel provenance."""
        for pdir in (packed["train_dir"], packed["val_dir"]):
            for shard in glob_overlay_shards(provenance_dir(pdir)):
                prov = load_overlay_shard(shard)
                assert bool((prov[PROVENANCE_BASE_SHARD_IDX_KEY] >= 0).all())
                assert bool((prov[PROVENANCE_DOC_IDX_KEY] >= 0).all())
                assert bool((prov[PROVENANCE_SEG_ORDINAL_KEY] >= 0).all())

    def test_prefix_ordinal_invariant(self, packed):
        """Within a packed row, each placed doc's segments carry ordinals
        0..n-1 in order (the join's prefix invariant)."""
        from steering.packed_batch_io import load_packed_row_shard

        rows_checked = 0
        for shard_path in glob_overlay_shards(packed["train_dir"]):
            shard = load_packed_row_shard(shard_path)
            prov = load_overlay_shard(
                provenance_dir(packed["train_dir"]) / shard_path.name
            )
            s_ord = prov[PROVENANCE_SEG_ORDINAL_KEY].tolist()
            rso = shard.row_seg_offsets.tolist()
            doc_ids = shard.seg_doc_id.tolist()
            for r in range(shard.n_rows):
                lo, hi = rso[r], rso[r + 1]
                expected_next: dict[int, int] = {}
                for s in range(lo, hi):
                    d = doc_ids[s]
                    assert s_ord[s] == expected_next.get(d, 0)
                    expected_next[d] = s_ord[s] + 1
                rows_checked += 1
        assert rows_checked > 0

    def test_packing_and_provenance_byte_deterministic(self, packed, tmp_path):
        """A fresh re-pack of the same corpus is byte-identical for packed AND
        provenance shards (regression: the uninitialized `t.empty` row tail
        made the final partial row's bytes allocator-dependent)."""
        fresh = tmp_path / "repack"
        _pack_split(fresh, packed["parquets"][:-1], packed["feature_index"])
        assert _shard_bytes(fresh) == _shard_bytes(packed["train_dir"])
        assert _shard_bytes(provenance_dir(fresh)) == _shard_bytes(
            provenance_dir(packed["train_dir"])
        )

    def test_resume_regenerates_missing_provenance(self, packed):
        """resume=true over a dir whose provenance was deleted (a dir packed
        before native provenance, or a crash between the two flushes) rewrites
        the now-incomplete shards through the normal flush path — packed
        shards come back BYTE-IDENTICAL (packing is deterministic) and their
        provenance is regenerated identically."""
        train = packed["train_dir"]
        prov = provenance_dir(train)
        before_prov = _shard_bytes(prov)
        before_packed = _shard_bytes(train)
        shutil.rmtree(prov)
        _pack_split(train, packed["parquets"][:-1], packed["feature_index"], resume=True)
        assert _shard_bytes(prov) == before_prov, "provenance not regenerated identically"
        assert _shard_bytes(train) == before_packed, "packed shards changed bytes"

    def test_resume_skips_complete_prefix_rewrites_incomplete(self, packed, tmp_path):
        """Crash-mid-flush on a MULTI-shard dir: resume replays the complete
        prefix (skipped, untouched) and rewrites only the first incomplete
        shard (packed missing its provenance) byte-identically. Uses
        rows_per_shard=1 so the tiny corpus yields several shards."""
        multi = tmp_path / "multi"
        _pack_split(
            multi, packed["parquets"][:-1], packed["feature_index"],
            rows_per_shard=1,
        )
        prov = provenance_dir(multi)
        packed_shards = sorted(glob_overlay_shards(multi))
        assert len(packed_shards) >= 2, "need >= 2 shards to test a prefix skip"
        before_packed = _shard_bytes(multi)
        first_mtime = packed_shards[0].stat().st_mtime_ns
        # Delete the LAST provenance shard => last packed shard is incomplete.
        sorted(glob_overlay_shards(prov))[-1].unlink()
        _pack_split(
            multi, packed["parquets"][:-1], packed["feature_index"],
            rows_per_shard=1, resume=True,
        )
        assert _shard_bytes(multi) == before_packed, "packed shards changed bytes"
        # The complete-prefix shard 0 was skipped (mtime unchanged).
        assert sorted(glob_overlay_shards(multi))[0].stat().st_mtime_ns == first_mtime

    def test_resume_with_complete_provenance_writes_nothing(self, packed):
        train = packed["train_dir"]
        prov = provenance_dir(train)
        before = {p.name: p.stat().st_mtime_ns for p in glob_overlay_shards(prov)}
        _pack_split(train, packed["parquets"][:-1], packed["feature_index"], resume=True)
        after = {p.name: p.stat().st_mtime_ns for p in glob_overlay_shards(prov)}
        assert after == before, "no-op resume rewrote provenance shards"

    def test_tokenizer_drift_resume_refused(self, packed):
        before = _recursive_digest_map(packed["train_dir"])
        with pytest.raises(PackedRowInvariantError, match="tokenizer"):
            _assert_resume_packing_identity_unchanged(
                packed["train_dir"],
                tok_checksum="f" * 64,
                buffer_size=BUFFER_SIZE,
                rows_per_shard=ROWS_PER_SHARD,
            )
        assert _recursive_digest_map(packed["train_dir"]) == before

    @pytest.mark.parametrize(
        ("field", "override"),
        [("buffer_size", BUFFER_SIZE + 1), ("rows_per_shard", ROWS_PER_SHARD + 1)],
    )
    def test_packing_identity_drift_refused_without_mutation(
        self, packed, field, override
    ):
        kwargs = {
            "tok_checksum": TOK_CHECKSUM,
            "buffer_size": BUFFER_SIZE,
            "rows_per_shard": ROWS_PER_SHARD,
        }
        kwargs[field] = override
        before = _recursive_digest_map(packed["train_dir"])
        with pytest.raises(PackedRowInvariantError, match=field):
            _assert_resume_packing_identity_unchanged(packed["train_dir"], **kwargs)
        assert _recursive_digest_map(packed["train_dir"]) == before

    @pytest.mark.parametrize(
        "missing",
        [
            "manifest",
            "tok_checksum",
            "buffer_size",
            "rows_per_shard",
        ],
    )
    def test_payload_requires_complete_readable_identity_manifest(self, packed, missing):
        packed_dir = packed["train_dir"]
        manifest_path = packed_dir / "manifest.json"
        if missing == "manifest":
            manifest_path.unlink()
        else:
            manifest = json.loads(manifest_path.read_text())
            del manifest["metadata"][missing]
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        before = _recursive_digest_map(packed_dir)

        with pytest.raises(PackedRowInvariantError, match="manifest|metadata"):
            _assert_resume_packing_identity_unchanged(
                packed_dir,
                tok_checksum=TOK_CHECKSUM,
                buffer_size=BUFFER_SIZE,
                rows_per_shard=ROWS_PER_SHARD,
            )
        assert _recursive_digest_map(packed_dir) == before

    def test_same_identity_and_payload_free_directory_are_no_ops(self, packed, tmp_path):
        packed_dir = packed["train_dir"]
        before = _recursive_digest_map(packed_dir)
        _assert_resume_packing_identity_unchanged(
            packed_dir,
            tok_checksum=TOK_CHECKSUM,
            buffer_size=BUFFER_SIZE,
            rows_per_shard=ROWS_PER_SHARD,
        )
        assert _recursive_digest_map(packed_dir) == before

        empty_dir = tmp_path / "empty_resume"
        empty_dir.mkdir()
        _assert_resume_packing_identity_unchanged(
            empty_dir,
            tok_checksum=TOK_CHECKSUM,
            buffer_size=BUFFER_SIZE,
            rows_per_shard=ROWS_PER_SHARD,
        )
        assert _recursive_digest_map(empty_dir) == {}


class TestPackRowCore:
    def test_zero_filled_tail(self):
        """The unplaced tail of a partial row is zero bytes, not uninitialized
        memory (the Linux nondeterminism root cause)."""
        buf = SortedDocBuffer()
        buf.append(([1, 2, 3], None, (0, 0)))
        row, info, prov = _pack_row_with_provenance(buf, 25)
        assert t.equal(row[3:], t.zeros(22, dtype=t.long))
        assert prov == [(0, 0)]
        assert [(p, dt) for p, _sf, dt in info] == [(0, [1, 2, 3])]

    def test_provenance_payload_does_not_affect_packing(self):
        """Different provenance identities produce the same placements and tokens."""

        def fill(buf, base_shard_idx):
            docs = [[1] * 5, [2] * 9, [3] * 3, [4] * 7]
            for i, d in enumerate(docs):
                buf.append((d, None, (base_shard_idx, i)))

        buf_a, buf_b = SortedDocBuffer(), SortedDocBuffer()
        fill(buf_a, 0)
        fill(buf_b, 1)
        rows_a, rows_b = [], []
        while buf_a:
            rows_a.append(_pack_row_with_provenance(buf_a, 12))
        while buf_b:
            rows_b.append(_pack_row_with_provenance(buf_b, 12))
        assert len(rows_a) == len(rows_b)
        for (tok_a, info_a, prov_a), (tok_b, info_b, prov_b) in zip(
            rows_a, rows_b, strict=True
        ):
            assert t.equal(tok_a, tok_b)
            assert [(p, dt) for p, _s, dt in info_a] == [
                (p, dt) for p, _s, dt in info_b
            ]
            assert [doc_idx for _shard, doc_idx in prov_a] == [
                doc_idx for _shard, doc_idx in prov_b
            ]
            assert all(shard == 0 for shard, _doc_idx in prov_a)
            assert all(shard == 1 for shard, _doc_idx in prov_b)

    def test_row_provenance_rejects_out_of_range_seg_doc_id(self):
        """A seg_doc_id referencing a doc index >= the placed-doc count is a
        structural break; _row_provenance raises a typed PackedRowInvariantError
        rather than a bare IndexError on row_doc_info[d]."""
        row_csr = _RowCSR(
            tokens=t.zeros(5, dtype=t.uint16),
            seg_token_start=t.tensor([0, 2], dtype=t.int32),
            seg_token_end=t.tensor([2, 4], dtype=t.int32),
            # one doc was placed, but seg_doc_id references doc index 1 (OOR)
            seg_doc_id=t.tensor([0, 1], dtype=t.uint8),
            feature_ids=t.tensor([], dtype=t.uint16),
            feature_pcts=t.tensor([], dtype=t.uint8),
            feat_counts=t.tensor([0, 0], dtype=t.int32),
        )
        row_doc_info = [(0, [object()], [1, 2, 3])]  # length 1 == one placed doc
        row_doc_prov = [(0, 0)]
        with pytest.raises(PackedRowInvariantError, match="seg_doc_id"):
            _row_provenance(row_csr, row_doc_info, row_doc_prov)

    def test_base_stems_alignment_guard(self):
        """The manifest's base_shard_stems must equal feature_index.shard_stems
        (provenance base_shard_idx indexes into it); a reordering raises rather
        than silently mislabeling every triple. The aligned case is a no-op."""
        _assert_base_stems_aligned(["s0", "s1"], ["s0", "s1"])  # no raise
        with pytest.raises(PackedRowInvariantError, match="stem order"):
            _assert_base_stems_aligned(["s0", "s1"], ["s1", "s0"])


def _pack_and_manifest(out_dir: Path, paths, fi: FakeFeatureIndex, base_paths,
                       context_len: int, **kw) -> None:
    """Pack a split AND write the manifest (the producer's run() does both;
    _pack_split alone does not), so _assert_join_contract can read
    base_shard_stems."""
    out_dir.mkdir(parents=True, exist_ok=True)
    precompute_packed_rows(
        parquet_paths=paths,
        tokenizer=NanochatTinyTok(),
        feature_index=cast(Any, fi),
        context_len=context_len,
        output_dir=out_dir,
        buffer_size=kw.get("buffer_size", BUFFER_SIZE),
        rows_per_shard=kw.get("rows_per_shard", ROWS_PER_SHARD),
        resume=kw.get("resume", False),
    )
    write_manifest(
        out_dir, f"{PCT_CHECKSUM}:{TOK_CHECKSUM}", OverlayKind.PACKED_ROW,
        metadata={
            "type": "packed_rows",
            "context_len": context_len,
            "buffer_size": kw.get("buffer_size", BUFFER_SIZE),
            "rows_per_shard": kw.get("rows_per_shard", ROWS_PER_SHARD),
            "pct_checksum": PCT_CHECKSUM, "tok_checksum": TOK_CHECKSUM,
            "base_shard_stems": fi.shard_stems,
            "provenance_shards": len(glob_overlay_shards(provenance_dir(out_dir))),
        },
        base_shard_paths=base_paths,
    )


# A grid of packing shapes. CONTEXT_LEN spans below/around the doc lengths so
# docs are cropped (incl. cropped-to-the-T+1-target-slot, count==0); buffer
# sizes change best-fit ordering; rows_per_shard changes shard boundaries.
@pytest.mark.parametrize("context_len", [12, 18, 24, 40])
@pytest.mark.parametrize("buffer_size", [1, 2, 4])
@pytest.mark.parametrize("rows_per_shard", [1, 3])
def test_join_contract_across_packing_configs(
    corpus, tmp_path, context_len, buffer_size, rows_per_shard  # noqa: F811
):
    """THE rigorous join proof: for an exhaustive grid of packing shapes,
    drive the REAL packer over the real corpus and assert every packed
    segment's fid list equals the base segment located via its provenance.
    Covers cropping, multi-doc rows, partial final shards, and the
    crop-to-target-slot (count==0) case across configurations."""
    feature_dir = corpus["feature_dir"]
    parquets = sorted(str(p) for p in corpus["data_dir"].glob("*.parquet"))
    fi = FakeFeatureIndex(feature_dir)
    base_paths = glob_features(feature_dir)
    out_dir = tmp_path / "packed"
    _pack_and_manifest(
        out_dir, parquets[:-1], fi, base_paths, context_len,
        buffer_size=buffer_size, rows_per_shard=rows_per_shard,
    )
    # 1:1 provenance/packed coverage.
    assert {p.name for p in glob_overlay_shards(provenance_dir(out_dir))} == {
        p.name for p in glob_overlay_shards(out_dir)
    }
    # Every packed segment's features match the base segment via provenance.
    checked = _assert_join_contract(out_dir, feature_dir)
    assert checked > 0, "join proof checked no segments"
    # And the result is byte-deterministic: a fresh re-pack matches.
    redo = tmp_path / "redo"
    _pack_and_manifest(
        redo, parquets[:-1], fi, base_paths, context_len,
        buffer_size=buffer_size, rows_per_shard=rows_per_shard,
    )
    assert _shard_bytes(redo) == _shard_bytes(out_dir)
    assert _shard_bytes(provenance_dir(redo)) == _shard_bytes(provenance_dir(out_dir))
