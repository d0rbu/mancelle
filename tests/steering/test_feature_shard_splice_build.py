"""Tests for build_derived_shard known-boundary construction (Task 8, issue #590).

Covers the exact char-boundary construction for BOTH root (new_started=False) and
new-started (new_started=True) docs: identity proof vs segment_document, BOS-char
accounting, feature verbatim through disk, tiling assert, parquet-text BOS-free,
CSR/dtype/metadata invariants, and the end-to-end splice smoke.
"""

from __future__ import annotations

import pyarrow as pa
import pytest
import torch as t

from nanochat.atomic_io import atomic_write_parquet
from steering._roundtrip_seam import is_roundtrip_stable
from steering.feature_io import load_features, read_shard_metadata, save_features
from steering.feature_shard_splice import (
    DerivedDoc,
    DerivedSegmentationError,
    ShardSpliceResult,
    build_derived_shard,
    load_source_segments,
    recombination_seed,
    recombine_segments,
    splice_feature_shard,
)
from steering.refined_types import PositiveInt
from steering.segmentation import (
    SegmentMode,
    segment_document,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

# ---------------------------------------------------------------------------
# Shared test tokenizer and corpus constants
# ---------------------------------------------------------------------------

TOK = CharMergeTokenizer(bos_id=500)  # no merges → every stitch is stable

# A loop-producing corpus/seed (verified in test_feature_shard_splice_recombine.py).
# Produces 2 docs: 1 root chain (new_started=False) + 1 loop doc (new_started=True).
LOOP_CORPUS = ["abcdefghi"]  # 1 doc, segment_size=2 → 5 segs; BOS+9 chars=10 toks
LOOP_SEED = 0
LOOP_SEGMENT_SIZE = 2


def _base_meta(*, segment_size: int, context_window: int = 2048) -> dict:
    return {
        "layer": 8,
        "context_window": context_window,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": segment_size,
        "context_independent_features": False,
        "ignore_bos_features": True,
    }


def _load_segs(texts, tok, segment_size, context_window=2048):
    """Build a base shard from texts and return source segments."""
    shard = make_base_shard(texts, tok, segment_size, context_window=context_window)
    segs, _ = load_source_segments(
        shard, texts, tok, segment_size=segment_size,
        context_window=context_window, sae_max_context=None, shard_name="test",
    )
    return segs, shard


# ---------------------------------------------------------------------------
# 1. Identity proof: root doc known-boundary == segment_document
# ---------------------------------------------------------------------------


def test_root_doc_matches_segment_document_construction():
    """For non-split root docs: known-boundary char starts/ends byte-equal segment_document.

    This is THE refactor-identity proof: the new single-path construction must
    produce identical char boundaries to the old segment_document path for all
    new_started=False docs.
    """
    texts = ["alpha beta gamma", "delta epsilon zeta eta"]
    segment_size = 4
    segs, _ = _load_segs(texts, TOK, segment_size)

    seg_tokens = [[s.tokens for s in doc] for doc in segs]

    # Build root-only ShardSpliceResult (each doc = all its source segs, no cuts).
    docs = tuple(
        DerivedDoc(tuple((d, j) for j in range(len(segs[d]))), False)
        for d in range(len(segs))
    )
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = _base_meta(segment_size=segment_size)

    shard, derived_texts = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)

    # Compare against the reference: segment_document on the same spliced tokens.
    gi = 0
    for d, doc in enumerate(docs):
        spliced = [tid for ref in doc.segment_refs for tid in seg_tokens[ref[0]][ref[1]]]
        ref_segs, ref_cs, ref_ce = segment_document(
            spliced,
            TOK,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(segment_size),
        )
        n_segs = len(ref_segs)
        cs = shard["segment_char_starts"][gi : gi + n_segs].tolist()
        ce = shard["segment_char_ends"][gi : gi + n_segs].tolist()
        assert cs == ref_cs, f"doc {d}: char_starts mismatch: {cs} vs {ref_cs}"
        assert ce == ref_ce, f"doc {d}: char_ends mismatch: {ce} vs {ref_ce}"
        gi += n_segs


# ---------------------------------------------------------------------------
# 2. new_started doc layout (slow — real gemma tokenizer)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_new_started_doc_layout_real_gemma(gemma_tok, tmp_path):
    """Seg-0 char span covers BOS + content; BOS char len consistent across doc types.

    For new-started docs:
      - seg-0 char_start == 0 (the synthetic BOS anchors the span at the start)
      - last_seg_end - len(parquet_text) equals the same value as for root docs
        (the BOS char overhead is consistent regardless of new_started flag)

    Constructs DerivedDocs directly for precision (avoids non-deterministic cuts).
    """
    # Use a text long enough to produce ≥ 3 segments with segment_size=4.
    texts = ["hello world this is a feature shard splice test"]
    segment_size = 4
    context_window = 128
    base = make_base_shard(texts, gemma_tok, segment_size, context_window=context_window)
    base_path = save_features(base, tmp_path / "base")
    base_loaded = load_features(base_path)

    segs, _ = load_source_segments(
        base_loaded, texts, gemma_tok, segment_size=segment_size,
        context_window=context_window, sae_max_context=None, shard_name="gemma_test",
    )
    n_segs_doc0 = len(segs[0])
    assert n_segs_doc0 >= 3, (
        f"need ≥ 3 source segments; got {n_segs_doc0} — adjust text or segment_size"
    )

    # Root doc: all segs of doc 0 (new_started=False)
    root_refs = tuple((0, j) for j in range(n_segs_doc0))
    # new-started doc: segs 1 and 2 of doc 0 with synthetic BOS
    new_refs = ((0, 1), (0, 2))

    docs = (
        DerivedDoc(root_refs, False),
        DerivedDoc(new_refs, True),
    )
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = _base_meta(segment_size=segment_size, context_window=context_window)

    shard, derived_texts = build_derived_shard(result, segs, meta, gemma_tok, segment_size=segment_size)

    # --- root doc stats ---
    n_segs_root = len(root_refs)
    root_ce = shard["segment_char_ends"][:n_segs_root].tolist()
    root_text = derived_texts[0]
    root_bos_char_len = root_ce[-1] - len(root_text)

    # --- new-started doc stats ---
    n_segs_new = len(new_refs)
    new_cs = shard["segment_char_starts"][n_segs_root : n_segs_root + n_segs_new].tolist()
    new_ce = shard["segment_char_ends"][n_segs_root : n_segs_root + n_segs_new].tolist()
    new_text = derived_texts[1]
    new_bos_char_len = new_ce[-1] - len(new_text)

    # seg-0 of new_started doc starts at char 0.
    assert new_cs[0] == 0, f"new-started seg-0 must start at char 0, got {new_cs[0]}"

    # BOS char contribution is the same for both doc types.
    assert root_bos_char_len == new_bos_char_len, (
        f"BOS char len inconsistent: root={root_bos_char_len}, new_started={new_bos_char_len}"
    )


# ---------------------------------------------------------------------------
# 3. Feature preservation through save/load round-trip
# ---------------------------------------------------------------------------


def test_new_started_features_verbatim_through_disk(tmp_path):
    """new-started docs: feature_ids + feature_mags survive save→load byte-exactly.

    Mirrors the fp16-consistent fixture pattern from the trim tests: the base shard
    is saved then loaded so source segment feature_mags are already fp16-representable.
    A second fp16 round-trip (through the derived shard) is then lossless, making
    t.equal hold.
    """
    texts = ["alpha beta gamma delta", "epsilon zeta eta theta iota"]
    segment_size = 4
    context_window = 2048

    # Go through save→load so feature_mags are fp16-quantised before comparison.
    base_dict = make_base_shard(texts, TOK, segment_size, context_window=context_window)
    base_path = save_features(base_dict, tmp_path / "base")
    base_loaded = load_features(base_path)
    segs, _ = load_source_segments(
        base_loaded, texts, TOK, segment_size=segment_size,
        context_window=context_window, sae_max_context=None, shard_name="test",
    )

    # Use (0,1) and (0,2) — second and third segs of doc 0 (both interior full segs).
    assert len(segs[0]) >= 3, f"doc 0 must have ≥ 3 segs; got {len(segs[0])}"
    new_refs = ((0, 1), (0, 2))
    docs = (DerivedDoc(new_refs, True),)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = _base_meta(segment_size=segment_size)

    derived_shard, _ = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)

    # Save and reload through the real pipeline.
    shard_path = save_features(derived_shard, tmp_path / "derived")
    loaded = load_features(shard_path)

    # Feature arrays must be verbatim (t.equal) for each segment.
    for k, ref in enumerate(new_refs):
        src = segs[ref[0]][ref[1]]
        lo = int(loaded["offsets"][k])
        hi = int(loaded["offsets"][k + 1])
        actual_fids = loaded["feature_ids"][lo:hi]
        actual_fmags = loaded["feature_mags"][lo:hi]
        # feature_ids: survive uint16→int32 exactly (values in [0, 4095] ⊆ [0, 65535]).
        assert t.equal(actual_fids, src.feature_ids), (
            f"seg {k} feature_ids mismatch after disk round-trip"
        )
        # feature_mags: already went through one fp16 round-trip (base shard save/load),
        # so they are exactly fp16-representable; a second round-trip is lossless.
        assert t.equal(actual_fmags, src.feature_mags), (
            f"seg {k} feature_mags mismatch after disk round-trip"
        )


# ---------------------------------------------------------------------------
# 4. Tiling assert fires on malformed doc
# ---------------------------------------------------------------------------


def test_tiling_assert_fires_on_malformed_doc():
    """An interior partial segment (too short) raises DerivedSegmentationError."""
    texts = ["abcdefgh", "ijklmnop"]
    segment_size = 4
    segs, _ = _load_segs(texts, TOK, segment_size)

    # Fabricate a bad DerivedDoc: (0,1) is an interior segment (should be segment_size tokens),
    # but we replace it with a fake SourceSegment that has only 2 tokens — interior short.
    # We cannot easily mutate segs, so instead construct a SourceSegment-patched segs clone.
    # Simpler: construct a DerivedDoc with refs pointing to a partial segment in the interior.
    #
    # The segment at (0, last_seg) might be partial (last doc segment).
    # For a malformed new_started=False doc: use (0, last_seg) as the FIRST ref
    # (interior position 0, non-last overall) so the tiling check fires.
    n_segs_0 = len(segs[0])
    assert n_segs_0 >= 2, "need ≥ 2 segs in doc 0"
    # (0, n_segs_0 - 1) = last seg of doc 0 (may be partial if text length % segment_size != 0)
    # Put it first (non-last overall), then add (0, 0) as second — this creates an
    # interior partial if the last seg of doc 0 has < segment_size tokens.
    partial_ref = (0, n_segs_0 - 1)
    partial_len = len(segs[0][n_segs_0 - 1].tokens)
    # Precondition (mirrors the sibling char-offset test): the last seg must be a
    # PARTIAL segment (< segment_size) for it to violate the interior-tiling rule
    # when placed non-last. Otherwise this test would be vacuous.
    assert partial_len < segment_size, "last seg must be partial for this fixture"

    # Partial last seg confirmed; use (partial, full) as a new_started=False interior.
    full_ref = (0, 0)  # first seg of doc 0 is always full (has segment_size tokens)
    # new_started=False: put the partial in the interior (non-last) position → fires.
    bad_refs = (partial_ref, full_ref)  # partial seg BEFORE a full seg → interior short
    bad_doc = DerivedDoc(bad_refs, False)
    docs = (bad_doc,)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = _base_meta(segment_size=segment_size)
    with pytest.raises(DerivedSegmentationError):
        build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)


# ---------------------------------------------------------------------------
# 5. Parquet text is BOS-free
# ---------------------------------------------------------------------------


def test_parquet_text_has_no_bos_marker():
    """new-started docs' parquet text must not contain the BOS token literal."""
    # BOS id = 500; for CharMergeTokenizer, decode skips BOS, so text never contains it.
    # For new_started=True, _piece_ids prepends BOS; to_text strips it.
    texts = ["alpha beta"]
    segment_size = 4
    segs, _ = _load_segs(texts, TOK, segment_size)
    n_segs = len(segs[0])
    assert n_segs >= 2, f"need ≥ 2 segs; got {n_segs}"

    new_refs = ((0, 1),)  # single interior full-segment new-started doc
    docs = (DerivedDoc(new_refs, True),)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = _base_meta(segment_size=segment_size)
    _, derived_texts = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)

    assert len(derived_texts) == 1
    text = derived_texts[0]
    # The BOS token has id 500; chr(500) is the Unicode character "ǴI-like"; it should not
    # appear literally in the decoded text (BOS is stripped by to_text before decoding).
    bos_char = chr(TOK.bos_id)  # type: ignore[arg-type]  # ty: ignore[invalid-argument-type]
    assert bos_char not in text, f"BOS literal found in parquet text: {text!r}"
    # Also verify the text is the decode of the segment body (without synthetic BOS).
    expected_body = tuple(segs[0][1].tokens)
    expected_text = TOK.decode(list(expected_body))
    assert text == expected_text


# ---------------------------------------------------------------------------
# 6. CSR / dtype / metadata invariants
# ---------------------------------------------------------------------------


def _build_shard_with_loop(tmp_path, segment_size=LOOP_SEGMENT_SIZE):
    """Build a derived shard via recombination that produces a new_started doc."""
    texts = LOOP_CORPUS
    segs, base_shard = _load_segs(texts, TOK, segment_size)
    result = recombine_segments(
        segs, TOK, cut_fraction=0.5,
        shard_seed=recombination_seed(LOOP_SEED, 7, 0.5),
        shard_name="test_loop", limit=2048,
    )
    assert result.n_loops == 1, "fixture requires exactly one loop"
    meta = _base_meta(segment_size=segment_size)
    shard, texts_out = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)
    return shard, texts_out, segs


def test_csr_offsets_monotone_nondecreasing():
    """offsets must be non-decreasing (CSR invariant)."""
    shard, _, _ = _build_shard_with_loop(None)
    offs = shard["offsets"]
    assert bool((offs[1:] >= offs[:-1]).all()), "CSR offsets not non-decreasing"


def test_csr_lengths_consistent():
    """offsets[-1] must equal feature_ids.numel()."""
    shard, _, _ = _build_shard_with_loop(None)
    assert int(shard["offsets"][-1]) == int(shard["feature_ids"].numel())


def test_dtype_contract():
    """Derived shard dtypes must match save_features expectations."""
    shard, _, _ = _build_shard_with_loop(None)
    assert shard["offsets"].dtype == t.int64
    assert shard["feature_ids"].dtype == t.int32
    assert shard["feature_mags"].dtype == t.float32
    assert shard["doc_ids"].dtype == t.int32
    assert shard["segment_indices"].dtype == t.int32
    assert shard["segment_positions"].dtype == t.int32
    assert shard["segment_char_starts"].dtype == t.int32
    assert shard["segment_char_ends"].dtype == t.int32
    assert shard["doc_token_counts"].dtype == t.int32


def test_metadata_propagated_from_base_meta():
    """Scalar metadata fields are copied from base_meta into the derived shard."""
    shard, _, _ = _build_shard_with_loop(None)
    assert shard["layer"] == 8
    assert shard["context_window"] == 2048
    assert shard["pooling"] == "max_pool"
    assert shard["segment_size"] == LOOP_SEGMENT_SIZE
    assert shard["context_independent_features"] is False
    assert shard["ignore_bos_features"] is True


def test_l0_tier_metadata_preserved_through_save_load_roundtrip(tmp_path):
    """GemmaScope-2 tier metadata must stay visible to downstream shard readers."""
    segs, _ = _load_segs(LOOP_CORPUS, TOK, LOOP_SEGMENT_SIZE)
    result = recombine_segments(
        segs, TOK, cut_fraction=0.5,
        shard_seed=recombination_seed(LOOP_SEED, 7, 0.5),
        shard_name="test_loop", limit=2048,
    )
    meta = _base_meta(segment_size=LOOP_SEGMENT_SIZE)
    meta["l0_tier"] = "medium"

    shard, _ = build_derived_shard(result, segs, meta, TOK, segment_size=LOOP_SEGMENT_SIZE)
    assert shard["l0_tier"] == "medium"

    path = save_features(shard, tmp_path / "derived_l0")
    assert read_shard_metadata(path)["l0_tier"] == "medium"
    assert load_features(path)["l0_tier"] == "medium"


def test_doc_ids_monotone_nondecreasing():
    """doc_ids must be non-decreasing (segments sorted by doc)."""
    shard, _, _ = _build_shard_with_loop(None)
    doc_ids = shard["doc_ids"]
    assert bool((doc_ids[1:] >= doc_ids[:-1]).all()), "doc_ids not non-decreasing"


def test_segment_indices_zero_to_k_minus_one_per_doc():
    """segment_indices for each doc must be exactly 0..k-1."""
    shard, _, _ = _build_shard_with_loop(None)
    n_docs = int(shard["n_docs"])
    doc_ids_t = shard["doc_ids"]
    seg_indices_t = shard["segment_indices"]
    for d in range(n_docs):
        mask = doc_ids_t == d
        indices = seg_indices_t[mask].tolist()
        assert indices == list(range(len(indices))), (
            f"doc {d} segment_indices are not 0..{len(indices)-1}: {indices}"
        )


def test_single_partial_remainder_doc_built_correctly():
    """A new-started doc with a single partial segment builds correctly.

    Overflow remainders can be a single partial source segment that is NOT the
    first segment of its source doc (engine invariant: loops/remainders start at
    non-root refs, i.e. seg ordinal ≥ 1). Verify:
    - tok_boundaries = [0, 1+partial_len]
    - char_offsets computed correctly
    - no tiling assert fires
    - text is non-empty and round-trip stable
    """
    # "abcde" with BOS=500 and segment_size=4:
    # kept_ids = [500, 97, 98, 99, 100, 101] (6 tokens)
    # seg 0 = [500, 97, 98, 99] (4 tokens, full)
    # seg 1 = [100, 101]         (2 tokens, partial last segment)
    texts = ["abcde"]
    segment_size = 4
    segs, _ = _load_segs(texts, TOK, segment_size)
    assert len(segs[0]) == 2, f"expected 2 segments, got {len(segs[0])}"

    # Use seg 1 (the partial last segment) as the single ref for a new_started doc.
    # Seg 1 tokens are the INTERIOR tokens of the original doc (no BOS), so prepending
    # a synthetic BOS gives a valid BOS-led doc without double-BOS.
    partial_ref = (0, 1)
    partial_len = len(segs[0][1].tokens)  # 2 tokens (e, d → [100, 101])
    assert partial_len < segment_size, "last seg must be partial for this fixture"

    docs = (DerivedDoc((partial_ref,), True),)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = _base_meta(segment_size=segment_size)
    shard, texts_out = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)

    assert int(shard["n_docs"]) == 1
    assert int(shard["n_segments"]) == 1
    assert shard["segment_indices"].tolist() == [0]

    # token count: 1 (synthetic BOS) + partial_len
    expected_toks = 1 + partial_len
    assert int(shard["doc_token_counts"][0]) == expected_toks

    # text is non-empty and round-trip stable
    text = texts_out[0]
    assert len(text) > 0

    # Build spliced_ids manually to verify round-trip.
    # Seg 1 tokens have no BOS (interior segment); synthetic BOS + those tokens is well-formed.
    seg1_toks = segs[0][1].tokens
    spliced_ids = (500,) + seg1_toks  # synthetic BOS + interior tokens
    assert is_roundtrip_stable(spliced_ids, TOK), (
        f"single-partial new-started doc not round-trip stable: {spliced_ids}"
    )


# ---------------------------------------------------------------------------
# 7. End-to-end splice_feature_shard smoke
# ---------------------------------------------------------------------------


def test_splice_feature_shard_smoke_no_placeholder_raise(tmp_path):
    """splice_feature_shard completes without SpliceInternalError on new-started docs.

    Uses LOOP_CORPUS + LOOP_SEED which is known to produce a loop → new-started doc.
    Verifies:
    - function returns without exception
    - derived shard has valid doc/seg counts
    - texts round-trip stable
    - features preserved verbatim from source (via provenance tracing)
    """
    texts = LOOP_CORPUS
    segment_size = LOOP_SEGMENT_SIZE
    context_window = 2048

    # Save base shard and parquet.
    base_shard_dict = make_base_shard(texts, TOK, segment_size, context_window=context_window)
    base_path = save_features(base_shard_dict, tmp_path / "shard_00007")
    parquet_path = tmp_path / "shard_00007.parquet"
    atomic_write_parquet(pa.table({"text": texts}), parquet_path)

    derived, derived_texts, provenance = splice_feature_shard(
        base_path, parquet_path, TOK,
        model="gemma-2-2b",  # model string doesn't affect CPU-only splice
        cut_fraction=0.5,
        seed=LOOP_SEED,
        shard_index=7,
        shard_name="shard_00007",
        segment_size=segment_size,
        context_window=context_window,
        sae_max_context=None,
    )

    # Basic structural checks.
    n_docs = int(derived["n_docs"])
    n_segs = int(derived["n_segments"])
    assert n_docs >= 1
    assert n_segs >= 1
    assert len(derived_texts) == n_docs

    # All derived texts round-trip stably and have length matching doc_token_counts.
    for i, text in enumerate(derived_texts):
        # Text is BOS-free; re-encoding adds BOS back automatically.
        re_encoded_ids = TOK(text)["input_ids"]
        # Check full round-trip stability (BOS + body tokens).
        assert is_roundtrip_stable(tuple(re_encoded_ids), TOK), (
            f"doc {i} not round-trip stable: {text!r}"
        )
        # Length parity: re-encoded token count must equal the stored doc_token_counts[i].
        # This proves the derived shard records the actual token length, not a placeholder.
        stored_tok_count = int(derived["doc_token_counts"][i])
        assert len(re_encoded_ids) == stored_tok_count, (
            f"doc {i} length mismatch: re-encoded {len(re_encoded_ids)} tokens "
            f"but doc_token_counts[{i}]={stored_tok_count}"
        )

    # Derived dir contains at least one new-started doc (the loop).
    # A doc is new-started if its first segment is interior (src_seg_idx > 0).
    assert any(
        (d["provenance"][0][2] > 0 if d["provenance"] else False)
        for d in provenance["docs"]
    ), "at least one new-started doc"

    # Feature preservation: load base, then verify each derived segment's features
    # match the provenance-pointed source segment.
    base_shard_loaded = load_features(base_path)
    base_offsets = base_shard_loaded["offsets"]
    base_fids = base_shard_loaded["feature_ids"]
    base_fmags = base_shard_loaded["feature_mags"]

    derived_offsets = derived["offsets"]
    derived_fids = derived["feature_ids"]
    derived_fmags = derived["feature_mags"]

    for doc_meta in provenance["docs"]:
        for seg_k, (_, src_doc_idx, src_seg_idx) in enumerate(doc_meta["provenance"]):
            # Find source segment global row in base shard.
            src_global = -1
            doc_ids_base = base_shard_loaded["doc_ids"]
            seg_indices_base = base_shard_loaded["segment_indices"]
            mask = (doc_ids_base == src_doc_idx) & (seg_indices_base == src_seg_idx)
            matches = mask.nonzero(as_tuple=True)[0]
            assert len(matches) == 1, f"provenance ({src_doc_idx},{src_seg_idx}) not unique"
            src_global = int(matches[0])

            src_lo = int(base_offsets[src_global])
            src_hi = int(base_offsets[src_global + 1])
            src_fids = base_fids[src_lo:src_hi]
            src_fmags = base_fmags[src_lo:src_hi]

            # Find derived global row for (doc_i, seg_k).
            doc_i = doc_meta["doc_id"]
            d_doc_ids = derived["doc_ids"]
            d_seg_indices = derived["segment_indices"]
            d_mask = (d_doc_ids == doc_i) & (d_seg_indices == seg_k)
            d_matches = d_mask.nonzero(as_tuple=True)[0]
            assert len(d_matches) == 1, f"derived ({doc_i},{seg_k}) not unique"
            d_global = int(d_matches[0])

            d_lo = int(derived_offsets[d_global])
            d_hi = int(derived_offsets[d_global + 1])
            d_fids = derived_fids[d_lo:d_hi]
            d_fmags = derived_fmags[d_lo:d_hi]

            assert t.equal(d_fids, src_fids), (
                f"feature_ids mismatch at provenance ({src_doc_idx},{src_seg_idx})"
            )
            assert t.equal(d_fmags, src_fmags), (
                f"feature_mags mismatch at provenance ({src_doc_idx},{src_seg_idx})"
            )


# ---------------------------------------------------------------------------
# 8. FIX-2: Derived shard must propagate all base scalar metadata
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", [None, "nnterp", "native", "transformerlens"])
def test_derived_shard_preserves_l0_tier_and_backend_through_disk(tmp_path, backend):
    """Provenance survives real base loading, splicing, and derived serialization."""
    texts = ["abcdef", "ghijkl"]
    segment_size = 4

    # Build base shard with l0_tier stamped
    base_dict = make_base_shard(texts, TOK, segment_size)
    base_dict["l0_tier"] = "medium"  # valid L0Tier value as string
    if backend is not None:
        base_dict["extraction_backend"] = backend
    base_path = save_features(base_dict, tmp_path / "base")
    base_loaded = load_features(base_path)
    assert base_loaded.get("l0_tier") == "medium", "sanity: l0_tier survives save/load of base"

    segs, _ = load_source_segments(
        base_loaded, texts, TOK, segment_size=segment_size,
        context_window=2048, sae_max_context=None, shard_name="test",
    )
    docs = tuple(
        DerivedDoc(tuple((d, j) for j in range(len(segs[d]))), False)
        for d in range(len(texts))
    )
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    derived, _ = build_derived_shard(result, segs, base_loaded, TOK, segment_size=segment_size)

    # Save and reload the derived shard through the real pipeline
    derived_path = save_features(derived, tmp_path / "derived")
    derived_loaded = load_features(derived_path)
    assert derived_loaded.get("l0_tier") == "medium", (
        f"l0_tier not preserved in derived shard: {derived_loaded.get('l0_tier')!r}"
    )
    assert derived_loaded.get("extraction_backend") == backend
    assert read_shard_metadata(derived_path).get("extraction_backend") == backend
    if backend is None:
        assert "extraction_backend" not in derived_loaded


def test_source_segment_views_do_not_mutate_base_and_output_is_independent():
    """SourceSegment feature slices can be views without aliasing derived output."""
    texts = ["alpha beta gamma delta", "epsilon zeta eta theta"]
    segment_size = 3
    base = make_base_shard(texts, TOK, segment_size)
    base_before = {
        key: value.clone()
        for key, value in base.items()
        if isinstance(value, t.Tensor)
    }

    segs, _ = load_source_segments(
        base,
        texts,
        TOK,
        segment_size=segment_size,
        context_window=2048,
        sae_max_context=None,
        shard_name="test",
    )
    docs = tuple(
        DerivedDoc(tuple((d, j) for j in range(len(segs[d]))), False)
        for d in range(len(segs))
    )
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    derived, _ = build_derived_shard(result, segs, base, TOK, segment_size=segment_size)

    for key, expected in base_before.items():
        assert t.equal(base[key], expected), f"base tensor mutated: {key}"
    assert derived["feature_ids"].data_ptr() != base["feature_ids"].data_ptr()
    assert derived["feature_mags"].data_ptr() != base["feature_mags"].data_ptr()


def test_derived_shard_propagates_arbitrary_extra_scalar_key():
    """Arbitrary scalar key in base_meta appears in the in-memory derived shard."""
    texts = ["abc"]
    segment_size = 2
    segs, _ = _load_segs(texts, TOK, segment_size)
    meta = {**_base_meta(segment_size=segment_size), "my_custom_key": "my_value"}
    docs = (DerivedDoc(tuple((0, j) for j in range(len(segs[0]))), False),)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    derived, _ = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)
    assert derived.get("my_custom_key") == "my_value", (
        f"arbitrary extra key lost from derived shard: {derived.get('my_custom_key')!r}"
    )


def test_derived_shard_drops_overlay_checksum_metadata():
    """Overlay checksum keys (percentile_checksum, tokenizer_checksum) do not
    propagate to derived shards. These checksums describe the base shard's overlay
    bytes and are invalid on a freshly built derived shard.

    Verifies: base_meta carrying checksum keys + custom scalar key → derived shard
    preserves custom key and l0_tier but has NO checksum keys.
    """
    texts = ["alpha beta gamma"]
    segment_size = 3
    segs, _ = _load_segs(texts, TOK, segment_size)

    # Build base_meta with checksums + custom key
    meta = {
        **_base_meta(segment_size=segment_size),
        "percentile_checksum": "abc123def456",
        "tokenizer_checksum": "xyz789uvw012",
        "l0_tier": "high",
        "my_custom_metadata": 42,
    }
    docs = (DerivedDoc(tuple((0, j) for j in range(len(segs[0]))), False),)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    derived, _ = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)

    # Checksum keys must be dropped
    assert "percentile_checksum" not in derived, (
        "percentile_checksum should not propagate to derived shard"
    )
    assert "tokenizer_checksum" not in derived, (
        "tokenizer_checksum should not propagate to derived shard"
    )

    # Custom keys must be preserved
    assert derived.get("l0_tier") == "high", (
        f"l0_tier not preserved in derived shard: {derived.get('l0_tier')!r}"
    )
    assert derived.get("my_custom_metadata") == 42, (
        f"custom metadata not preserved in derived shard: {derived.get('my_custom_metadata')!r}"
    )


def test_derived_n_docs_reflects_derived_not_base():
    """n_docs in derived shard reflects the derived result count, not the base count."""
    texts = ["abcde", "fghij"]  # 2 base docs
    segment_size = 2
    segs, _ = _load_segs(texts, TOK, segment_size)
    # Build derived shard with only 1 doc (first doc only)
    docs = (DerivedDoc(tuple((0, j) for j in range(len(segs[0]))), False),)
    result = ShardSpliceResult(docs=docs, n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0)
    meta = {**_base_meta(segment_size=segment_size), "n_docs": 999}  # wrong n_docs in meta
    derived, _ = build_derived_shard(result, segs, meta, TOK, segment_size=segment_size)
    assert derived["n_docs"] == 1, (
        f"n_docs should be 1 (derived), not base value 999: got {derived['n_docs']}"
    )
