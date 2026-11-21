"""End-to-end integration tests for the ``ignore_bos_features`` feature.

This is the real runtime correctness gate for ``ignore_bos_features``: it
drives the production :func:`steering.extract_features.process_shard` against
the *existing* tiny stub model/SAE/tokenizer harness (the same one used by
``tests/steering/test_context_independent_extraction.py``) so the assertions
exercise the full pooling-mask code path, not a unit-level stand-in.

Harness reused (imported, not re-invented):
  * ``_tiny_model_sae`` — builds ``TinyHookedTransformer`` + ``TinySAE``.
    The tokenizer prepends a fixed BOS id of ``0`` (``_encode`` returns
    ``[0, *chars]``) and ``tokenizer.bos_token_id == 0``. The model is
    *positionless* (per-token embeddings, no attention / positional encoding)
    and the SAE is ``relu(x @ W_enc + b_enc)`` with random non-zero weights,
    so every distinct token id yields a deterministic, generally non-zero
    activation row. Dropping the ``<bos>``-id token from a >1-token segment
    therefore changes the max/mean pool deterministically.
  * ``_make_parquet`` — writes ``pa.table({"text": texts})``.
  * ``_make_stats`` — minimal ``ExtractionStatistics`` so a stats file exists
    for ``verify_feature_dir`` (it does NOT set ``ignore_bos_features``, so the
    guarded stats-level check is skipped by design — back-compat — and the
    ``no_bos`` path leaf is validated against the *shard* metadata that
    ``process_shard`` writes).

Why the BOS-id token has a real effect under this stub: the BOS id is ``0``;
the stub embedding row ``_embedding[0]`` is a random vector, its SAE row is
``relu(emb @ W_enc + b_enc)`` (generally non-zero across the 128 features).
With ``segment_size`` such that segment 0 is ``[0, c1, c2, c3]``, max/mean
pooling over 4 rows differs from pooling over the 3 non-BOS rows unless the
BOS row is element-wise dominated everywhere, which is statistically
impossible across 128 random features. Hence test #2's seg-0 difference is a
genuine consequence of the feature working, and test #1's byte-identity is a
genuine consequence of ``keep_mask=None`` ⇒ legacy path.
"""

from __future__ import annotations

import dataclasses
import glob

import pytest
import torch as t

import steering.extract_features as ef_mod
from errors import BosOnlySegmentError
from steering.extract_features import (
    ShardResult,
    extract_document_features,
    find_optimal_batch_sizes,
    process_shard,
)
from steering.feature_io import load_features
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
    verify_feature_dir,
)
from steering.sae_resolver import (
    SAE_WIDTH_TO_NUM_FEATURES,
    ActivationType,
    SAEWidth,
)
from steering.segmentation import SegmentMode
from steering.stats_io import save_statistics

# Reuse the EXACT existing stub harness — do not re-invent.
from tests.steering.test_context_independent_extraction import (
    _make_parquet,
    _tiny_model_sae,
)
from tests.steering.test_feature_dir_verification import _make_stats

# Shared extraction config — kept identical across on/off runs so the only
# variable is the ``ignore_bos_features`` flag.
_LAYER = 8
_CONTEXT_WINDOW = 128
_POOLING = PoolingMode.max_pool
_SEGMENT_SIZE = 4
_GPU_BS = 2
_CPU_BS = 16
_WIDTH = SAEWidth("16k")  # SAEWidth.W_16K
_ACT = ActivationType.residual
_MODEL = "gemma-2-2b"


def _shard_path(directory) -> str:
    """Return the single written compressed shard under ``directory``."""
    hits = glob.glob(str(directory / "*.safetensors.zst"))
    assert len(hits) == 1, f"expected exactly one shard in {directory}, got {hits}"
    return hits[0]


def _run(
    model,
    sae,
    parquet_path,
    out_dir,
    *,
    ignore_bos,
    omit_flag=False,
    dataset_phase=None,
):
    """Drive process_shard with the shared config; return loaded shard dict.

    ``model``/``sae`` are passed in (not built here): ``_tiny_model_sae`` seeds
    its weights randomly *per call*, so an on-vs-off comparison MUST reuse the
    same model/SAE instances or the activations differ for reasons unrelated to
    the flag. (This mirrors how the existing harness tests, e.g.
    ``test_context_independent_pipeline_post_false_matches_true``, share one
    ``model, sae`` pair across multiple ``process_shard`` calls.)
    """
    # Explicit literal kwargs (not **dict unpacking) so the static type
    # checker can match each argument to its declared parameter type — this
    # mirrors the call style of the existing harness tests. The only branch
    # is whether ``ignore_bos_features`` is passed at all (the "default
    # omitted" path for the byte-identity test).
    if omit_flag:
        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=_LAYER,
            output_dir=str(out_dir),
            context_window=_CONTEXT_WINDOW,
            pooling=_POOLING,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            gpu_batch_size=_GPU_BS,
            cpu_batch_size=_CPU_BS,
            dataset_phase=dataset_phase,
        )
    else:
        result = process_shard(
            parquet_path,
            model,
            sae,
            layer=_LAYER,
            output_dir=str(out_dir),
            context_window=_CONTEXT_WINDOW,
            pooling=_POOLING,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            gpu_batch_size=_GPU_BS,
            cpu_batch_size=_CPU_BS,
            dataset_phase=dataset_phase,
            ignore_bos_features=ignore_bos,
        )
    return load_features(result.output_path)


# ---------------------------------------------------------------------------
# Test 1 — flag-off byte-identity (golden): off == default-omitted
# ---------------------------------------------------------------------------


def test_flag_off_byte_identical_to_default(tmp_path):
    """``ignore_bos_features=False`` must be byte-identical to omitting it.

    Both paths feed ``keep_mask=None`` into pooling, so the produced shard
    tensors must be EXACTLY equal (not approximately). This pins
    "off ⇒ legacy / no behavioural change".
    """
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(
        parquet_path,
        [
            "hello world this is a longer document for pooling",
            "another document with several tokens here too",
            "third doc",
        ],
    )

    off_dir = tmp_path / "off"
    off_dir.mkdir()
    default_dir = tmp_path / "default"
    default_dir.mkdir()

    # ONE model/SAE pair shared across both runs (weights are random per
    # _tiny_model_sae() call) — so the only variable is the flag.
    model, sae = _tiny_model_sae()
    off = _run(model, sae, parquet_path, off_dir, ignore_bos=False)
    default = _run(
        model, sae, parquet_path, default_dir, ignore_bos=False, omit_flag=True
    )

    assert t.equal(off["feature_ids"], default["feature_ids"]), (
        "feature_ids differ between ignore_bos_features=False and omitted "
        "(flag-off must be byte-identical to legacy)"
    )
    assert t.equal(off["feature_mags"], default["feature_mags"]), (
        "feature_mags differ between ignore_bos_features=False and omitted"
    )
    assert t.equal(off["offsets"], default["offsets"]), (
        "offsets differ between ignore_bos_features=False and omitted"
    )
    # Sanity: legacy path leaves the flag falsey in shard metadata.
    assert off["ignore_bos_features"] is False
    assert default["ignore_bos_features"] is False


# ---------------------------------------------------------------------------
# Test 2 — flag-on differs at seg0 + metadata + verify_feature_dir passes
# ---------------------------------------------------------------------------


def test_flag_on_differs_at_seg0_and_dir_verifies(tmp_path):
    """``ignore_bos_features=True`` drops the BOS-id token from pooling.

    Asserts three things:
      1. The loaded shard records ``ignore_bos_features is True``.
      2. The pooled feature_mags DIFFER vs the flag-off run — segment 0 of a
         pretraining doc is ``[bos, c1, c2, c3]`` (segment_size=4 > 1), so
         dropping the BOS row changes the max pool. (We compare full-shard
         feature_mags; any difference is necessarily caused by seg0 since
         every other segment has no BOS-id token, so for them the keep-mask
         is all-True and pooling is unchanged.)
      3. ``verify_feature_dir`` does NOT raise on the canonical
         ``…/base/no_bos`` directory (built via ``build_feature_dir`` with the
         SAME params passed to ``process_shard``, plus a consistent stats
         file).
    """
    parquet_path = str(tmp_path / "shard.parquet")
    # Single doc whose seg0 (4 tokens) is [bos, h, e, l] — >1 token so the
    # BOS drop materially changes the seg0 max-pool.
    _make_parquet(parquet_path, ["hello world this is a longer pretraining document"])

    # ONE model/SAE pair shared across both runs so the only variable is the
    # flag (weights are random per _tiny_model_sae() call).
    model, sae = _tiny_model_sae()

    # Flag-OFF reference (plain dir; not validated, only used for the diff).
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    off = _run(model, sae, parquet_path, off_dir, ignore_bos=False)

    # Flag-ON into the canonical hierarchical .../base/no_bos directory.
    on_dir = build_feature_dir(
        root=tmp_path / "feats",
        segment_size=_SEGMENT_SIZE,
        layer=_LAYER,
        activation_type=_ACT,
        pooling=_POOLING,
        width=_WIDTH,
        model=_MODEL,
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=True,
    )
    on_dir.mkdir(parents=True, exist_ok=True)
    assert on_dir.name == "no_bos", on_dir
    on = _run(model, sae, parquet_path, on_dir, ignore_bos=True)

    # (1) shard metadata records the flag.
    assert on["ignore_bos_features"] is True

    # (2) on vs off feature tensors differ because seg0 lost its BOS token.
    assert not t.equal(on["feature_mags"], off["feature_mags"]), (
        "ignore_bos_features=True produced identical feature_mags to "
        "ignore_bos_features=False — the BOS-id token was NOT excluded from "
        "seg0 pooling (feature is not working end-to-end)."
    )
    # Segmentation-structure invariance (spec §7/§17): ignore_bos changes the
    # pooled feature *content* of <bos>-containing segments — so feature_ids,
    # feature_mags AND `offsets` (the CSR row-pointer = cumsum of per-segment
    # non-zero-feature counts) all legitimately move. What MUST stay
    # bit-identical to the flag-off run is the tokenization-derived
    # segmentation metadata that downstream token-boundary / packed-row
    # overlays consume: n_segments/n_docs, doc grouping, segment positions,
    # char bounds, and per-doc token counts. (`offsets` is feature content,
    # NOT segmentation structure — it is deliberately excluded here.)
    assert on["n_segments"] == off["n_segments"]
    assert on["n_docs"] == off["n_docs"]
    for k in (
        "doc_ids",
        "segment_indices",
        "segment_positions",
        "segment_char_starts",
        "segment_char_ends",
        "doc_token_counts",
    ):
        assert t.equal(on[k], off[k]), (
            f"segmentation array {k!r} changed under ignore_bos_features=True; "
            f"only pooled feature content (feature_ids/mags/offsets) may "
            f"differ (spec §7/§17). on={on[k]} off={off[k]}"
        )
    # Sanity: offsets DOES change (proves seg0 lost BOS-derived features) and
    # is internally consistent (last offset == #feature_ids).
    assert not t.equal(on["offsets"], off["offsets"]), (
        "offsets identical → ignore_bos did not change any segment's "
        "non-zero feature count (feature not working end-to-end)"
    )
    assert int(on["offsets"][-1].item()) == on["feature_ids"].shape[0]
    assert int(off["offsets"][-1].item()) == off["feature_ids"].shape[0]

    # (3) the directory must verify cleanly. verify_feature_dir requires a
    # stats file; build a CONSISTENT one. Stats files now always carry
    # ignore_bos_features (plain bool), so for a /no_bos directory the stats
    # MUST declare ignore_bos_features=True — exactly what a real
    # compute_statistics run stamps (the uniform value across the True
    # shards). verify_feature_dir cross-checks stats AND shard metadata
    # against the path leaf; an inconsistent stats file is (correctly) a hard
    # FeatureDirectoryMismatchError.
    stats = dataclasses.replace(
        _make_stats(
            segment_size=_SEGMENT_SIZE,
            layer=_LAYER,
            pooling_mode=_POOLING.value,
            n_sae_features=SAE_WIDTH_TO_NUM_FEATURES[_WIDTH],
            context_window=_CONTEXT_WINDOW,
        ),
        ignore_bos_features=True,
    )
    save_statistics(stats, on_dir / "feature_statistics")

    # Must not raise.
    verify_feature_dir(on_dir)


# ---------------------------------------------------------------------------
# Test 3 (Task 7.2) — segment_size=1 pretraining raises BosOnlySegmentError
# ---------------------------------------------------------------------------


def test_segment_size_1_pretraining_raises_bos_only(tmp_path):
    """With ``segment_size=1`` on a pretraining parquet, segment 0 is exactly
    ``[<bos>]``. Under ``ignore_bos_features=True`` the keep-mask for that
    segment is all-``False`` ⇒ ``BosOnlySegmentError``, which MUST propagate
    out of ``process_shard`` (there is no try/except around
    ``pool_segments_batched_gpu`` in the pretraining path).
    """
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(parquet_path, ["hello world", "second doc here"])

    model, sae = _tiny_model_sae()
    with pytest.raises(BosOnlySegmentError):
        process_shard(
            parquet_path,
            model,
            sae,
            layer=_LAYER,
            output_dir=str(tmp_path),
            context_window=_CONTEXT_WINDOW,
            pooling=_POOLING,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=1,
            gpu_batch_size=2,
            cpu_batch_size=2,
            ignore_bos_features=True,
        )


# ---------------------------------------------------------------------------
# Test 5 — sequential path (pipeline_post=False), non-CI, ignore_bos=True
# ---------------------------------------------------------------------------


def _run_seq(
    model,
    sae,
    parquet_path,
    out_dir,
    *,
    ignore_bos,
    context_independent=False,
):
    """Drive ``process_shard`` on the SEQUENTIAL path (``pipeline_post=False``).

    The shared ``_run`` helper at the top of this file deliberately does NOT
    expose ``pipeline_post`` (it only exercises the default pipelined path), so
    the sequential-path branch — the ``elif all_text_chunks:`` block in
    ``process_shard`` that builds the per-doc keep-mask at lines ~2908-2922 for
    non-CI and calls ``_run_context_independent_chunk(..., ignore_bos_features=
    ...)`` for CI — has zero coverage. This thin wrapper closes that gap while
    reusing the module's shared config constants (``_LAYER``/``_POOLING``/...).
    ``process_shard``'s signature (confirmed) accepts ``pipeline_post: bool =
    True`` and ``context_independent_features: bool = False``.
    """
    result = process_shard(
        parquet_path,
        model,
        sae,
        layer=_LAYER,
        output_dir=str(out_dir),
        context_window=_CONTEXT_WINDOW,
        pooling=_POOLING,
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=_SEGMENT_SIZE,
        gpu_batch_size=_GPU_BS,
        cpu_batch_size=_CPU_BS,
        pipeline_post=False,
        context_independent_features=context_independent,
        ignore_bos_features=ignore_bos,
    )
    return load_features(result.output_path)


# Segmentation arrays that MUST stay bit-identical between an ignore_bos-on and
# ignore_bos-off run — mirrors the strict structural-invariance set in
# ``test_flag_on_differs_at_seg0_and_dir_verifies``. ``offsets`` is feature
# *content* (CSR row pointer = cumsum of per-segment non-zero counts) and
# legitimately moves when seg0 drops BOS-derived features, so it is excluded.
_SEGMENTATION_INVARIANT_KEYS = (
    "doc_ids",
    "segment_indices",
    "segment_positions",
    "segment_char_starts",
    "segment_char_ends",
    "doc_token_counts",
)


def test_sequential_path_non_ci_ignore_bos_differs_at_seg0(tmp_path):
    """Sequential path (``pipeline_post=False``), non-CI, ``ignore_bos=True``.

    Pins the sequential-path non-CI keep-mask wiring (``process_shard``'s
    ``elif all_text_chunks:`` block, ~L2908-2922: ``doc_keep_mask = tensor(
    kept_ids_doc) != bos_token_id`` fed into ``pool_segments_batched_gpu``).

    Asserts (real behaviour, not smoke):
      1. the written shard records ``ignore_bos_features is True``;
      2. pooled ``feature_mags`` DIFFER vs a ``pipeline_post=False,
         ignore_bos=False`` run (seg0 of the pretraining doc is ``[bos,h,e,l]``;
         dropping the BOS row changes the max-pool — the only segment with a
         BOS-id token);
      3. every tokenization-derived segmentation array is bit-identical between
         the two runs (only pooled feature content may move) — the same strict
         invariant asserted for the pipelined path.
    """
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(parquet_path, ["hello world this is a longer pretraining document"])

    # ONE model/SAE pair shared across both runs (random weights per call) so
    # the flag is the only variable.
    model, sae = _tiny_model_sae()

    off_dir = tmp_path / "seq_off"
    off_dir.mkdir()
    on_dir = tmp_path / "seq_on"
    on_dir.mkdir()

    off = _run_seq(model, sae, parquet_path, off_dir, ignore_bos=False)
    on = _run_seq(model, sae, parquet_path, on_dir, ignore_bos=True)

    # (1) shard metadata records the flag (set on the sequential path too).
    assert on["ignore_bos_features"] is True
    assert off["ignore_bos_features"] is False

    # (2) pooled feature content differs because seg0 lost its BOS-id row.
    assert not t.equal(on["feature_mags"], off["feature_mags"]), (
        "sequential-path ignore_bos_features=True produced identical "
        "feature_mags to ignore_bos_features=False — the BOS-id token was NOT "
        "excluded from seg0 pooling on the pipeline_post=False path."
    )

    # (3) segmentation structure is bit-identical (spec §7/§17); only pooled
    # feature content (feature_ids/mags/offsets) may differ.
    assert on["n_segments"] == off["n_segments"]
    assert on["n_docs"] == off["n_docs"]
    for k in _SEGMENTATION_INVARIANT_KEYS:
        assert t.equal(on[k], off[k]), (
            f"segmentation array {k!r} changed under sequential-path "
            f"ignore_bos_features=True; only pooled feature content may differ "
            f"(spec §7/§17). on={on[k]} off={off[k]}"
        )
    # Sanity: offsets DID move (proves a segment's non-zero feature count
    # changed → the BOS drop actually took effect) and stays self-consistent.
    assert not t.equal(on["offsets"], off["offsets"]), (
        "offsets identical → sequential-path ignore_bos did not change any "
        "segment's non-zero feature count (feature not working end-to-end)"
    )
    assert int(on["offsets"][-1].item()) == on["feature_ids"].shape[0]
    assert int(off["offsets"][-1].item()) == off["feature_ids"].shape[0]


# ---------------------------------------------------------------------------
# Test 6 — sequential path, CI + ignore_bos
# ---------------------------------------------------------------------------


def test_sequential_path_ci_plus_ignore_bos_records_both_flags(tmp_path):
    """Sequential path, ``context_independent_features=True`` AND
    ``ignore_bos=True``.

    Covers the sequential-path CI branch's ``_run_context_independent_chunk(
    ..., ignore_bos_features=ignore_bos_features, ...)`` call (~L2821-2833).
    CI preconditions read from ``process_shard``: bos must resolve (stub
    ``bos_token_id=0``); ``segment_size+1`` (=5) must be ``<= model.cfg.n_ctx``
    (=2048); ``segment_size`` (=4) ``< context_window`` (=128) so no
    semantic-equivalence info-log path. The stub's ``_encode`` prepends id 0
    (== bos_token_id), so the loud pretraining seg-0-BOS assertion passes.

    Asserts the shard runs to completion AND records BOTH flags — and, as a
    real consequence (not just metadata), that CI+ignore_bos differs from a
    sequential CI-only run (seg0's natural BOS is dropped under CI).
    """
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(parquet_path, ["hello world this is a longer pretraining document"])

    model, sae = _tiny_model_sae()

    ci_only_dir = tmp_path / "seq_ci_only"
    ci_only_dir.mkdir()
    ci_bos_dir = tmp_path / "seq_ci_bos"
    ci_bos_dir.mkdir()

    ci_only = _run_seq(
        model, sae, parquet_path, ci_only_dir,
        ignore_bos=False, context_independent=True,
    )
    ci_bos = _run_seq(
        model, sae, parquet_path, ci_bos_dir,
        ignore_bos=True, context_independent=True,
    )

    # Records BOTH flags (sequential-path metadata wiring).
    assert ci_bos["ignore_bos_features"] is True
    assert ci_bos["context_independent_features"] is True
    assert ci_only["ignore_bos_features"] is False
    assert ci_only["context_independent_features"] is True

    # Real consequence: under CI seg0 is [bos, h, e, l] in its own forward;
    # ignore_bos drops the natural seg-0 BOS row → pooled content differs.
    assert not t.equal(ci_bos["feature_mags"], ci_only["feature_mags"]), (
        "sequential-path CI+ignore_bos produced identical feature_mags to "
        "CI-only — the natural seg-0 BOS was NOT dropped under CI on the "
        "pipeline_post=False path."
    )
    # Segmentation structure is still invariant under the flag.
    assert ci_bos["n_segments"] == ci_only["n_segments"]
    assert ci_bos["n_docs"] == ci_only["n_docs"]
    for k in _SEGMENTATION_INVARIANT_KEYS:
        assert t.equal(ci_bos[k], ci_only[k]), (
            f"segmentation array {k!r} changed under sequential-path "
            f"CI+ignore_bos. ci_bos={ci_bos[k]} ci_only={ci_only[k]}"
        )


# ---------------------------------------------------------------------------
# Test 7 — extract_document_features CI + ignore_bos (functional contrast)
# ---------------------------------------------------------------------------


def test_extract_document_features_ci_ignore_bos_drops_natural_seg0_bos():
    """``extract_document_features(..., context_independent_features=True,
    ignore_bos_features=True)`` drops the natural seg-0 BOS under CI.

    ``extract_document_features``'s signature (confirmed) returns
    ``(feat_ids, feat_mags, positions, char_starts, char_ends, n_tokens)``, so
    ``result[1][0]`` is segment 0's pooled sparse magnitudes.

    Functional (not smoke) assertion: for a pretraining-style doc whose seg0
    contains the natural BOS (stub ``_encode`` prepends id 0 == bos_token_id),
    CI+ignore_bos differs from CI-only at ``mags[0]`` — proving the natural
    seg-0 BOS row is excluded from pooling under CI (spec §3.2). ``mean_pool``
    is used for the contrast: dropping a contributing row is guaranteed to move
    the mean (max-pool could coincidentally tie). ``not t.equal`` also handles
    a sparse-length change gracefully (different non-zero counts ⇒ not equal).
    """
    model, sae = _tiny_model_sae()
    text = "hello world this is a longer pretraining document"

    ci_only = extract_document_features(
        model, sae, text,
        layer=_LAYER,
        context_window=_CONTEXT_WINDOW,
        pooling=PoolingMode.mean_pool,
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=_SEGMENT_SIZE,
        context_independent_features=True,
        ignore_bos_features=False,
    )
    ci_bos = extract_document_features(
        model, sae, text,
        layer=_LAYER,
        context_window=_CONTEXT_WINDOW,
        pooling=PoolingMode.mean_pool,
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=_SEGMENT_SIZE,
        context_independent_features=True,
        ignore_bos_features=True,
    )

    ci_only_mags = ci_only[1]
    ci_bos_mags = ci_bos[1]
    assert len(ci_only_mags) >= 1, "CI-only returned no segments"
    assert len(ci_bos_mags) >= 1, "CI+ignore_bos returned no segments"
    # Segment count itself is unchanged (only pooled content of the BOS seg).
    assert len(ci_bos_mags) == len(ci_only_mags)
    # seg0's pooled magnitudes MUST differ: the natural BOS row was dropped.
    assert not t.equal(ci_bos_mags[0], ci_only_mags[0]), (
        "extract_document_features CI+ignore_bos produced identical seg0 "
        "pooled magnitudes to CI-only — the natural seg-0 BOS row was NOT "
        "excluded from the mean pool under CI (spec §3.2)."
    )


# ---------------------------------------------------------------------------
# Test 8 — extract_document_features bos_token_id=None → ValueError
# ---------------------------------------------------------------------------


def test_extract_document_features_ignore_bos_no_bos_token_raises():
    """``extract_document_features(..., ignore_bos_features=True)`` must raise
    ``ValueError`` when the tokenizer defines no BOS id.

    Mirrors ``test_context_independent_extraction.py::test_missing_bos_token_
    raises``: the stub tokenizer is a ``MagicMock``, so nulling only
    ``bos_token_id`` is insufficient (``get_bos_token_id`` would auto-vivify a
    truthy Mock); both BOS contracts must be nulled so ``resolve_bos_token_id``
    genuinely returns ``None`` and the guard at ``extract_features.py`` ~L1858
    fires.
    """
    model, sae = _tiny_model_sae()
    model.tokenizer.bos_token_id = None
    model.tokenizer.get_bos_token_id = None
    with pytest.raises(ValueError, match="bos_token_id"):
        extract_document_features(
            model, sae, "hello world this is a document",
            layer=_LAYER,
            context_window=_CONTEXT_WINDOW,
            pooling=_POOLING,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            ignore_bos_features=True,
        )


# ---------------------------------------------------------------------------
# Test 9 — process_shard bos_token_id=None → ValueError
# ---------------------------------------------------------------------------


def test_process_shard_ignore_bos_no_bos_token_raises(tmp_path):
    """``process_shard(..., ignore_bos_features=True)`` must raise
    ``ValueError`` when the tokenizer defines no BOS id (guard at
    ``extract_features.py`` ~L2444-2450 — shared with
    ``context_independent_features``).

    Same MagicMock dual-nulling as test 8 / the existing CI loud-failure test.
    """
    model, sae = _tiny_model_sae()
    model.tokenizer.bos_token_id = None
    model.tokenizer.get_bos_token_id = None
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(parquet_path, ["hello world"])
    with pytest.raises(ValueError, match="bos_token_id"):
        process_shard(
            parquet_path,
            model,
            sae,
            layer=_LAYER,
            output_dir=str(tmp_path),
            context_window=_CONTEXT_WINDOW,
            pooling=_POOLING,
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_SEGMENT_SIZE,
            gpu_batch_size=1,
            cpu_batch_size=1,
            ignore_bos_features=True,
        )


# ---------------------------------------------------------------------------
# Test 10 — find_optimal_batch_sizes forwards ignore_bos_features
# ---------------------------------------------------------------------------


def test_find_optimal_batch_sizes_forwards_ignore_bos_features(monkeypatch):
    """``find_optimal_batch_sizes`` must forward ``ignore_bos_features`` into
    every profiled ``process_shard`` call (~L1462-1481).

    The real grid search runs many full ``process_shard`` passes (expensive),
    so ``steering.extract_features.process_shard`` is monkeypatched with a stub
    that records ``kwargs["ignore_bos_features"]`` and returns a minimal valid
    ``ShardResult`` (fields confirmed from ``extract_features.py``:
    ``n_docs, n_tokens_gpt2, n_total_features, elapsed_seconds, output_path,
    bytes_on_disk``). Bounds are pinned to ``min==max==1`` for both gpu/cpu so
    exactly one (1, 1) pair is profiled with ``n_trials=1`` ⇒ exactly one
    ``process_shard`` invocation. ``find_optimal_batch_sizes``'s signature
    (confirmed) takes ``ignore_bos_features: bool = False`` as the last kwarg
    and accepts ``min/max_{gpu,cpu}_batch_size`` + ``n_trials``.

    Asserts the captured kwarg is ``True`` (genuine forwarding, not metadata
    theater) AND that the function returned the pinned ``(1, 1)`` pair.
    """
    captured: list[bool] = []

    def _stub_process_shard(*args, **kwargs):
        captured.append(kwargs["ignore_bos_features"])
        return ShardResult(
            n_docs=1,
            n_tokens_gpt2=1,
            n_total_features=1,
            elapsed_seconds=0.001,
            output_path=str(kwargs["output_dir"]),
            bytes_on_disk=0,
        )

    monkeypatch.setattr(ef_mod, "process_shard", _stub_process_shard)

    model, sae = _tiny_model_sae()
    best = find_optimal_batch_sizes(
        model,
        sae,
        layer=_LAYER,
        context_window=_CONTEXT_WINDOW,
        min_gpu_batch_size=1,
        max_gpu_batch_size=1,
        min_cpu_batch_size=1,
        max_cpu_batch_size=1,
        n_trials=1,
        n_docs=2,
        pooling=_POOLING,
        segment_size=_SEGMENT_SIZE,
        ignore_bos_features=True,
    )

    assert captured, (
        "find_optimal_batch_sizes never called process_shard — the (1,1) pair "
        "was not profiled (bounds/precondition regression)"
    )
    assert all(flag is True for flag in captured), (
        f"find_optimal_batch_sizes did NOT forward ignore_bos_features=True "
        f"into process_shard; captured={captured}"
    )
    assert best == (1, 1), (
        f"expected the single profiled (gpu,cpu)=(1,1) pair, got {best}"
    )
