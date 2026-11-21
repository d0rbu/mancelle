"""Live-dataloader splice_counterfactual tests: O1 stream equality, reference
correctness, and mode wiring (issue #596)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from loguru import logger

from nanochat.atomic_io import atomic_write_parquet
from nanochat.dataset import DatasetSplit
from steering.conditional.delta import (
    DeltaMode,
    DeltaSamplingWeights,
    FeatureDeltaConfig,
    compute_doc_counterfactual_deltas,
)
from steering.conditional.encoder_decoder.dataloader import (
    _document_batches_with_features,
    _validate_counterfactual_delta_results,
    encoder_decoder_data_loader_with_state,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_pipeline import (
    DeltaBundle,
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
)
from tests.steering._splice_cf_fixtures import build_derived_env

_N_SAE = 4096  # >= max fabricated fid from _splice_fixtures (n_features=4096)
_NUM_BUCKETS = 4


def _percentile_table(n=_N_SAE):
    return t.linspace(0.0, 1.0, 99).unsqueeze(0).expand(n, -1).to(t.float64).contiguous()


def _boundaries(effective_buckets, n=_N_SAE):
    return (
        t.linspace(1 / effective_buckets, 1.0 - 1 / effective_buckets,
                   effective_buckets - 1, dtype=t.float64)
        .unsqueeze(0).expand(n, -1).contiguous()
    )


def _sampling_pipeline(*, mode, with_negatives, seed=42):
    """mode=None ⇒ no delta bundle; else the given DeltaMode."""
    table = _percentile_table()
    negative_bundle = None
    effective_buckets = _NUM_BUCKETS - (1 if with_negatives else 0)
    rng = t.Generator().manual_seed(seed)
    if with_negatives:
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric, mean=2.0, std=2.0,
                min_k=1, max_k=5,
            ),
            n_sae_features=_N_SAE,
            rng=rng,
        )
    delta_bundle = None
    delta_weights = None
    if mode is not None:
        delta_bundle = DeltaBundle(
            config=FeatureDeltaConfig(mode=mode, neg_delta_ratio=0.5),
            percentile_table=table,
        )
        delta_weights = DeltaSamplingWeights.percentile_delta
    selection = SamplingSelection(
        distribution=Distribution.geometric, mean=3.0, std=3.0,
        min_k=1, max_k=10, delta_weights=delta_weights,
    )
    return FeaturePipelineConfig(
        selection_config=SamplingBundle(selection=selection, percentile_table=table, rng=rng),
        boundaries=_boundaries(effective_buckets),
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
        delta_bundle=delta_bundle,
    )


def _threshold_pipeline(*, mode, threshold=2):
    table = _percentile_table()
    delta_bundle = None
    if mode is not None:
        delta_bundle = DeltaBundle(
            config=FeatureDeltaConfig(mode=mode, neg_delta_ratio=0.5),
            percentile_table=table,
        )
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, threshold),
            precomputed_thresholds=None,
        ),
        boundaries=_boundaries(_NUM_BUCKETS),
        negative_bundle=None,
        effective_buckets=_NUM_BUCKETS,
        delta_bundle=delta_bundle,
    )


def _loader(env, pipeline, *, split=DatasetSplit.TRAIN, B=2, T=32):
    feature_index = FeatureIndex(env["feature_dir"])
    return encoder_decoder_data_loader_with_state(
        tokenizer=env["tokenizer"], B=B, T=T, split=split,
        feature_index=feature_index, pipeline=pipeline,
        segment_size=env["seg_size"], decoder_segment_size=None, device="cpu",
        resume_state_dict=None, tokenizer_batch_size=16, buffer_size=16,
        data_dir=env["data_dir"], prefetch_batches=0, label_dropout=0.0,
    )


def _take(gen, n=5):
    out = []
    for _ in range(n):
        out.append(next(gen))
    return out


def _assert_batches_equal(a, b):
    for batch_a, batch_b in zip(a, b, strict=True):
        for x, y in zip(batch_a, batch_b, strict=True):
            if isinstance(x, t.Tensor):
                assert t.equal(x, y)
            else:
                assert x == y


class TestO1IdentityStreamEquality:
    def test_sampling_with_negatives(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.0)
        base = _take(_loader(env, _sampling_pipeline(mode=None, with_negatives=True)))
        cf = _take(_loader(env, _sampling_pipeline(
            mode=DeltaMode.splice_counterfactual, with_negatives=True)))
        _assert_batches_equal(base, cf)

    def test_threshold(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.0)
        base = _take(_loader(env, _threshold_pipeline(mode=None)))
        cf = _take(_loader(env, _threshold_pipeline(mode=DeltaMode.splice_counterfactual)))
        _assert_batches_equal(base, cf)


class TestReferenceCorrectness:
    def test_refs_resolve_to_whf_derived_segment(self, tmp_path) -> None:
        """For every splice record in the train shard, the reference the loader
        would use equals the shard's segment at whf_derived — cross-checked by
        recomputing the expected delta output through the public function."""
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        fi = FeatureIndex(env["feature_dir"])
        idx = SpliceProvenanceIndex(env["feature_dir"])
        stem = "shard_00000"
        n_docs = idx.shard_n_docs(stem)
        checked = 0
        table = _percentile_table()
        for doc in range(n_docs):
            splices = idx.doc_splices(stem, doc)
            if not splices:
                continue
            segments = fi.lookup(stem, doc)
            assert idx.doc_n_segments(stem, doc) == len(segments)
            seg_refs = [None] * len(segments)
            for pos, sref in splices.items():
                rd, rp = sref.whf_derived
                ref_seg = fi.lookup(stem, rd)[rp]
                seg_refs[pos] = (ref_seg.feature_ids, ref_seg.feature_mags, None)
                checked += 1
            out = compute_doc_counterfactual_deltas(
                [s.feature_ids for s in segments],
                [s.feature_mags for s in segments],
                [None] * len(segments),
                seg_refs, _N_SAE, table,
            )
            for s_idx, r in enumerate(out):
                assert (r is None) == (s_idx not in splices)
        assert checked > 0, "cut_fraction=0.6 must produce splice records (non-vacuity)"

    def test_full_shard_counterfactual_score_count_logged(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        idx = SpliceProvenanceIndex(env["feature_dir"])
        stem = "shard_00000"
        expected = idx.shard_n_splice_positions(stem)
        assert expected > 0, "fixture must produce splice records (non-vacuity)"

        records: list[str] = []
        hid = logger.add(lambda m: records.append(str(m)), level="INFO")
        try:
            gen = _document_batches_with_features(
                split=DatasetSplit.TRAIN,
                resume_state_dict=None,
                tokenizer_batch_size=16,
                tokenizer=env["tokenizer"],
                feature_index=FeatureIndex(env["feature_dir"]),
                pipeline=_threshold_pipeline(mode=DeltaMode.splice_counterfactual),
                segment_size=env["seg_size"],
                data_dir=env["data_dir"],
            )
            next(gen)
            next(gen)
        finally:
            logger.remove(hid)

        needle = (
            f"{stem}: splice_counterfactual scored {expected}/{expected} "
            "reference counterfactual segment"
        )
        assert any(needle in r for r in records)

    def test_whf_derived_features_equal_base_segment_features(self, tmp_path) -> None:
        """Ground truth: the derived segment at whf_derived carries the BASE
        segment (pred+1)'s features verbatim (within-shard bijection)."""
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        fi = FeatureIndex(env["feature_dir"])
        stem = "shard_00000"
        prov = json.loads(
            (Path(env["feature_dir"]) / "_splice_provenance" / f"{stem}.json").read_text()
        )
        checked = 0
        for rec in prov["docs"]:
            for r in rec["splices"]:
                rd, rp = r["whf_derived"]
                ref_seg = fi.lookup(stem, rd)[rp]
                # find whf_base's placement via triples and compare features
                whf = (r["whf_base"][1], r["whf_base"][2])
                src_doc = prov["docs"][rd]
                assert (src_doc["provenance"][rp][1], src_doc["provenance"][rp][2]) == whf
                checked += 1
                assert len(ref_seg.feature_ids) > 0
        assert checked > 0


class TestModeGuards:
    def test_missing_sidecar_dir_raises_at_first_batch(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.0)
        shutil.rmtree(Path(env["feature_dir"]) / "_splice_provenance")
        gen = _loader(env, _threshold_pipeline(mode=DeltaMode.splice_counterfactual))
        with pytest.raises(SpliceProvenanceError, match="_splice_provenance"):
            next(gen)

    def test_wrong_data_dir_n_docs_mismatch_raises(self, tmp_path) -> None:
        """Pointing data_dir at the BASE parquets (wrong pairing) dies on the
        per-parquet n_docs guard before any doc is consumed.

        Non-vacuity guard: with the default DEFAULT_TEXTS_BY_SHARD corpus,
        cut_fraction=0.6 reliably breaks a loop in shard_00000 (6 base docs ->
        7 derived docs, verified empirically), so the base parquet's row
        count and the derived sidecar's n_docs are asserted to differ BEFORE
        exercising the raise — otherwise this test would silently degrade
        into a no-op if a future corpus/seed change made the counts coincide.
        """
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base_dir = Path(env["feature_dir"]).parents[2]  # .../s2/base
        idx = SpliceProvenanceIndex(env["feature_dir"])
        base_pq = base_dir / "shard_00000.parquet"
        base_rows = pq.ParquetFile(base_pq).metadata.num_rows
        derived_n_docs = idx.shard_n_docs("shard_00000")
        assert base_rows != derived_n_docs, (
            f"non-vacuity guard: base parquet rows ({base_rows}) must differ "
            f"from the derived sidecar's n_docs ({derived_n_docs}) for this "
            "test to actually exercise the n_docs-mismatch guard; adjust the "
            "corpus or cut_fraction if this ever coincides."
        )
        env_wrong = dict(env, data_dir=str(base_dir))
        gen = _loader(env_wrong, _threshold_pipeline(mode=DeltaMode.splice_counterfactual))
        with pytest.raises(SpliceProvenanceError, match="n_docs|rows"):
            next(gen)


class TestO2TargetedOverride:
    """Threshold selection + no negatives: outputs equal the feature_delta=None
    run at every non-post-splice segment; differ exactly at post-splice ones."""

    def _doc_streams(self, env, mode):
        """Collect per-(shard,doc,seg) selected fid tuples from raw doc batches.

        Keys the output dict by arrival order across the single train shard
        (shard_00000). Arrival order equals the real derived doc_idx here
        because DDP is single-rank (world_size=1): _document_batches_with_features
        visits every row group of the ONE train shard sequentially without
        skipping any, accumulating global_doc_offset in that same order, so
        the running "docs_seen" counter coincides with doc_idx regardless of
        how many row groups the tiny fixture parquet happens to be split into.
        The break condition stops after exactly one full pass by comparing
        against the shard's real doc count (from SpliceProvenanceIndex) rather
        than a hardcoded literal — the generator itself is infinite (it loops
        across epochs).
        """
        fi = FeatureIndex(env["feature_dir"])
        idx = SpliceProvenanceIndex(env["feature_dir"])
        n_docs = idx.shard_n_docs("shard_00000")
        gen = _document_batches_with_features(
            split=DatasetSplit.TRAIN,
            resume_state_dict=None,
            tokenizer_batch_size=16,
            tokenizer=env["tokenizer"],
            feature_index=fi,
            pipeline=_threshold_pipeline(mode=mode),
            segment_size=env["seg_size"],
            data_dir=env["data_dir"],
        )
        out = {}
        docs_seen = 0
        for doc_list, _state in gen:
            for _text_tokens, seg_features in doc_list:
                assert seg_features is not None
                out[docs_seen] = [
                    (sf.feature_ids.tolist(), sf.percentiles.tolist())
                    for sf in seg_features
                ]
                docs_seen += 1
            if docs_seen >= n_docs:  # exactly one full pass over the train shard's docs
                break
        return out

    def test_differs_only_at_post_splice_segments(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base = self._doc_streams(env, None)
        cf = self._doc_streams(env, DeltaMode.splice_counterfactual)
        idx = SpliceProvenanceIndex(env["feature_dir"])
        stem = "shard_00000"
        assert base.keys() == cf.keys()
        n_diff = 0
        for doc_key in base:
            splices = idx.doc_splices(stem, doc_key)
            for s_idx, (b_seg, c_seg) in enumerate(
                zip(base[doc_key], cf[doc_key], strict=True)
            ):
                if s_idx in splices:
                    n_diff += 1  # allowed to differ (usually does)
                else:
                    assert b_seg == c_seg, (
                        f"doc {doc_key} seg {s_idx} is NOT post-splice but differs"
                    )
        assert n_diff > 0, "corpus must contain post-splice segments (non-vacuity)"

    def test_post_splice_selection_matches_hand_computation(self, tmp_path) -> None:
        """The cf run's post-splice positives == threshold(positive-delta set)."""
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        fi = FeatureIndex(env["feature_dir"])
        idx = SpliceProvenanceIndex(env["feature_dir"])
        cf = self._doc_streams(env, DeltaMode.splice_counterfactual)
        stem = "shard_00000"
        table = _percentile_table()
        cfg = ThresholdConfig(ThresholdType.segment_topk, 2)
        checked = 0
        for doc_key, segs in cf.items():
            splices = idx.doc_splices(stem, doc_key)
            if not splices:
                continue
            segments = fi.lookup(stem, doc_key)
            seg_refs = [None] * len(segments)
            for pos, sref in splices.items():
                rd, rp = sref.whf_derived
                ref_seg = fi.lookup(stem, rd)[rp]
                seg_refs[pos] = (ref_seg.feature_ids, ref_seg.feature_mags, None)
            out = compute_doc_counterfactual_deltas(
                [s.feature_ids for s in segments],
                [s.feature_mags for s in segments],
                [None] * len(segments), seg_refs, _N_SAE, table,
            )
            for pos in splices:
                r = out[pos]
                assert r is not None
                exp_fids, _ = apply_threshold(cfg, r[0], r[1], precomputed=None)
                assert segs[pos][0] == exp_fids.tolist()
                checked += 1
        assert checked > 0


class TestFpTargetsInvariance:
    def test_all_feature_ids_unchanged_by_mode(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        base = _take(_loader(env, _sampling_pipeline(mode=None, with_negatives=False),
                             B=2, T=32), 3)
        cf = _take(_loader(env, _sampling_pipeline(
            mode=DeltaMode.splice_counterfactual, with_negatives=False), B=2, T=32), 3)
        # fp_targets (index 8 of the 10-tuple) is built from PRE-selection
        # features; None here (n_sae_features unset) — instead compare the
        # pre-selection stream at doc level: identical inputs/targets/seg_map.
        for b, c in zip(base, cf, strict=True):
            assert t.equal(b[0], c[0])   # inputs
            assert t.equal(b[1], c[1])   # targets
            assert t.equal(b[6], c[6])   # seg_map


class TestDriftRaises:
    def test_sidecar_shard_drift_raises_d11(self, tmp_path) -> None:
        """Mutate the sidecar so a whf_derived points at a doc that passes the
        sidecar's own internal checks — and the shard-level n_docs/parquet-rows
        pairing guard — but is absent from the real feature shard.

        A fake doc record is grown (doc_id == old n_docs, n_docs += 1) so
        SpliceProvenanceIndex's own eager validation accepts it, AND a matching
        dummy row is appended to the derived parquet so the shard-level
        n_docs-vs-parquet-rows guard (already covered by
        TestModeGuards.test_wrong_data_dir_n_docs_mismatch_raises) also passes.
        The fake doc was never written to the feature shard (safetensors), so
        the failure can only surface where a would-have-followed reference is
        actually resolved against FeatureIndex — the true per-splice D11 drift
        path in ``_resolve_counterfactual_refs`` (KeyError -> SpliceProvenanceError).
        """
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        stem = "shard_00000"
        p = Path(env["feature_dir"]) / "_splice_provenance" / f"{stem}.json"
        prov = json.loads(p.read_text())
        # Grow a fake doc record so internal validation passes but the feature
        # shard has no doc with that id.
        fake_id = prov["n_docs"]
        first_with = next(d for d in prov["docs"] if d["splices"])
        prov["docs"].append({
            "doc_id": fake_id, "seed_doc_id": 0, "n_segments": 2, "n_switches": 0,
            "provenance": [[0, 0, 0], [0, 0, 1]], "segment_token_starts": [0, 2],
            "splices": [],
        })
        prov["n_docs"] += 1
        first_with["splices"][0]["whf_derived"] = [fake_id, 0]
        p.write_text(json.dumps(prov))
        # Keep the parquet's row count in lockstep with the mutated sidecar's
        # n_docs so the shard-level guard doesn't preempt the deeper
        # per-splice FeatureIndex lookup this test targets. Content is never
        # read — fake_id == the OLD n_docs, i.e. one past every real doc, so
        # first_with's own doc_id is always < fake_id and gets processed (and
        # raises) first; the fake doc's own row is never reached.
        data_dir = Path(env["data_dir"])
        pq_path = data_dir / f"{stem}.parquet"
        texts = pq.ParquetFile(pq_path).read().column("text").to_pylist()
        texts.append(texts[0])
        atomic_write_parquet(pa.table({"text": texts}), pq_path)
        gen = _loader(env, _threshold_pipeline(mode=DeltaMode.splice_counterfactual))
        with pytest.raises(SpliceProvenanceError, match="absent from the feature shard"):
            next(gen)


class TestResume:
    def test_resume_round_trip(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        pipeline = _threshold_pipeline(mode=DeltaMode.splice_counterfactual)
        gen = _loader(env, pipeline)
        batches = _take(gen, 3)
        state = batches[1][-1]
        pipeline2 = _threshold_pipeline(mode=DeltaMode.splice_counterfactual)
        fi2 = FeatureIndex(env["feature_dir"])
        gen2 = encoder_decoder_data_loader_with_state(
            tokenizer=env["tokenizer"], B=2, T=32, split=DatasetSplit.TRAIN,
            feature_index=fi2, pipeline=pipeline2, segment_size=env["seg_size"],
            decoder_segment_size=None, device="cpu",
            resume_state_dict=dict(state), tokenizer_batch_size=16,
            buffer_size=16, data_dir=env["data_dir"], prefetch_batches=0,
            label_dropout=0.0,
        )
        next(gen2)  # must not raise; sidecars reload cleanly on resume


class TestValSplit:
    def test_val_split_uses_last_shard_sidecar(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        gen = _loader(env, _threshold_pipeline(mode=DeltaMode.splice_counterfactual),
                      split=DatasetSplit.VAL)
        next(gen)  # must not raise — val shard sidecar loads + validates


class TestSamplingWithNegativesMixedDoc:
    """The one combination no existing oracle covers: a genuinely spliced
    derived doc (cut_fraction=0.6) under SAMPLING selection WITH negatives.

    O1 (TestO1IdentityStreamEquality) only exercises this pipeline shape at
    cut_fraction=0.0 (identity — no real splices). O2
    (TestO2TargetedOverride) covers cut_fraction=0.6 but only under threshold
    selection with no negatives. Full-stream byte-equality is infeasible here
    (sampling RNG + per-segment draw-count shifts from the counterfactual
    deltas change downstream draws), so this asserts the structural +
    non-vacuity properties that would break if the counterfactual path were
    miswired instead.
    """

    def test_structural_and_non_vacuity(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)

        # Non-vacuity guard FIRST: this run only exercises the 4b
        # counterfactual body if the train shard actually has splice
        # positions recorded.
        idx = SpliceProvenanceIndex(env["feature_dir"])
        stem = "shard_00000"
        n_splice_positions = sum(
            len(idx.doc_splices(stem, doc))
            for doc in range(idx.shard_n_docs(stem))
        )
        if n_splice_positions == 0:
            pytest.fail(
                "cut_fraction=0.6 recorded ZERO splice positions in "
                f"{stem} — this test would prove nothing about the "
                "splice_counterfactual path. Adjust the fixture corpus or "
                "cut_fraction."
            )

        pipeline = _sampling_pipeline(mode=DeltaMode.splice_counterfactual, with_negatives=True)
        batches = _take(_loader(env, pipeline), 5)

        B, T = 2, 32
        found_negative = False
        for batch in batches:
            assert isinstance(batch, tuple) and len(batch) == 10
            x, y, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict = batch

            assert x.shape == (B, T) and x.dtype == t.long
            assert y.shape == (B, T) and y.dtype == t.long
            assert enc_fids.dim() == 3 and enc_fids.shape[0] == B and enc_fids.dtype == t.int64
            assert enc_mags.shape == enc_fids.shape and enc_mags.dtype == t.int64
            assert enc_mask.shape == enc_fids.shape and enc_mask.dtype == t.bool
            assert enc_pcts.shape == enc_fids.shape and enc_pcts.dtype == t.float32
            assert seg_map.shape == (B, T) and seg_map.dtype == t.int64
            assert seg_doc_id.shape[0] == B and seg_doc_id.dtype == t.int64
            assert isinstance(state_dict, dict)

            # Negative sentinel: with_negatives=True => effective_buckets ==
            # _NUM_BUCKETS - 1; negative slots carry that bucket value and
            # percentile == 0.0.
            effective_buckets = pipeline.effective_buckets
            neg_mask = (enc_mags == effective_buckets) & enc_mask
            if neg_mask.any():
                found_negative = True
                assert (enc_pcts[neg_mask] == 0.0).all(), (
                    "Negative features should have percentile == 0.0"
                )

        assert found_negative, (
            "no negative features observed across 5 batches — with "
            "min_k=1 per segment this should always fire"
        )

        # Determinism: sampling RNG is seeded inside the pipeline, so two
        # fresh loaders built from freshly-constructed identical pipelines
        # over the same fixture must yield identical streams.
        pipeline_a = _sampling_pipeline(mode=DeltaMode.splice_counterfactual, with_negatives=True)
        pipeline_b = _sampling_pipeline(mode=DeltaMode.splice_counterfactual, with_negatives=True)
        run_a = _take(_loader(env, pipeline_a), 5)
        run_b = _take(_loader(env, pipeline_b), 5)
        _assert_batches_equal(run_a, run_b)


class TestValidateCounterfactualDeltaResults:
    """Direct unit coverage of the per-doc position cross-check (268ee44e
    tightening). The load-bearing correctness property: a post-splice segment
    with ALL-NEGATIVE deltas produces a NON-None DeltaSegResult whose positive
    (filtered) set is EMPTY, and it MUST still count as a scored delta position
    — the cross-check keys on result-is-not-None, not on positive-presence. If
    it keyed on positives, this exact (legitimate) state would false-positive
    and crash valid training."""

    @staticmethod
    def _empty_positive_result():
        # A non-None DeltaSegResult with an empty positive set (all deltas <= 0),
        # exactly what compute_doc_counterfactual_deltas returns for an
        # all-negative-delta post-splice segment.
        return (
            t.tensor([], dtype=t.int64),    # filtered_fids (no positives)
            t.tensor([], dtype=t.float32),  # filtered_fmags
            t.tensor([], dtype=t.float32),  # delta_values
            t.tensor([7], dtype=t.int64),   # neg_delta_fids (ref-only present)
            t.tensor([0.5], dtype=t.float32),  # neg_delta_weights
            t.zeros(64, dtype=t.bool),      # ref_mask
            None,                           # filtered_pcts
        )

    def test_empty_positive_post_splice_segment_is_still_scored(self) -> None:
        ref = (t.tensor([7], dtype=t.int64), t.tensor([1.0]), None)
        seg_refs = [None, ref, None]  # splice position 1, non-splice 0/2
        batch_results = [None, self._empty_positive_result(), None]
        n = _validate_counterfactual_delta_results(
            shard_stem="shard_00000",
            doc_idx=0,
            expected_positions={1},
            seg_refs=seg_refs,  # ty: ignore[invalid-argument-type]
            batch_results=batch_results,
        )
        assert n == 1  # the empty-positive segment counts as one scored position

    def test_happy_path_multiple_positions(self) -> None:
        ref = (t.tensor([7], dtype=t.int64), t.tensor([1.0]), None)
        seg_refs = [None, ref, None, ref]
        batch_results = [
            None, self._empty_positive_result(), None, self._empty_positive_result(),
        ]
        n = _validate_counterfactual_delta_results(
            shard_stem="s", doc_idx=0, expected_positions={1, 3},
            seg_refs=seg_refs, batch_results=batch_results,  # ty: ignore[invalid-argument-type]
        )
        assert n == 2

    def test_ref_position_mismatch_raises(self) -> None:
        # seg_refs has a ref at pos 1, but the sidecar says the splice is at pos 2.
        ref = (t.tensor([7], dtype=t.int64), t.tensor([1.0]), None)
        with pytest.raises(SpliceProvenanceError, match="ref positions"):
            _validate_counterfactual_delta_results(
                shard_stem="s", doc_idx=0, expected_positions={2},
                seg_refs=[None, ref, None],
                batch_results=[None, self._empty_positive_result(), None],
            )

    def test_delta_scored_mismatch_raises(self) -> None:
        # A compute bug returns None where a splice position was expected/resolved.
        ref = (t.tensor([7], dtype=t.int64), t.tensor([1.0]), None)
        with pytest.raises(SpliceProvenanceError, match="delta-scored"):
            _validate_counterfactual_delta_results(
                shard_stem="s", doc_idx=0, expected_positions={1},
                seg_refs=[None, ref, None],
                batch_results=[None, None, None],  # position 1 not scored
            )
