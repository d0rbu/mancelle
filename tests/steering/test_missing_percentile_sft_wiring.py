"""The missing-percentile policy is honoured by EVERY conditioned-SFT consumer.

The unit behaviour of the policy lives in ``test_missing_percentile.py``. This
file pins the wiring: which id set each table is actually queried with (the
open question the owner asked about), and that the ``lowest`` substitutions
reach unclustered selectors, feature-prediction targets, weighted and top-k
sampling, and the bucket space — not just the one call site that happens to be
easiest to patch.
"""

from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch as t
from omegaconf import OmegaConf

import scripts.train.conditional_encdec_sft as conditional_encdec_sft
from scripts.train.conditional_encdec_sft import (
    SFTState,
    make_sft_data_generator,
)
from steering.conditional.encoder_decoder import sft_selection
from steering.conditional.encoder_decoder.sft_features import (
    scan_required_feature_id_bitmap,
)
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.missing_percentile import (
    LOWEST_MAGNITUDE_BUCKET,
    LOWEST_NORMALIZED_RANK,
    MissingPercentileError,
    MissingPercentilePolicy,
    MissingPercentileResolver,
    check_percentile_coverage,
    enforce_percentile_coverage,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.sft_schema import SFTSplit
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    _full_sequence_char_count,
    make_lazy_patched_sft_generator,
    patch_load_sft_features,
)

N_SAE = 32
NUM_BUCKETS = 8

# Only these ids have percentile statistics. Everything else has an all-zero
# table row and an all-zero boundaries row.
COVERED_IDS = frozenset({0, 1, 2, 3})
# Feature 5 is ACTIVE in the data but has NO statistics — the failure this
# whole mechanism is about.
UNCOVERED_ACTIVE_ID = 5


def _percentile_table(covered=COVERED_IDS) -> t.Tensor:
    table = t.zeros(N_SAE, 99, dtype=t.float64)
    row = t.arange(1, 100, dtype=t.float64)
    for fid in covered:
        table[fid] = row
    return table


def _boundaries(covered=COVERED_IDS) -> t.Tensor:
    """Per-feature boundaries mirroring the table: uncovered rows are all zero."""
    out = t.zeros(N_SAE, NUM_BUCKETS - 1, dtype=t.float64)
    row = t.linspace(0.5, float(NUM_BUCKETS) - 0.5, NUM_BUCKETS - 1, dtype=t.float64)
    for fid in covered:
        out[fid] = row
    return out


def _features_with(task, tokenizer, fids: list[int], mags: list[float]):
    """One segment per conversation carrying exactly ``fids`` / ``mags``."""
    out: dict[int, list[SegmentFeatures]] = {}
    for idx in range(len(task)):
        conv = task[idx]
        total_chars = _full_sequence_char_count(tokenizer, conv)
        out[idx] = [
            SegmentFeatures(
                segment_idx=0,
                position=0,
                feature_ids=t.tensor(fids, dtype=t.int32),
                feature_mags=t.tensor(mags, dtype=t.float32),
                char_start=0,
                char_end=total_chars,
            )
        ]
    return out


def _lowest_resolver(covered=COVERED_IDS) -> MissingPercentileResolver:
    bitmap = t.zeros(N_SAE, dtype=t.bool)
    for fid in covered:
        bitmap[fid] = True
    return MissingPercentileResolver(
        policy=MissingPercentilePolicy.LOWEST, covered=bitmap
    )


def _gen(
    task,
    tokenizer,
    features,
    *,
    missing_percentile=None,
    sampling_bundle=None,
    threshold_cfg=None,
    fp_percentile_table=None,
    feature_mask=None,
    precomputed_thresholds=None,
    B: int = 1,
    T: int = 64,
):
    def factory():
        return make_sft_data_generator(
            task=task,
            tokenizer=tokenizer,
            B=B,
            T=T,
            device="cpu",
            sft_state=SFTState(),
            feature_dir="/dummy/features",
            parquet_dir="/dummy/parquet",
            threshold_cfg=threshold_cfg
            if threshold_cfg is not None or sampling_bundle is not None
            else ThresholdConfig(
                threshold_type=ThresholdType.segment_topk, threshold=1000
            ),
            boundaries=_boundaries(),
            precomputed_thresholds=precomputed_thresholds,
            sampling_bundle=sampling_bundle,
            n_sae_features=N_SAE if fp_percentile_table is not None else None,
            fp_percentile_table=fp_percentile_table,
            feature_mask=feature_mask,
            missing_percentile=missing_percentile,
            min_loss_tokens=1,
            alignment_probe_samples=0,
            expected_split=SFTSplit.TRAIN,
        )

    return make_lazy_patched_sft_generator(features, factory)


# ---------------------------------------------------------------------------
# The open question: RAW pre-selection ids vs post-mask candidates
# ---------------------------------------------------------------------------


class TestWhichIdsEachTableIsQueriedWith:
    """#1316 scoped its check to post-mask positives; the FP lookup does not."""

    def test_scan_returns_the_raw_pre_selection_union(self):
        """The scan must see ids the feature mask will later drop.

        ``full_fids = seg.feature_ids`` feeds ``magnitude_to_percentile``
        BEFORE ``feature_mask.apply_to_positive`` and before cluster
        consolidation, so a scan restricted to allowed ids would under-report
        exactly the population that makes the primitive raise.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID, 9], [4.0, 3.0, 2.0]
        )
        seen = scan_required_feature_id_bitmap(
            features, list(features), n_sae_features=N_SAE, source="train"
        )
        assert t.nonzero(seen).squeeze(1).tolist() == [1, UNCOVERED_ACTIVE_ID, 9]

    def test_out_of_range_feature_id_is_a_loud_width_mismatch(self):
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=1, user_chars=3, asst_chars=10)
        features = _features_with(task, tokenizer, [1, N_SAE + 4], [2.0, 1.0])
        with pytest.raises(Exception, match=r"outside \[0, 32\)"):
            scan_required_feature_id_bitmap(
                features, list(features), n_sae_features=N_SAE, source="train"
            )

    def test_the_two_tables_get_different_verdicts_on_the_same_data(self):
        """This is the answer, made executable.

        A whitelist allows {0, 1, 2, 3}. The data activates {1, 5, 9}. The
        percentile table covers exactly the whitelist. Then:

        * the FP table, queried with the RAW ids, is MISSING 5 and 9;
        * the sampling table, queried with the post-mask candidates, is
          complete.

        A single shared check would either false-alarm on the sampling table or
        wave through the FP hole. Validating them independently is the only
        answer that is right for both.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=1, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID, 9], [4.0, 3.0, 2.0]
        )
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(sorted(COVERED_IDS)))

        seen = scan_required_feature_id_bitmap(
            features, list(features), n_sae_features=N_SAE, source="train"
        )
        raw_required = t.nonzero(seen, as_tuple=False).squeeze(1)
        masked_required = raw_required[mask.keep_mask(raw_required)]
        assert raw_required.tolist() == [1, UNCOVERED_ACTIVE_ID, 9]
        assert masked_required.tolist() == [1]

        table = _percentile_table()
        fp_report = check_percentile_coverage(
            percentile_table=table,
            required_ids=raw_required,
            table_name="fp_percentile_table",
            required_set_name="raw pre-selection segment feature ids",
        )
        sampling_report = check_percentile_coverage(
            percentile_table=table,
            required_ids=masked_required,
            table_name="sampling_bundle.percentile_table",
            required_set_name="post-feature-mask selection candidates",
        )
        assert fp_report.n_uncovered == 2
        assert sampling_report.n_uncovered == 0

        with pytest.raises(
            MissingPercentileError,
            match=r"fp_percentile_table is missing percentile statistics for 2 of the 3",
        ):
            enforce_percentile_coverage(
                [fp_report, sampling_report], MissingPercentilePolicy.ERROR
            )
        # Scoped to the post-mask set alone, the run would have been cleared.
        enforce_percentile_coverage([sampling_report], MissingPercentilePolicy.ERROR)


# ---------------------------------------------------------------------------
# error policy: the primitive is still the backstop
# ---------------------------------------------------------------------------


class TestErrorPolicyIsStillFailLoud:
    def test_fp_lookup_on_an_uncovered_raw_id_raises(self):
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID], [4.0, 3.0]
        )
        gen = _gen(
            task,
            tokenizer,
            features,
            fp_percentile_table=_percentile_table(),
        )
        with pytest.raises(ValueError, match=r"missing for 1 distinct feature ID"):
            next(gen)

    def test_default_none_resolver_behaves_as_error(self):
        """Omitting the kwarg must not silently tolerate holes."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID], [4.0, 3.0]
        )
        gen = _gen(
            task,
            tokenizer,
            features,
            missing_percentile=None,
            fp_percentile_table=_percentile_table(),
        )
        with pytest.raises(ValueError, match=r"all-zero percentile rows"):
            next(gen)


# ---------------------------------------------------------------------------
# lowest policy: one contract, every consumer
# ---------------------------------------------------------------------------


class TestLowestPolicyReachesEveryConsumer:
    def test_feature_prediction_target_is_the_documented_floor(self):
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID], [4.0, 3.0]
        )
        gen = _gen(
            task,
            tokenizer,
            features,
            missing_percentile=_lowest_resolver(),
            fp_percentile_table=_percentile_table(),
        )
        batch = next(gen)
        fp_targets = batch[-2] if len(batch) == 8 else batch[-1]
        assert fp_targets is not None
        column = fp_targets[..., UNCOVERED_ACTIVE_ID]
        active = column[column > 0]
        assert active.numel() > 0, "the uncovered feature must still be a target"
        assert active.allclose(
            t.full_like(active, LOWEST_NORMALIZED_RANK), atol=1e-7
        ), active

    def test_unclustered_bucket_is_pinned_to_zero_not_the_top(self):
        """``discretize_activations`` puts an uncovered id in the TOP bucket.

        The all-zero boundaries row means every positive magnitude searchsorts
        past every boundary. Without the pin, the encoder is told the unknown
        feature fired harder than anything else in the segment.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID], [4.0, 3.0]
        )
        pinned = next(
            _gen(
                task,
                tokenizer,
                features,
                missing_percentile=_lowest_resolver(),
                fp_percentile_table=_percentile_table(),
            )
        )
        enc_fids, enc_mags, enc_mask = pinned[2], pinned[3], pinned[4]
        selected = enc_mask & (enc_fids == UNCOVERED_ACTIVE_ID)
        assert bool(selected.any()), "the uncovered feature must be selected here"
        assert (enc_mags[selected] == LOWEST_MAGNITUDE_BUCKET).all(), enc_mags[selected]

        covered_selected = enc_mask & (enc_fids == 1)
        assert (enc_mags[covered_selected] > LOWEST_MAGNITUDE_BUCKET).all(), (
            "a covered feature's bucket must be untouched"
        )

    def test_weighted_sampling_never_reaches_the_unresolved_primitive(self, monkeypatch):
        """``sample_features`` would do its OWN lookup without override weights.

        That call bypasses the policy entirely, so ``lowest`` would still die on
        the very id it is meant to tolerate.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID], [4.0, 3.0]
        )
        bundle = SamplingBundle(
            percentile_table=_percentile_table(),
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=10.0,
                std=10.0,
                min_k=1,
                max_k=8,
                mode=SamplingMode.weighted,
            ),
            rng=t.Generator().manual_seed(0),
        )
        seen_overrides: list[t.Tensor | None] = []
        real_sample = sft_selection.sample_features

        def spy(fids, mags, table, selection, rng, override_weights=None, **kwargs):
            seen_overrides.append(override_weights)
            return real_sample(
                fids, mags, table, selection, rng, override_weights=override_weights, **kwargs
            )

        monkeypatch.setattr(sft_selection, "sample_features", spy)
        gen = _gen(
            task,
            tokenizer,
            features,
            missing_percentile=_lowest_resolver(),
            sampling_bundle=bundle,
        )
        next(gen)
        assert seen_overrides, "sample_features was never called"
        assert all(w is not None for w in seen_overrides), seen_overrides
        for weights in seen_overrides:
            assert weights is not None
            assert float(weights.min().item()) == pytest.approx(0.5)

    def test_top_k_percentile_sampling_ranks_the_unknown_feature_last(self):
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID, 2], [50.0, 99.0, 60.0]
        )
        bundle = SamplingBundle(
            percentile_table=_percentile_table(),
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=1.0,
                std=1.0,
                min_k=1,
                max_k=2,
                mode=SamplingMode.top_k_percentile,
            ),
            rng=t.Generator().manual_seed(0),
        )
        gen = _gen(
            task,
            tokenizer,
            features,
            missing_percentile=_lowest_resolver(),
            sampling_bundle=bundle,
        )
        batch = next(gen)
        enc_fids, enc_mask = batch[2], batch[4]
        selected = enc_fids[enc_mask].tolist()
        assert selected, "top-1 must select something"
        assert UNCOVERED_ACTIVE_ID not in selected, (
            "rank 0.5 must lose to every covered feature, even at magnitude 99"
        )

    def test_feature_percentile_threshold_does_not_admit_the_unknown_feature(self):
        """An uncovered precomputed threshold is 0.0 and admits everything.

        Lifting it to +inf is what makes the unclustered magnitude-space
        selector agree with the percentile-space ones, where 0.005 already
        fails any cutoff above the floor.
        """
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(
            task, tokenizer, [1, UNCOVERED_ACTIVE_ID], [40.0, 90.0]
        )
        thresholds = t.zeros(N_SAE, dtype=t.float64)
        for fid in COVERED_IDS:
            thresholds[fid] = 10.0
        threshold_cfg = ThresholdConfig(
            threshold_type=ThresholdType.feature_percentile, threshold=0.5
        )

        tolerated = next(
            _gen(
                task,
                tokenizer,
                features,
                missing_percentile=_lowest_resolver(),
                threshold_cfg=threshold_cfg,
                precomputed_thresholds=thresholds,
            )
        )
        enc_fids, enc_mask = tolerated[2], tolerated[4]
        selected = enc_fids[enc_mask].tolist()
        assert 1 in selected, "the covered feature clears its own threshold"
        assert UNCOVERED_ACTIVE_ID not in selected, selected

    def test_a_fully_covered_run_is_untouched_by_either_policy(self):
        """``lowest`` must be inert when nothing is missing."""
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = _features_with(task, tokenizer, [1, 2, 3], [4.0, 3.0, 2.0])
        strict = next(
            _gen(task, tokenizer, features, fp_percentile_table=_percentile_table())
        )
        pilot = next(
            _gen(
                task,
                tokenizer,
                features,
                missing_percentile=_lowest_resolver(),
                fp_percentile_table=_percentile_table(),
            )
        )
        for a, b in zip(strict, pilot, strict=True):
            if a is None:
                assert b is None
            else:
                assert t.equal(a, b)


class TestMainReadsTheConfigKnob:
    """Non-vacuity for the plumbing: ``main()`` must actually parse the key.

    Every generator-level test above hands the resolver in directly, which
    proves the consumers honour it but not that a config value ever reaches
    them. These drive ``main()`` far enough to hit ``parse_missing_percentile_policy``.
    """

    def _cfg(self, policy):
        return OmegaConf.create(
            {
                "lr_warmup_steps": 0,
                "num_iterations": -1,
                "dataset": "safety",
                "feature_prediction_block": "mlp",
                "feature_prediction_head_layers": 0,
                "sft_loss_mask": "full_sequence",
                "missing_percentile_policy": policy,
                "feature_clusters": None,
                "feature_selection": {
                    "threshold_type": "segment_topk",
                    "threshold": 1,
                },
            }
        )

    def _drive(self, monkeypatch, cfg):
        monkeypatch.setattr(conditional_encdec_sft.sys, "argv", ["conditional_encdec_sft.py"])
        monkeypatch.setattr(
            conditional_encdec_sft, "initialize_config_dir", lambda **_: nullcontext()
        )
        monkeypatch.setattr(conditional_encdec_sft, "compose", lambda **_: cfg)
        conditional_encdec_sft.main()

    def test_unknown_policy_aborts_at_startup(self, monkeypatch):
        with pytest.raises(ValueError, match=r"Unknown missing_percentile_policy 'skip'"):
            self._drive(monkeypatch, self._cfg("skip"))

    def test_a_valid_policy_gets_past_the_parse(self, monkeypatch):
        """Guard the guard: 'lowest' must NOT be rejected by the same call.

        Without this, a parser that rejected everything would pass the test
        above vacuously. ``main()`` continues to the next startup requirement
        (the model checkpoint), which is a different error entirely.
        """
        with pytest.raises(Exception) as excinfo:
            self._drive(monkeypatch, self._cfg("lowest"))
        assert "missing_percentile_policy" not in str(excinfo.value), excinfo.value


def test_patch_helper_is_exercised():
    """Guard against the synthetic store silently not being installed."""
    with patch_load_sft_features({}) as store:
        assert conditional_encdec_sft.load_sft_features("/a", "/b") is store
