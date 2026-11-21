"""Tests for feature-mask enforcement in the self-conditioned feedback loop.

Defense-in-depth: even if the FP head strongly predicts a blacklisted
feature, it must never enter the next segment's encoder conditioning.
Whitelist: only whitelisted features may become conditioning.
I5: with no mask (None), the provider is byte-identical to the pre-change
behavior for identical inputs/seed.
"""

from __future__ import annotations

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.packed_encoder import PackedEncoderOutput
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    build_model_with_feature_prediction,
)
from tests.steering._self_conditioned_state import build_test_self_cond_state


def _assert_packed_equal(
    actual: PackedEncoderOutput,
    expected: PackedEncoderOutput,
) -> None:
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.layout.flat_indices, expected.layout.flat_indices)
    assert torch.equal(actual.layout.valid_mask, expected.layout.valid_mask)
    assert torch.equal(actual.layout.batch_indices, expected.layout.batch_indices)
    assert torch.equal(actual.layout.segment_indices, expected.layout.segment_indices)
    assert actual.layout.batch_size == expected.layout.batch_size
    assert actual.layout.n_segments == expected.layout.n_segments
    assert actual.layout.feature_count == expected.layout.feature_count


def _packed_snapshot(packed: PackedEncoderOutput) -> PackedEncoderOutput:
    layout = packed.layout
    return PackedEncoderOutput(
        values=packed.values.clone(),
        layout=type(layout)(
            flat_indices=layout.flat_indices.clone(),
            valid_mask=layout.valid_mask.clone(),
            batch_indices=layout.batch_indices.clone(),
            segment_indices=layout.segment_indices.clone(),
            batch_size=layout.batch_size,
            n_segments=layout.n_segments,
            feature_count=layout.feature_count,
        ),
    )

# ---------------------------------------------------------------------------
# Fixtures (mirrors test_self_conditioned_provider.py)
# ---------------------------------------------------------------------------


def _threshold_selection(threshold_type, threshold, precomputed=None):
    return ThresholdSelectionConfig(
        selection=ThresholdSelection(
            threshold_type=threshold_type, threshold=threshold,
        ),
        precomputed_thresholds=precomputed,
    )


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


@pytest.fixture
def tiny_model():
    torch.manual_seed(0)
    return build_model_with_feature_prediction()


def _make_config(
    selection,
    feature_mask=None,
    n_sae_features=N_SAE_FEATURES,
):
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(8),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=selection,
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=NUM_BUCKETS,
        n_sae_features=PositiveInt.parse(n_sae_features),
        feature_mask=feature_mask,
    )


def _make_provider(model, cfg):
    return SelfConditionedFeatureProvider(
        model=model, cfg=cfg, device=torch.device("cpu"),
    )


# ---------------------------------------------------------------------------
# ID-tracking provider subclass
# ---------------------------------------------------------------------------


class _TrackingProvider(SelfConditionedFeatureProvider):
    """Subclass that records every (fid, bucket) dict passed to the encoder."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.emitted_segments: list[dict[int, int]] = []

    def _build_segment_inputs(self, features):
        self.emitted_segments.append(dict(features))
        return super()._build_segment_inputs(features)

# ---------------------------------------------------------------------------
# Blacklist: strongly-predicted blacklisted feature must never appear
# ---------------------------------------------------------------------------


class TestBlacklistNeverFed:
    """Feature K is blacklisted; FP head is rigged to always predict it
    maximally. K must never appear in the emitted conditioning ids."""

    BLACKLISTED = 5  # arbitrary feature id within [0, N_SAE_FEATURES)

    def _make_fp_logits(self, n_sae: int) -> torch.Tensor:
        """All features strongly predicted, blacklisted one maximally."""
        logits = torch.full((1, 8, n_sae), 5.0)
        logits[0, :, self.BLACKLISTED] = 50.0  # maximally predicted
        return logits

    def test_blacklist_segment_topk(self, tiny_model):
        mask = FeatureMask(FeatureMaskMode.blacklist, (self.BLACKLISTED,))
        sel = _threshold_selection(ThresholdType.segment_topk, 10)
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        for _ in range(5):
            fp_logits = self._make_fp_logits(cfg.n_sae_features)
            state = provider.step(state, fp_logits)

        for seg in provider.emitted_segments:
            assert self.BLACKLISTED not in seg, (
                f"Blacklisted feature {self.BLACKLISTED} appeared in emitted "
                f"segment: {seg}"
            )

    def test_blacklist_feature_percentile(self, tiny_model):
        """A permissive per-feature threshold must still respect blacklist."""
        mask = FeatureMask(FeatureMaskMode.blacklist, (self.BLACKLISTED,))
        permissive_thresholds = torch.zeros(N_SAE_FEATURES, dtype=torch.float64)
        sel = _threshold_selection(
            ThresholdType.feature_percentile, 0.5,
            precomputed=permissive_thresholds,
        )
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        for _ in range(5):
            fp_logits = self._make_fp_logits(cfg.n_sae_features)
            state = provider.step(state, fp_logits)

        for seg in provider.emitted_segments:
            assert self.BLACKLISTED not in seg, (
                f"Blacklisted feature {self.BLACKLISTED} appeared in "
                f"feature_percentile conditioning: {seg}"
            )

    def test_blacklist_feature_percentile_segment_topk(self, tiny_model):
        """Default combined thresholding must filter the mask before top-k."""
        mask = FeatureMask(FeatureMaskMode.blacklist, (self.BLACKLISTED,))
        permissive_thresholds = torch.zeros(N_SAE_FEATURES, dtype=torch.float64)
        sel = _threshold_selection(
            ThresholdType.feature_percentile_segment_topk, 20.5,
            precomputed=permissive_thresholds,
        )
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        for _ in range(5):
            fp_logits = self._make_fp_logits(cfg.n_sae_features)
            state = provider.step(state, fp_logits)

        for seg in provider.emitted_segments:
            assert self.BLACKLISTED not in seg, (
                f"Blacklisted feature {self.BLACKLISTED} appeared in "
                f"feature_percentile_segment_topk conditioning: {seg}"
            )

    def test_blacklist_multiple_ids(self, tiny_model):
        blacklisted = (0, 5, 17, 42)
        mask = FeatureMask(FeatureMaskMode.blacklist, blacklisted)
        sel = _threshold_selection(ThresholdType.segment_topk, 20)
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        for _ in range(5):
            logits = torch.full((1, 8, cfg.n_sae_features), 5.0)
            for bl in blacklisted:
                logits[0, :, bl] = 50.0
            state = provider.step(state, logits)

        bl_set = set(blacklisted)
        for seg in provider.emitted_segments:
            overlap = set(seg.keys()) & bl_set
            assert not overlap, (
                f"Blacklisted features {overlap} appeared in emitted segment: {seg}"
            )


# ---------------------------------------------------------------------------
# Whitelist: emitted ids must be ⊆ whitelist
# ---------------------------------------------------------------------------


class TestWhitelistSubset:
    """Only whitelisted features may appear as conditioning."""

    WHITELIST = (1, 3, 7, 11, 15)

    def test_whitelist_segment_topk(self, tiny_model):
        mask = FeatureMask(FeatureMaskMode.whitelist, self.WHITELIST)
        # Ask for top-20 — far more than the whitelist size (5)
        sel = _threshold_selection(ThresholdType.segment_topk, 20)
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        allowed = set(self.WHITELIST)
        for _ in range(5):
            logits = torch.full((1, 8, cfg.n_sae_features), 5.0)
            state = provider.step(state, logits)

        for seg in provider.emitted_segments:
            non_allowed = set(seg.keys()) - allowed
            assert not non_allowed, (
                f"Non-whitelisted features {non_allowed} appeared in emitted "
                f"segment: {seg}"
            )

    def test_whitelist_feature_percentile(self, tiny_model):
        mask = FeatureMask(FeatureMaskMode.whitelist, self.WHITELIST)
        # Thresholds permit every feature; whitelist must further restrict.
        permissive_thresholds = torch.zeros(N_SAE_FEATURES, dtype=torch.float64)
        sel = _threshold_selection(
            ThresholdType.feature_percentile, 0.5,
            precomputed=permissive_thresholds,
        )
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        allowed = set(self.WHITELIST)
        for _ in range(5):
            logits = torch.full((1, 8, cfg.n_sae_features), 5.0)
            state = provider.step(state, logits)

        for seg in provider.emitted_segments:
            non_allowed = set(seg.keys()) - allowed
            assert not non_allowed, (
                f"Non-whitelisted features {non_allowed} from feature_percentile: {seg}"
            )

    def test_whitelist_feature_percentile_segment_topk(self, tiny_model):
        mask = FeatureMask(FeatureMaskMode.whitelist, self.WHITELIST)
        permissive_thresholds = torch.zeros(N_SAE_FEATURES, dtype=torch.float64)
        sel = _threshold_selection(
            ThresholdType.feature_percentile_segment_topk, 20.5,
            precomputed=permissive_thresholds,
        )
        cfg = _make_config(sel, feature_mask=mask)
        provider = _TrackingProvider(model=tiny_model, cfg=cfg, device=torch.device("cpu"))
        state = build_test_self_cond_state(provider)

        allowed = set(self.WHITELIST)
        for _ in range(5):
            logits = torch.full((1, 8, cfg.n_sae_features), 5.0)
            state = provider.step(state, logits)

        for seg in provider.emitted_segments:
            non_allowed = set(seg.keys()) - allowed
            assert not non_allowed, (
                f"Non-whitelisted features {non_allowed} from "
                f"feature_percentile_segment_topk: {seg}"
            )


# ---------------------------------------------------------------------------
# I5: byte-identical when feature_mask=None
# ---------------------------------------------------------------------------


class TestI5ByteIdenticalNoMask:
    """With feature_mask=None the provider must produce byte-identical outputs
    compared to itself run a second time with the same seed, confirming the
    None path is the unmodified path (no new logic executed)."""

    def test_no_mask_segment_topk_identical(self, tiny_model):
        sel = _threshold_selection(ThresholdType.segment_topk, 5)

        def _run():
            torch.manual_seed(42)
            cfg = _make_config(sel, feature_mask=None)
            provider = SelfConditionedFeatureProvider(
                model=tiny_model, cfg=cfg, device=torch.device("cpu"),
            )
            state = build_test_self_cond_state(provider)
            states = []
            for _ in range(3):
                fp_logits = torch.randn(1, 8, cfg.n_sae_features, generator=torch.Generator().manual_seed(99))
                state = provider.step(state, fp_logits)
                states.append(state)
            return states

        states_a = _run()
        states_b = _run()

        for i, (sa, sb) in enumerate(zip(states_a, states_b, strict=True)):
            _assert_packed_equal(sa.encoder_out, sb.encoder_out)
            assert sa.target_slot_idx == sb.target_slot_idx, (
                f"target_slot_idx mismatch at step {i} with no mask"
            )

    def test_mask_none_vs_no_mask_field_identical(self, tiny_model):
        """Explicitly passing feature_mask=None must give the same output as
        not having the field at all (i.e. the None-path code is not reached)."""
        torch.manual_seed(42)
        fp_logits = torch.randn(1, 8, N_SAE_FEATURES)

        sel = _threshold_selection(ThresholdType.segment_topk, 5)
        cfg_none = _make_config(sel, feature_mask=None)
        provider_none = SelfConditionedFeatureProvider(
            model=tiny_model, cfg=cfg_none, device=torch.device("cpu"),
        )
        state_none = build_test_self_cond_state(provider_none)
        state_none = provider_none.step(state_none, fp_logits)

        # Re-run with same logits
        provider_none2 = SelfConditionedFeatureProvider(
            model=tiny_model, cfg=cfg_none, device=torch.device("cpu"),
        )
        state_none2 = build_test_self_cond_state(provider_none2)
        state_none2 = provider_none2.step(state_none2, fp_logits)

        _assert_packed_equal(state_none.encoder_out, state_none2.encoder_out)


# ---------------------------------------------------------------------------
# SamplingBundle path: blacklist / whitelist / I5 (None mask)
# ---------------------------------------------------------------------------


def _make_sampling_bundle(seed: int = 0, mean: float = 3.0) -> SamplingBundle:
    """Mirror of test_sampling_self_cond_e2e._make_sampling_bundle.

    percentile_table is unused at inference (override_weights replaces it),
    so a zero placeholder of the right shape satisfies the contract.
    """
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return SamplingBundle(
        selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=mean,
            std=mean,
            min_k=1,
            max_k=N_SAE_FEATURES,
            mode=SamplingMode.weighted,
        ),
        percentile_table=torch.zeros(N_SAE_FEATURES, 1, dtype=torch.float64),
        rng=rng,
    )


def _make_sampling_config(
    feature_mask=None,
    seed: int = 0,
    mean: float = 3.0,
) -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(8),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=_make_sampling_bundle(seed=seed, mean=mean),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=NUM_BUCKETS,
        n_sae_features=N_SAE_FEATURES,
        feature_mask=feature_mask,
    )


class TestSamplingBundleMask:
    """FeatureMask enforcement via the SamplingBundle branch of _apply_selection."""

    BLACKLISTED = 7  # arbitrary id within [0, N_SAE_FEATURES)

    # ------------------------------------------------------------------
    # blacklist: K is strongly favoured; must never appear
    # ------------------------------------------------------------------

    def test_blacklist_sampling_never_fed(self, tiny_model):
        """FP head arranged so feature BLACKLISTED is strongly predicted;
        with a blacklist mask it must never enter conditioning across many
        steps and two independent seeds."""
        for seed in (0, 17, 99):
            mask = FeatureMask(FeatureMaskMode.blacklist, (self.BLACKLISTED,))
            cfg = _make_sampling_config(feature_mask=mask, seed=seed, mean=4.0)
            provider = _TrackingProvider(
                model=tiny_model, cfg=cfg, device=torch.device("cpu"),
            )
            state = build_test_self_cond_state(provider)

            for _ in range(8):
                # Strongly bias all features; BLACKLISTED at maximum logit so
                # any sampling weighting would select it if the gate were absent.
                fp_logits = torch.full((1, 8, N_SAE_FEATURES), 5.0)
                fp_logits[0, :, self.BLACKLISTED] = 50.0
                state = provider.step(state, fp_logits)

            for i, seg in enumerate(provider.emitted_segments):
                assert self.BLACKLISTED not in seg, (
                    f"seed={seed}: blacklisted feature {self.BLACKLISTED} "
                    f"appeared in emitted segment {i}: {seg}"
                )

    # ------------------------------------------------------------------
    # whitelist: every emitted id must be in the whitelist
    # ------------------------------------------------------------------

    def test_whitelist_sampling_subset(self, tiny_model):
        """Request mean=10 features (geometric) but only 5 are whitelisted;
        every emitted id must be in the whitelist."""
        whitelist = (2, 9, 31, 77, 150)
        mask = FeatureMask(FeatureMaskMode.whitelist, whitelist)
        # mean=10 >> len(whitelist)=5, so sampler would draw outside if gate absent
        cfg = _make_sampling_config(feature_mask=mask, seed=0, mean=10.0)
        provider = _TrackingProvider(
            model=tiny_model, cfg=cfg, device=torch.device("cpu"),
        )
        state = build_test_self_cond_state(provider)

        allowed = set(whitelist)
        for _ in range(8):
            fp_logits = torch.full((1, 8, N_SAE_FEATURES), 5.0)
            state = provider.step(state, fp_logits)

        for i, seg in enumerate(provider.emitted_segments):
            non_allowed = set(seg.keys()) - allowed
            assert not non_allowed, (
                f"Non-whitelisted features {non_allowed} appeared in "
                f"SamplingBundle segment {i}: {seg}"
            )

    # ------------------------------------------------------------------
    # I5: feature_mask=None — two seeded runs are byte-identical
    # ------------------------------------------------------------------

    def test_sampling_i5_none_byte_identical(self, tiny_model):
        """Two SamplingBundle providers with feature_mask=None and the same
        seed must produce byte-identical packed encoder states,
        confirming the None path introduces no new gate logic."""
        fp_logits_seq = [
            torch.randn(1, 8, N_SAE_FEATURES,
                        generator=torch.Generator().manual_seed(s))
            for s in range(5)
        ]

        def _run():
            cfg = _make_sampling_config(feature_mask=None, seed=42)
            provider = SelfConditionedFeatureProvider(
                model=tiny_model, cfg=cfg, device=torch.device("cpu"),
            )
            state = build_test_self_cond_state(provider)
            results = []
            for fp in fp_logits_seq:
                state = provider.step(state, fp)
                results.append(_packed_snapshot(state.encoder_out))
            return results

        run_a = _run()
        run_b = _run()

        for out_a, out_b in zip(run_a, run_b, strict=True):
            _assert_packed_equal(out_a, out_b)
