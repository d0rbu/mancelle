"""Fail-fast guards: gradient-attribution ranking on LIVE selection paths.

Mirrors the feature_delta precedent in the opposite direction — gradient
ranking is precompute-only. Covers the central sampling guard (which
auto-protects every caller that merely threads the parsed config), the
dataloader entry guards, the parse-level rejections, and the checkpoint-leaf
helper's I2 byte-identity.
"""

from __future__ import annotations

import pytest
import torch as t

from errors import GradientRankingUnsupportedError
from steering.conditional.encoder_decoder.dataloader import (
    _assert_pipeline_supports_live_selection,
)
from steering.conditional.feature_pipeline import (
    append_gradient_leaf,
    build_feature_pipeline,
)
from steering.conditional.sampling import (
    SamplingSelection,
    assert_no_gradient_ranking,
    parse_feature_selection,
    sample_features,
    sample_features_batched,
)
from tests.steering.test_precompute_selected_features_mask import (
    N_SAE_FEATURES,
    _make_per_feature_stats,
)

_GRAD_FS = {
    "distribution": "geometric",
    "mean": 5.0,
    "mode": "top_k_percentile",
    "gradient": {"mode": "future_half_split"},
}
_LEGACY_FS = {
    "distribution": "geometric",
    "mean": 5.0,
    "mode": "top_k_percentile",
}


def _pipeline(fs_raw: dict):
    return build_feature_pipeline(
        feature_selection=parse_feature_selection(dict(fs_raw)),
        negative_selection=None,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=2,
        seed=1,
    )


class TestCentralSamplingGuard:
    """Every live caller threads the parsed config into sample_features* —
    the central guard makes them all safe even without site-level guards."""

    def setup_method(self):
        fs = parse_feature_selection(dict(_GRAD_FS))
        assert isinstance(fs, SamplingSelection)
        self.fs = fs
        self.table = t.arange(1, 100, dtype=t.float64).repeat(
            N_SAE_FEATURES, 1
        )
        self.rng = t.Generator().manual_seed(0)

    def test_sample_features_raises(self):
        with pytest.raises(GradientRankingUnsupportedError, match="offline-only"):
            sample_features(t.arange(4), t.ones(4), self.table, self.fs, self.rng)

    def test_sample_features_batched_raises(self):
        with pytest.raises(GradientRankingUnsupportedError, match="offline-only"):
            sample_features_batched(
                t.arange(4), t.ones(4), [4], self.table, self.fs, self.rng
            )

    def test_supplied_flag_allows_precompute_path(self):
        out = sample_features_batched(
            t.arange(4),
            t.ones(4),
            [4],
            self.table,
            self.fs,
            self.rng,
            override_weights_list=[t.ones(4)],
            gradient_rank_scores_supplied=True,
        )
        assert len(out) == 1

    def test_legacy_config_unaffected(self):
        fs = parse_feature_selection(dict(_LEGACY_FS))
        assert isinstance(fs, SamplingSelection)
        out = sample_features_batched(
            t.arange(4), t.ones(4), [4], self.table, fs, self.rng
        )
        assert len(out) == 1


class TestDataloaderGuards:
    def test_live_pipeline_guard_raises(self):
        pipeline = _pipeline(_GRAD_FS)
        with pytest.raises(GradientRankingUnsupportedError, match="offline-only"):
            _assert_pipeline_supports_live_selection(pipeline, "test live loader")

    def test_legacy_pipeline_passes(self):
        _assert_pipeline_supports_live_selection(_pipeline(_LEGACY_FS), "test")


class TestParseRejections:
    def test_weighted_mode_rejected(self):
        with pytest.raises(ValueError, match="top_k_percentile"):
            parse_feature_selection(
                {**_GRAD_FS, "mode": "weighted"}
            )

    def test_delta_weights_rejected(self):
        with pytest.raises(ValueError, match="delta_weights"):
            parse_feature_selection(
                {**_GRAD_FS, "delta_weights": "absolute_percentile"}
            )

    def test_threshold_selection_rejected(self):
        with pytest.raises(ValueError, match="threshold_type"):
            parse_feature_selection(
                {
                    "threshold_type": "segment_topk",
                    "threshold": 10,
                    "gradient": {"mode": "self"},
                }
            )

    def test_assert_no_gradient_ranking(self):
        fs = parse_feature_selection(dict(_GRAD_FS))
        with pytest.raises(GradientRankingUnsupportedError, match="my context"):
            assert_no_gradient_ranking(fs, "my context")
        assert_no_gradient_ranking(parse_feature_selection(dict(_LEGACY_FS)), "x")
        # Threshold selections never carry a gradient spec.
        thr = parse_feature_selection(
            {"threshold_type": "segment_topk", "threshold": 10}
        )
        assert_no_gradient_ranking(thr, "x")


class TestCheckpointLeaf:
    def test_legacy_path_byte_identical(self):
        fs = parse_feature_selection(dict(_LEGACY_FS))
        assert append_gradient_leaf("/ck/d12/s128/dist_g", fs) == "/ck/d12/s128/dist_g"

    def test_gradient_leaf_appended(self):
        fs = parse_feature_selection(dict(_GRAD_FS))
        assert append_gradient_leaf("/ck/d12", fs) == (
            "/ck/d12/grad_future_half_split_mf8"
        )

    def test_leaf_distinguishes_configs(self):
        leaves = set()
        for grad in [
            {"mode": "self"},
            {"mode": "future_half_split"},
            {"mode": "future_half_split", "min_final_segment_tokens": 4},
            {"mode": "future_half_split", "model_dtype": "float32"},
        ]:
            fs = parse_feature_selection({**_GRAD_FS, "gradient": grad})
            leaves.add(append_gradient_leaf("/ck", fs))
        assert len(leaves) == 4
