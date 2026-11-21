"""H9: axbench Stage-4 extraction feature_mask threading tests.

Covers:
1. Cache-signature I1-analog: no-mask signature is byte-identical to pre-H9
   (FROZEN golden guard prevents orphaning existing extraction caches).
2. Masked ⇒ distinct + canonical: different modes/ids yield distinct sigs;
   id-order invariance via canonical_dict.
3. Metadata threading: detect_target_model_metadata flows feature_mask from
   user_config; decoder-only ⇒ None.
4. Final-list enforcement: blacklist/whitelist applied, assert_no_masked fires
   on violation, mask None ⇒ result byte-identical (I5).
5. I5 no-op: mask=None path skips all mask construction/filter/assert.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as t

from errors import FeatureMaskError
from steering.conditional.feature_mask import resolve_feature_mask
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

# Number of features in our synthetic metadata (small enough for id range)
_N_FEATURES = 16


def _make_encdec_model(
    *,
    num_buckets: int = 4,
    n_sae_features: int = _N_FEATURES,
    user_config: dict | None = None,
) -> SimpleNamespace:
    """Lightweight stand-in for an EncoderDecoderGPT."""
    encoder_config = SimpleNamespace(
        num_buckets=num_buckets,
        n_sae_features=n_sae_features,
    )
    model = SimpleNamespace(encoder=SimpleNamespace(config=encoder_config))
    model._loaded_meta_data = {"user_config": user_config or {}}
    return model


def _make_decoder_only_model() -> SimpleNamespace:
    """Stand-in for a decoder-only GPT — no ``.encoder`` attribute."""
    return SimpleNamespace(transformer=SimpleNamespace(h=[]))


# ---------------------------------------------------------------------------
# 3. Metadata threading
# ---------------------------------------------------------------------------


class TestMetadataThreading:
    """detect_target_model_metadata flows feature_mask from user_config."""

    def test_encdec_with_feature_mask_in_user_config(self):
        """A checkpoint with feature_mask in user_config → field propagated."""
        mask_cfg = {"mode": "blacklist", "ids": [1, 2, 3]}
        model = _make_encdec_model(user_config={"feature_mask": mask_cfg})
        cfg = make_valid_config()
        meta = detect_target_model_metadata(model, cfg)
        assert meta.feature_mask == mask_cfg

    def test_encdec_without_feature_mask_in_user_config(self):
        """A checkpoint with no feature_mask key ⇒ field is None."""
        model = _make_encdec_model(user_config={"feature_selection": {"threshold_type": "segment_topk", "threshold": 10}})
        cfg = make_valid_config()
        meta = detect_target_model_metadata(model, cfg)
        assert meta.feature_mask is None

    def test_encdec_empty_user_config(self):
        """Empty user_config ⇒ feature_mask is None."""
        model = _make_encdec_model(user_config={})
        cfg = make_valid_config()
        meta = detect_target_model_metadata(model, cfg)
        assert meta.feature_mask is None

    def test_decoder_only_feature_mask_is_none(self):
        """Decoder-only targets always have feature_mask=None."""
        model = _make_decoder_only_model()
        cfg = make_valid_config()
        meta = detect_target_model_metadata(model, cfg)
        assert meta.feature_mask is None

    def test_feature_mask_whitelist_propagated_as_raw_dict(self):
        """Whitelist mask survives the detection round-trip as a raw dict."""
        mask_cfg = {"mode": "whitelist", "ids": [7, 8, 9]}
        model = _make_encdec_model(user_config={"feature_mask": mask_cfg})
        cfg = make_valid_config()
        meta = detect_target_model_metadata(model, cfg)
        assert meta.feature_mask == mask_cfg


# ---------------------------------------------------------------------------
# 4. Final-list enforcement
# ---------------------------------------------------------------------------


class TestFinalListEnforcement:
    """Mask enforcement on the final (fid, bucket) list."""

    def test_defense_in_depth_assert_raises_on_poisoned_result(self):
        """assert_no_masked raises FeatureMaskError when a blacklisted fid
        is present — the defense-in-depth guard catches regressions loudly."""
        fm = resolve_feature_mask(
            {"feature_mask": {"mode": "blacklist", "ids": [5]}},
            n_sae_features=_N_FEATURES,
        )
        assert fm is not None
        # A result containing fid 5 (blacklisted) must raise.
        poisoned_fids = t.tensor([1, 3, 5, 7], dtype=t.int64)
        with pytest.raises(FeatureMaskError, match="feature_mask violated"):
            fm.assert_no_masked(poisoned_fids)
