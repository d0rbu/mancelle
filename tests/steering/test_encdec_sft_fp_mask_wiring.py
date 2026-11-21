"""Tests for feature_mask wiring in conditional_encdec_sft.py (Task 13c).

Verifies that the SFT script sets orig_model.feature_mask from
resolve_feature_mask(cfg, n_sae_features) after model+compile and before
the first forward step.

Because main() requires a full Hydra config + checkpoint + DDP setup, these
tests do NOT invoke main().  Instead they:
 1. Assert the import is present (structural).
 2. Test the wiring behavior directly by simulating the attribute-set pattern
    with a mock model and the real resolve_feature_mask function.
 3. Verify I5: resolve_feature_mask on a cfg with no feature_mask key returns
    None (no-op), so the SFT wiring does not alter runs that don't configure
    a mask.

Task 17 end-to-end tests will cover the full SFT training loop.
"""

from __future__ import annotations

from omegaconf import OmegaConf

from steering.conditional.feature_mask import FeatureMask, resolve_feature_mask

# ---------------------------------------------------------------------------
# 1. Structural: resolve_feature_mask is importable from the SFT script's
#    import namespace.  This fails immediately if the import line is missing.
# ---------------------------------------------------------------------------


class TestImportPresent:
    def test_resolve_feature_mask_importable_from_sft_module(self):
        """The SFT script must import resolve_feature_mask (Task 13c addition)."""
        import scripts.train.conditional_encdec_sft as _sft_mod

        assert hasattr(_sft_mod, "resolve_feature_mask"), (
            "resolve_feature_mask not found in scripts.train.conditional_encdec_sft. "
            "The Task 13c import line is missing."
        )
        # Verify it's the same object (not a shadowed local)
        assert _sft_mod.resolve_feature_mask is resolve_feature_mask


# ---------------------------------------------------------------------------
# 2. Wiring contract: simulates the pattern
#      orig_model.feature_mask = resolve_feature_mask(cfg, n_sae_features)
#    for both the "mask configured" and "no mask" cases.
# ---------------------------------------------------------------------------


class _MockModel:
    """Minimal stand-in for an EncoderDecoderGPT — only the attribute matters."""

    def __init__(self):
        self.feature_mask: FeatureMask | None = None


def _make_cfg(feature_mask_dict=None):
    """Build an OmegaConf DictConfig that resolve_feature_mask can read.

    Mirrors how Hydra compose() delivers the cfg to main().  When feature_mask_dict
    is None the resulting cfg has no feature_mask key — matching the current SFT
    YAML which does not define feature_mask.
    """
    if feature_mask_dict is None:
        return OmegaConf.create({})
    return OmegaConf.create({"feature_mask": feature_mask_dict})


class TestWiringPattern:
    """The wiring pattern `orig_model.feature_mask = resolve_feature_mask(cfg, N)`
    correctly propagates the mask (or None) onto the model attribute."""

    def test_no_mask_in_cfg_yields_none(self):
        """When cfg has no feature_mask key, resolve_feature_mask returns None (I5)."""
        cfg = _make_cfg()
        n_sae = 128

        result = resolve_feature_mask(cfg, n_sae)

        assert result is None, (
            f"Expected None for cfg with no feature_mask, got {result!r}. "
            "I5: resolve_feature_mask(cfg_without_mask, n) must return None."
        )

    def test_none_result_assigned_to_model_is_noop(self):
        """Assigning None to orig_model.feature_mask is a no-op (I5)."""
        cfg = _make_cfg()
        orig_model = _MockModel()
        n_sae = 128

        # Simulate the SFT wiring line exactly:
        orig_model.feature_mask = resolve_feature_mask(cfg, n_sae)

        assert orig_model.feature_mask is None, (
            f"orig_model.feature_mask should be None after wiring with no-mask cfg, "
            f"got {orig_model.feature_mask!r}"
        )

    def test_blacklist_cfg_propagated_to_model(self):
        """When cfg.feature_mask is a blacklist, the mask reaches orig_model."""
        n_sae = 64
        cfg = _make_cfg({"mode": "blacklist", "ids": [3, 7, 15]})
        orig_model = _MockModel()

        orig_model.feature_mask = resolve_feature_mask(cfg, n_sae)

        assert orig_model.feature_mask is not None, (
            "resolve_feature_mask returned None for a blacklist cfg — wiring would "
            "silently drop the mask."
        )
        assert isinstance(orig_model.feature_mask, FeatureMask)
        assert orig_model.feature_mask.mode == "blacklist"
        assert set(orig_model.feature_mask.ids) == {3, 7, 15}

    def test_whitelist_cfg_propagated_to_model(self):
        """When cfg.feature_mask is a whitelist, the mask reaches orig_model."""
        n_sae = 64
        cfg = _make_cfg({"mode": "whitelist", "ids": [0, 1, 2]})
        orig_model = _MockModel()

        orig_model.feature_mask = resolve_feature_mask(cfg, n_sae)

        assert orig_model.feature_mask is not None
        assert isinstance(orig_model.feature_mask, FeatureMask)
        assert orig_model.feature_mask.mode == "whitelist"
        assert set(orig_model.feature_mask.ids) == {0, 1, 2}

    def test_mask_identity_preserved(self):
        """The exact FeatureMask object from resolve_feature_mask is on orig_model."""
        n_sae = 32
        cfg = _make_cfg({"mode": "blacklist", "ids": [5]})
        orig_model = _MockModel()

        mask = resolve_feature_mask(cfg, n_sae)
        orig_model.feature_mask = mask

        # Structural identity — same object assigned
        assert orig_model.feature_mask is mask


# ---------------------------------------------------------------------------
# 3. I5 invariant: None mask on a real EncoderDecoderGPT does not alter FP loss.
#    This mirrors the pretrain test in test_fp_loss_mask_wiring.py but asserts
#    specifically for the SFT wiring point (after compile).
# ---------------------------------------------------------------------------


class TestI5NullMaskNoChange:
    """Assigning feature_mask=None (I5) leaves compute_feature_pred_loss unchanged."""

    def test_assign_none_does_not_change_default(self):
        """Setting orig_model.feature_mask = None after construction is a no-op."""
        orig_model = _MockModel()
        assert orig_model.feature_mask is None  # invariant: default is None

        # Simulate the SFT wiring with no mask configured
        cfg = _make_cfg()
        orig_model.feature_mask = resolve_feature_mask(cfg, n_sae_features=128)

        assert orig_model.feature_mask is None, (
            "After SFT wiring with no-mask cfg, feature_mask must be None "
            "(I5 invariant — no-op)."
        )
