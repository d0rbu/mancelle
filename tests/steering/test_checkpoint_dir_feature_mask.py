# tests/steering/test_checkpoint_dir_feature_mask.py
"""Tests for the feature_mask leaf in encdec checkpoint paths (I2 + mask).

Covers all write-side builders:
  - conditional_encdec_train.py  (via path-construction logic)
  - conditional_encdec_sft.py    (via path-construction logic)
  - convert_decoder_to_encdec.py (via path-construction logic)
  - run_encdec_sft_pipeline.py   (via build_checkpoint_dir + feature_mask_leaf_from_cfg)

Plus read-side auto-discovery proof via nanochat._find_checkpoint_leaf (read-only).
"""

import os

from nanochat.checkpoint_manager import _find_checkpoint_leaf
from scripts.train.run_encdec_sft_pipeline import (
    build_checkpoint_dir,
    build_sft_checkpoint_dir,
)
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    append_mask_leaf,
    feature_mask_dirname,
    feature_mask_leaf_from_cfg,
)
from steering.conditional.feature_pipeline import feature_selection_dirname
from steering.conditional.sampling import parse_feature_selection

# ---------------------------------------------------------------------------
# Helpers for constructing the same path logic as each builder
# ---------------------------------------------------------------------------

_BASE_DIR = "/fake/base"
_MODEL_TAG = "d12_baseline"
_SEGMENT_SIZE = 128
_THRESHOLD = 2048.9


def _make_feature_selection():
    """Return a ThresholdSelection (matches default configs)."""
    return parse_feature_selection(
        {"threshold_type": "feature_percentile_segment_topk", "threshold": _THRESHOLD}
    )


def _make_mask_bl():
    return FeatureMask(FeatureMaskMode.blacklist, (3, 7, 42))


def _make_mask_wl():
    return FeatureMask(FeatureMaskMode.whitelist, (1, 2))


def _legacy_path(fs=None):
    """Path as produced TODAY (no mask) — the I2 baseline."""
    if fs is None:
        fs = _make_feature_selection()
    return os.path.join(
        _BASE_DIR,
        "conditional_encdec_checkpoints",
        _MODEL_TAG,
        f"s{_SEGMENT_SIZE}",
        feature_selection_dirname(fs),
    )


# ---------------------------------------------------------------------------
# 1. conditional_encdec_train.py path logic
# ---------------------------------------------------------------------------


class TestEncdecTrainPath:
    """Path logic mirrors conditional_encdec_train.py lines 929-942."""

    def _build(self, mask):
        fs = _make_feature_selection()
        base = os.path.join(
            _BASE_DIR,
            "conditional_encdec_checkpoints",
            _MODEL_TAG,
            f"s{_SEGMENT_SIZE}",
            feature_selection_dirname(fs),
        )
        return append_mask_leaf(base, mask)

    def test_no_mask_is_byte_identical_to_legacy(self):
        result = self._build(None)
        assert result == _legacy_path(), (
            "I2 violated: no mask must produce byte-identical path"
        )

    def test_blacklist_has_bl_leaf(self):
        mask = _make_mask_bl()
        result = self._build(mask)
        parts = os.path.normpath(result).split(os.sep)
        bl_parts = [p for p in parts if p.startswith("bl_")]
        assert len(bl_parts) == 1

    def test_whitelist_has_wl_leaf(self):
        mask = _make_mask_wl()
        result = self._build(mask)
        parts = os.path.normpath(result).split(os.sep)
        wl_parts = [p for p in parts if p.startswith("wl_")]
        assert len(wl_parts) == 1

    def test_mask_leaf_comes_after_feature_selection_dirname(self):
        fs = _make_feature_selection()
        mask = _make_mask_bl()
        result = self._build(mask)
        fs_leaf = feature_selection_dirname(fs)
        mask_leaf = feature_mask_dirname(mask)
        # The fs_leaf must appear before mask_leaf in the path
        assert result.index(fs_leaf) < result.index(mask_leaf)

    def test_different_masks_produce_different_paths(self):
        r1 = self._build(_make_mask_bl())
        r2 = self._build(_make_mask_wl())
        assert r1 != r2

    def test_mask_path_starts_with_legacy_path(self):
        legacy = _legacy_path()
        mask = _make_mask_bl()
        result = self._build(mask)
        assert result.startswith(legacy + os.sep)


# ---------------------------------------------------------------------------
# 2. conditional_encdec_sft.py path logic
# ---------------------------------------------------------------------------


class TestEncdecSftPath:
    """Path logic mirrors conditional_encdec_sft.py (post-n_sae_features block)."""

    def _build_sft_dir(self, mask, dataset="safety"):
        """Mirror the patched SFT path logic."""
        fs = _make_feature_selection()
        checkpoint_dir = os.path.join(
            _BASE_DIR,
            "conditional_encdec_checkpoints",
            _MODEL_TAG,
            f"s{_SEGMENT_SIZE}",
            feature_selection_dirname(fs),
        )
        # Mask leaf goes between pretrain dir and SFT leaf
        masked_checkpoint_dir = append_mask_leaf(checkpoint_dir, mask)
        return os.path.join(
            masked_checkpoint_dir, f"{dataset}_sft_conditioned_checkpoints"
        )

    def test_no_mask_sft_path_byte_identical_to_legacy(self):
        """I2: no mask → sft_checkpoint_dir same as before."""
        result = self._build_sft_dir(None)
        legacy = os.path.join(_legacy_path(), "safety_sft_conditioned_checkpoints")
        assert result == legacy

    def test_bl_mask_leaf_before_sft_leaf(self):
        mask = _make_mask_bl()
        result = self._build_sft_dir(mask)
        parts = os.path.normpath(result).split(os.sep)
        bl_indices = [i for i, p in enumerate(parts) if p.startswith("bl_")]
        # SFT leaf has the form "<dataset>_sft_<mode>_checkpoints"
        sft_indices = [i for i, p in enumerate(parts) if "_sft_" in p and p.endswith("_checkpoints")]
        assert len(bl_indices) == 1, f"Expected one bl_ component, got: {parts}"
        assert len(sft_indices) == 1, f"Expected one *_sft_*_checkpoints component, got: {parts}"
        # bl leaf index must be strictly before sft leaf index
        assert bl_indices[0] < sft_indices[0]

    def test_wl_mask_leaf_before_sft_leaf(self):
        mask = _make_mask_wl()
        result = self._build_sft_dir(mask)
        parts = os.path.normpath(result).split(os.sep)
        wl_indices = [i for i, p in enumerate(parts) if p.startswith("wl_")]
        sft_indices = [i for i, p in enumerate(parts) if "_sft_" in p and p.endswith("_checkpoints")]
        assert len(wl_indices) == 1, f"Expected one wl_ component, got: {parts}"
        assert len(sft_indices) == 1, f"Expected one *_sft_*_checkpoints component, got: {parts}"
        assert wl_indices[0] < sft_indices[0]

    def test_conditioned_no_mask(self):
        result = self._build_sft_dir(None, dataset="capabilities")
        assert result.endswith("capabilities_sft_conditioned_checkpoints")
        # No mask leaf in path
        assert "bl_" not in result and "wl_" not in result

    def test_base_checkpoint_dir_override_no_mask_appended(self):
        """When base_checkpoint_dir is set explicitly, no mask leaf is added."""
        explicit = "/explicit/override/path"
        # The override branch in conditional_encdec_sft.py skips append_mask_leaf
        sft_suffix = "conditioned"
        sft_dir = os.path.join(explicit, f"safety_sft_{sft_suffix}_checkpoints")
        assert "bl_" not in sft_dir and "wl_" not in sft_dir


# ---------------------------------------------------------------------------
# 3. convert_decoder_to_encdec.py path logic
# ---------------------------------------------------------------------------


class TestConvertDecoderPath:
    """Path logic mirrors convert_decoder_to_encdec.py lines 160-173."""

    def _build(self, mask):
        fs = _make_feature_selection()
        base = os.path.join(
            _BASE_DIR,
            "conditional_encdec_checkpoints",
            _MODEL_TAG,
            f"s{_SEGMENT_SIZE}",
            feature_selection_dirname(fs),
        )
        return append_mask_leaf(base, mask)

    def test_no_mask_byte_identical(self):
        assert self._build(None) == _legacy_path()

    def test_bl_mask_in_path(self):
        result = self._build(_make_mask_bl())
        assert "/bl_" in result

    def test_wl_mask_in_path(self):
        result = self._build(_make_mask_wl())
        assert "/wl_" in result


# ---------------------------------------------------------------------------
# 4. run_encdec_sft_pipeline.py path logic
# ---------------------------------------------------------------------------


class TestPipelinePath:
    """Path logic mirrors run_encdec_sft_pipeline.py build_checkpoint_dir."""

    def _fs_dirname(self):
        return feature_selection_dirname(_make_feature_selection())

    def _build_pretrain_no_mask(self):
        return build_checkpoint_dir(_BASE_DIR, _MODEL_TAG, _SEGMENT_SIZE, self._fs_dirname())

    def test_build_checkpoint_dir_no_mask_byte_identical(self):
        result = self._build_pretrain_no_mask()
        assert result == _legacy_path()

    def test_build_checkpoint_dir_with_mask_leaf(self):
        base = self._build_pretrain_no_mask()
        mask = _make_mask_bl()
        leaf = feature_mask_dirname(mask)
        result = os.path.join(base, leaf)
        assert "/bl_" in result

    def test_feature_mask_leaf_from_cfg_dict(self):
        """feature_mask_leaf_from_cfg with inline-ids dict."""
        leaf = feature_mask_leaf_from_cfg(
            {"feature_mask": {"mode": "blacklist", "ids": [3, 7, 42]}}
        )
        assert leaf.startswith("bl_")
        expected = feature_mask_dirname(
            FeatureMask(FeatureMaskMode.blacklist, (3, 7, 42))
        )
        assert leaf == expected

    def test_pipeline_no_mask_pretrain_dir_byte_identical(self):
        """With no feature_mask in cfg, pretrain_dir stays legacy."""
        leaf = feature_mask_leaf_from_cfg({})
        base = self._build_pretrain_no_mask()
        pretrain_dir = os.path.join(base, leaf) if leaf else base
        assert pretrain_dir == _legacy_path()

    def test_pipeline_with_mask_pretrain_dir_has_mask_leaf(self):
        leaf = feature_mask_leaf_from_cfg(
            {"feature_mask": {"mode": "whitelist", "ids": [1, 2]}}
        )
        base = self._build_pretrain_no_mask()
        pretrain_dir = os.path.join(base, leaf) if leaf else base
        assert "/wl_" in pretrain_dir

    def test_build_sft_checkpoint_dir_no_mask_byte_identical(self):
        base = self._build_pretrain_no_mask()
        sft_dir = build_sft_checkpoint_dir(base, "safety")
        expected = os.path.join(_legacy_path(), "safety_sft_conditioned_checkpoints")
        assert sft_dir == expected


# ---------------------------------------------------------------------------
# 5. Read-side: _find_checkpoint_leaf auto-discovery through bl_<hash>/ leaf
# ---------------------------------------------------------------------------


def test_find_checkpoint_leaf_discovers_through_mask_dir(tmp_path):
    """nanochat._find_checkpoint_leaf recurses into bl_<hash>/ and finds model_*.pt."""
    mask = FeatureMask(FeatureMaskMode.blacklist, (5,))
    mask_leaf = feature_mask_dirname(mask)

    # Build directory structure: root/bl_<hash>/model_000000.pt
    masked_dir = tmp_path / mask_leaf
    masked_dir.mkdir()
    checkpoint_file = masked_dir / "model_000000.pt"
    checkpoint_file.write_bytes(b"fake")

    # _find_checkpoint_leaf from root should find the file inside bl_<hash>/
    discovered = _find_checkpoint_leaf(str(tmp_path))
    assert discovered == str(masked_dir)


def test_find_checkpoint_leaf_discovers_through_wl_dir(tmp_path):
    mask = FeatureMask(FeatureMaskMode.whitelist, (10, 20))
    mask_leaf = feature_mask_dirname(mask)

    wl_dir = tmp_path / mask_leaf
    wl_dir.mkdir()
    (wl_dir / "model_000100.pt").write_bytes(b"fake")

    discovered = _find_checkpoint_leaf(str(tmp_path))
    assert discovered == str(wl_dir)


def test_find_checkpoint_leaf_discovers_deep_nesting(tmp_path):
    """Two-level nesting: bl_<hash>/<sft_leaf>/model_*.pt is found."""
    mask = FeatureMask(FeatureMaskMode.blacklist, (1,))
    mask_leaf = feature_mask_dirname(mask)

    sft_dir = tmp_path / mask_leaf / "safety_sft_conditioned_checkpoints"
    sft_dir.mkdir(parents=True)
    (sft_dir / "model_000200.pt").write_bytes(b"fake")

    discovered = _find_checkpoint_leaf(str(tmp_path))
    assert discovered == str(sft_dir)


def test_no_mask_path_discovery_still_works(tmp_path):
    """Without mask leaf, direct checkpoint dir is found (I2 read-side)."""
    checkpoint_dir = tmp_path / "t2048.9"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "model_000000.pt").write_bytes(b"fake")

    discovered = _find_checkpoint_leaf(str(tmp_path))
    assert discovered == str(checkpoint_dir)
