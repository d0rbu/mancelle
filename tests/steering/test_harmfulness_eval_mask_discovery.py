"""Tests: pinned-discovery mask-leaf gap in harmfulness_eval.

Verifies that:
  1. Pinned feature-selection paths append the bl_/wl_ leaf
     when cfg carries a feature_mask block.
  2. Paths are byte-identical to pre-mask layout when no mask is configured
     (mask_leaf="" → no extra component).
  3. Auto-discovery path is NOT modified (no mask_leaf appended; _find_checkpoint_leaf
     already recurses into bl_/wl_ leaves).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nanochat.checkpoint_manager import get_checkpoint_dir
from scripts.eval.harmfulness_eval import (
    _resolve_checkpoint_dir_for_harmfulness,
    resolve_harmfulness_paths,
)
from steering.conditional.feature_mask import feature_mask_leaf_from_cfg
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FEATURE_SELECTION_DICT = {
    "distribution": "geometric",
    "mean": 10.0,
    "std": 10.0,
    "min_k": 1,
    "max_k": 40,
    "mode": "top_k_percentile",
}

_MASK_CFG_BL = {"mode": "blacklist", "ids": [42, 137]}
_MASK_CFG_WL = {"mode": "whitelist", "ids": [7, 99]}


def _threshold_feature_selection(threshold: float) -> dict[str, object]:
    return {"threshold_type": "segment_topk", "threshold": threshold}


def _mask_leaf(mask_dict: dict) -> str:
    """Expected leaf for a given raw mask dict."""
    return feature_mask_leaf_from_cfg({"feature_mask": mask_dict})


# ---------------------------------------------------------------------------
# _resolve_checkpoint_dir_for_harmfulness — pinned feature_selection branch
# ---------------------------------------------------------------------------


class TestPinnedFeatureSelectionMaskLeaf:
    """Pinned (feature_selection) path appends mask leaf; no mask → identity."""

    def test_mask_leaf_appended_for_blacklist(self, tmp_path: Path) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_BL)
        assert expected_leaf.startswith("bl_")

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            path = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                mask_leaf=expected_leaf,
            )

        assert path.endswith(os.sep + expected_leaf)

    def test_mask_leaf_appended_for_whitelist(self, tmp_path: Path) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_WL)
        assert expected_leaf.startswith("wl_")

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            path = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                mask_leaf=expected_leaf,
            )

        assert path.endswith(os.sep + expected_leaf)

    def test_no_mask_leaf_byte_identical(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            path_no_mask = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                mask_leaf="",
            )
            path_default = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                # mask_leaf omitted — default ""
            )

        assert path_no_mask == path_default
        # Leaf must NOT be bl_* or wl_*
        leaf = os.path.basename(path_no_mask)
        assert not leaf.startswith("bl_")
        assert not leaf.startswith("wl_")

    def test_mask_leaf_is_last_component(self, tmp_path: Path) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_BL)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            masked = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                mask_leaf=expected_leaf,
            )
            unmasked = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                mask_leaf="",
            )

        assert masked == os.path.join(unmasked, expected_leaf)


# ---------------------------------------------------------------------------
# _resolve_checkpoint_dir_for_harmfulness — threshold-form feature selection
# ---------------------------------------------------------------------------


class TestPinnedThresholdSelectionMaskLeaf:
    """A threshold-form feature-selection path also appends the mask leaf."""

    def test_modern_threshold_block_preserves_threshold_checkpoint_path(
        self, tmp_path: Path
    ) -> None:
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            path = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_threshold_feature_selection(2048.9),
            )
            old_scalar_alias_path = get_checkpoint_dir(
                "encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=ThresholdSelection(
                    threshold_type=ThresholdType.segment_topk,
                    threshold=2048.9,
                ),
            )
        assert path == old_scalar_alias_path

    def test_mask_leaf_appended_for_blacklist(self, tmp_path: Path) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_BL)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            path = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_threshold_feature_selection(2048.9),
                mask_leaf=expected_leaf,
            )

        assert path.endswith(os.sep + expected_leaf)

    def test_no_mask_byte_identical(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            path_no_mask = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_threshold_feature_selection(2048.9),
                mask_leaf="",
            )
            path_default = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_threshold_feature_selection(2048.9),
            )

        assert path_no_mask == path_default
        leaf = os.path.basename(path_no_mask)
        assert not leaf.startswith("bl_")
        assert not leaf.startswith("wl_")

    def test_mask_leaf_is_last_component(self, tmp_path: Path) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_WL)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            masked = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_threshold_feature_selection(2048.9),
                mask_leaf=expected_leaf,
            )
            unmasked = _resolve_checkpoint_dir_for_harmfulness(
                source="encdec",
                model_tag="d26",
                segment_size=128,
                feature_selection=_threshold_feature_selection(2048.9),
                mask_leaf="",
            )

        assert masked == os.path.join(unmasked, expected_leaf)


# ---------------------------------------------------------------------------
# resolve_harmfulness_paths — mask_leaf flows through to report dir
# ---------------------------------------------------------------------------


class TestResolveHarmfulnessPathsMaskLeaf:
    """mask_leaf propagates through resolve_harmfulness_paths into the report dir."""

    def test_feature_selection_with_mask_leaf_embeds_in_report_dir(
        self, tmp_path: Path
    ) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_BL)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            resp, out = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                config_name="features",
                mask_leaf=expected_leaf,
            )

        # The mask leaf must appear as a path component before "report/"
        assert expected_leaf in out
        assert expected_leaf in resp
        assert out.index(expected_leaf) < out.index("report")

    def test_no_mask_leaf_paths_are_unchanged(self, tmp_path: Path) -> None:
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            resp_masked_empty, out_masked_empty = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_threshold_feature_selection(10),
                config_name="features",
                mask_leaf="",
            )
            resp_default, out_default = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_threshold_feature_selection(10),
                config_name="features",
                # mask_leaf omitted
            )

        assert resp_masked_empty == resp_default
        assert out_masked_empty == out_default

    def test_masked_and_unmasked_paths_differ(self, tmp_path: Path) -> None:
        expected_leaf = _mask_leaf(_MASK_CFG_BL)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            _resp_masked, out_masked = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_threshold_feature_selection(10),
                config_name="features",
                mask_leaf=expected_leaf,
            )
            _resp_unmasked, out_unmasked = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_threshold_feature_selection(10),
                config_name="features",
            )

        assert out_masked != out_unmasked
        assert out_masked == out_unmasked.replace(
            os.path.join("t10.0", "report"),
            os.path.join("t10.0", expected_leaf, "report"),
        )

    def test_distinct_masks_produce_distinct_paths(self, tmp_path: Path) -> None:
        """Two different masks → two distinct checkpoint dirs (no-clobber)."""
        leaf_bl = _mask_leaf(_MASK_CFG_BL)
        leaf_wl = _mask_leaf(_MASK_CFG_WL)
        assert leaf_bl != leaf_wl

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            _, out_bl = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_threshold_feature_selection(10),
                config_name="features",
                mask_leaf=leaf_bl,
            )
            _, out_wl = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=None,
                responses_file=None,
                segment_size=128,
                feature_selection=_threshold_feature_selection(10),
                config_name="features",
                mask_leaf=leaf_wl,
            )

        assert out_bl != out_wl

    def test_explicit_output_file_ignores_mask_leaf(self, tmp_path: Path) -> None:
        """Explicit output_file always wins — mask_leaf is irrelevant."""
        custom = str(tmp_path / "custom.yaml")
        expected_leaf = _mask_leaf(_MASK_CFG_BL)

        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            _resp, out = resolve_harmfulness_paths(
                source="encdec",
                model_tag="d26",
                output_file=custom,
                responses_file=None,
                segment_size=128,
                feature_selection=_FEATURE_SELECTION_DICT,
                config_name="features",
                mask_leaf=expected_leaf,
            )

        assert out == custom


# ---------------------------------------------------------------------------
# feature_mask_leaf_from_cfg integration — the function used at call sites
# ---------------------------------------------------------------------------


class TestFeatureMaskLeafFromCfgIntegration:
    """feature_mask_leaf_from_cfg returns '' for cfg without feature_mask."""

    def test_cfg_without_feature_mask_returns_empty(self) -> None:
        cfg = {"source": "encdec", "segment_size": 128}
        assert feature_mask_leaf_from_cfg(cfg) == ""

    def test_cfg_with_null_feature_mask_returns_empty(self) -> None:
        cfg = {"feature_mask": None}
        assert feature_mask_leaf_from_cfg(cfg) == ""

    def test_cfg_with_blacklist_returns_bl_leaf(self) -> None:
        cfg = {"feature_mask": {"mode": "blacklist", "ids": [1, 2, 3]}}
        leaf = feature_mask_leaf_from_cfg(cfg)
        assert leaf.startswith("bl_")
        assert len(leaf) == 19  # "bl_" + 16 hex chars

    def test_cfg_with_whitelist_returns_wl_leaf(self) -> None:
        cfg = {"feature_mask": {"mode": "whitelist", "ids": [10]}}
        leaf = feature_mask_leaf_from_cfg(cfg)
        assert leaf.startswith("wl_")
        assert len(leaf) == 19  # "wl_" + 16 hex chars

    def test_same_mask_same_leaf_deterministic(self) -> None:
        cfg_a = {"feature_mask": {"mode": "blacklist", "ids": [42, 137]}}
        cfg_b = {"feature_mask": {"mode": "blacklist", "ids": [42, 137]}}
        assert feature_mask_leaf_from_cfg(cfg_a) == feature_mask_leaf_from_cfg(cfg_b)

    def test_different_ids_produce_different_leaves(self) -> None:
        cfg_a = {"feature_mask": {"mode": "blacklist", "ids": [1]}}
        cfg_b = {"feature_mask": {"mode": "blacklist", "ids": [2]}}
        assert feature_mask_leaf_from_cfg(cfg_a) != feature_mask_leaf_from_cfg(cfg_b)
