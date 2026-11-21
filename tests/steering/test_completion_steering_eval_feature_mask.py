"""H8: feature_mask threading into completion_steering_eval — comprehensive tests.

Covers:
  1. I5 byte-identical (regression guard): no-kwarg == explicit None == hardcoded golden
  2. Blacklist excludes: no blacklisted fid in selections
  3. Whitelist restricts: only whitelisted fids appear
  4. All-masked ⇒ empty (no crash, no RNG error)
  5. assert_selections_within_mask: raises FeatureMaskError for violation; no-op for None / empty
  6. Config round-trip: both YAMLs load feature_mask as None
  7. Pipeline auto-resolve: build_feature_pipeline_from_cfg_block wires feature_mask correctly
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
import yaml
from omegaconf import OmegaConf

from errors import FeatureMaskError
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.eval.completion_steering_eval import (
    EvalMode,
    FeatureSelection,
    RawFeatureSelection,
    ScanResultTensors,
    assert_selections_within_mask,
    select_from_tensors,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Shared synthetic-data helpers
# ---------------------------------------------------------------------------

_CONFIGS_EVAL_DIR = Path(__file__).parent.parent.parent / "configs" / "eval"


def _make_scan_tensors(
    n_features: int = 10, docs_per_feature: int = 3
) -> ScanResultTensors:
    """Create synthetic ScanResultTensors with sequential feature ids 0..n-1."""
    feature_ids = t.arange(n_features, dtype=t.int64)
    n_docs_activated = t.full((n_features,), docs_per_feature, dtype=t.int64)
    total_acts = n_features * docs_per_feature
    offsets = t.arange(0, total_acts + 1, docs_per_feature, dtype=t.int64)
    doc_idx = t.zeros(total_acts, dtype=t.int64)
    max_activation = t.ones(total_acts, dtype=t.float32)
    max_segment_idx = t.ones(total_acts, dtype=t.int64)
    percentile = t.full((total_acts,), 0.5, dtype=t.float32)
    total_segments = t.full((total_acts,), 3, dtype=t.int64)
    for i in range(n_features):
        start = i * docs_per_feature
        doc_idx[start : start + docs_per_feature] = t.arange(
            docs_per_feature, dtype=t.int64
        )
    return ScanResultTensors(
        feature_ids=feature_ids,
        n_docs_activated=n_docs_activated,
        offsets=offsets,
        doc_idx=doc_idx,
        max_activation=max_activation,
        max_segment_idx=max_segment_idx,
        percentile=percentile,
        total_segments=total_segments,
    )


def _make_per_feature_stats(n: int) -> dict[str, PerFeatureStat]:
    """Build minimal per-feature stats for n features (used in pipeline tests)."""
    pcts = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    return {
        str(i): PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(pcts),
        )
        for i in range(n)
    }


# ---------------------------------------------------------------------------
# 1. I5 byte-identical regression guard
# ---------------------------------------------------------------------------

# FROZEN GOLDEN: computed once with the new select_from_tensors (feature_mask=None
# path) against a 10-feature, 3-docs-per-feature synthetic scan. Any accidental
# change to the no-mask RNG path will break these assertions immediately.
_I5_GOLDENS: dict[int, list[int]] = {
    42: [2, 6, 1, 8, 4],
    123: [2, 0, 8, 1, 3],
    7: [5, 0, 3, 4, 1],
}


class TestI5ByteIdentical:
    """Regression guard: no-mask path must be byte-identical before and after H8."""

    def test_no_kwarg_matches_explicit_none(self) -> None:
        """Calling without feature_mask kwarg == passing feature_mask=None."""
        scan = _make_scan_tensors(10, 3)
        for seed in [42, 123, 7]:
            r_no_kwarg = select_from_tensors(
                scan, top_k=5, docs_per_feature=2, mode=EvalMode.ABLATION, seed=seed
            )
            r_none = select_from_tensors(
                scan,
                top_k=5,
                docs_per_feature=2,
                mode=EvalMode.ABLATION,
                seed=seed,
                feature_mask=None,
            )
            fids_no_kwarg = [s.feature_id for s in r_no_kwarg]
            fids_none = [s.feature_id for s in r_none]
            assert fids_no_kwarg == fids_none, (
                f"seed={seed}: no-kwarg path differs from explicit None path "
                f"({fids_no_kwarg} vs {fids_none}) — I5 violated"
            )
            # Full selection equality (docs + feature_id)
            for a, b in zip(r_no_kwarg, r_none, strict=True):
                assert a.feature_id == b.feature_id
                assert a.selected_docs == b.selected_docs

    def test_no_mask_matches_frozen_golden(self) -> None:
        """No-mask path must match the pre-H8 frozen golden for 3 seeds."""
        scan = _make_scan_tensors(10, 3)
        for seed, expected_fids in _I5_GOLDENS.items():
            result = select_from_tensors(
                scan,
                top_k=5,
                docs_per_feature=2,
                mode=EvalMode.ABLATION,
                seed=seed,
                feature_mask=None,
            )
            actual_fids = [s.feature_id for s in result]
            assert actual_fids == expected_fids, (
                f"I5 golden drift at seed={seed}: "
                f"expected {expected_fids}, got {actual_fids}. "
                "No-mask RNG path changed — H8 regression!"
            )

    def test_full_list_equality_none_vs_no_kwarg(self) -> None:
        """Complete list (feature_id + selected_docs + n_docs_activated) must match."""
        scan = _make_scan_tensors(10, 3)
        seed = 42
        r_no_kwarg = select_from_tensors(
            scan, top_k=5, docs_per_feature=3, mode=EvalMode.ABLATION, seed=seed
        )
        r_none = select_from_tensors(
            scan,
            top_k=5,
            docs_per_feature=3,
            mode=EvalMode.ABLATION,
            seed=seed,
            feature_mask=None,
        )
        assert len(r_no_kwarg) == len(r_none)
        for a, b in zip(r_no_kwarg, r_none, strict=True):
            assert a.feature_id == b.feature_id
            assert a.n_docs_activated == b.n_docs_activated
            assert a.selected_docs == b.selected_docs


# ---------------------------------------------------------------------------
# 2. Blacklist excludes
# ---------------------------------------------------------------------------


class TestBlacklist:
    def test_blacklisted_features_absent(self) -> None:
        """No selection has a feature_id in the blacklisted set."""
        scan = _make_scan_tensors(10, 3)
        blacklisted = {2, 6, 8}
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=tuple(sorted(blacklisted)))
        for seed in [42, 123, 7, 99]:
            result = select_from_tensors(
                scan,
                top_k=5,
                docs_per_feature=2,
                mode=EvalMode.ABLATION,
                seed=seed,
                feature_mask=mask,
            )
            for sel in result:
                assert sel.feature_id not in blacklisted, (
                    f"seed={seed}: blacklisted feature {sel.feature_id} appeared in selection"
                )

    def test_blacklist_returns_up_to_top_k(self) -> None:
        """With 7 allowed features (out of 10) and top_k=5, returns exactly 5."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(0, 1, 7))
        result = select_from_tensors(
            scan, top_k=5, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        assert len(result) == 5

    def test_blacklist_deterministic_for_fixed_seed(self) -> None:
        """Same seed ⇒ same selection order under blacklist."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 6, 8))
        r1 = select_from_tensors(
            scan, top_k=4, docs_per_feature=2, mode=EvalMode.ABLATION, seed=77,
            feature_mask=mask,
        )
        r2 = select_from_tensors(
            scan, top_k=4, docs_per_feature=2, mode=EvalMode.ABLATION, seed=77,
            feature_mask=mask,
        )
        assert [s.feature_id for s in r1] == [s.feature_id for s in r2]

    def test_blacklist_fids_match_frozen_golden(self) -> None:
        """Frozen golden for blacklist=(2,6,8), seed=42, top_k=5."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(2, 6, 8))
        result = select_from_tensors(
            scan, top_k=5, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        expected = [1, 9, 4, 7, 5]
        actual = [s.feature_id for s in result]
        assert actual == expected, (
            f"Blacklist golden drift: expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# 3. Whitelist restricts
# ---------------------------------------------------------------------------


class TestWhitelist:
    def test_only_whitelisted_features_selected(self) -> None:
        """Every returned feature_id must be in the whitelist."""
        scan = _make_scan_tensors(10, 3)
        allowed = {0, 1, 3, 5}
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(sorted(allowed)))
        for seed in [42, 123, 7]:
            result = select_from_tensors(
                scan, top_k=4, docs_per_feature=2, mode=EvalMode.ABLATION, seed=seed,
                feature_mask=mask,
            )
            for sel in result:
                assert sel.feature_id in allowed, (
                    f"seed={seed}: non-whitelisted feature {sel.feature_id} appeared"
                )

    def test_whitelist_top_k_capped_by_allowed_count(self) -> None:
        """top_k > |whitelist| ⇒ returns exactly |whitelist| features."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(2, 5, 7))
        result = select_from_tensors(
            scan, top_k=10, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        assert len(result) == 3
        returned_fids = {s.feature_id for s in result}
        assert returned_fids == {2, 5, 7}

    def test_whitelist_deterministic_for_fixed_seed(self) -> None:
        """Same seed ⇒ same selection order under whitelist."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0, 1, 3, 5))
        r1 = select_from_tensors(
            scan, top_k=3, docs_per_feature=2, mode=EvalMode.ABLATION, seed=55,
            feature_mask=mask,
        )
        r2 = select_from_tensors(
            scan, top_k=3, docs_per_feature=2, mode=EvalMode.ABLATION, seed=55,
            feature_mask=mask,
        )
        assert [s.feature_id for s in r1] == [s.feature_id for s in r2]

    def test_whitelist_fids_match_frozen_golden(self) -> None:
        """Frozen golden for whitelist=(0,1,3,5), seed=42, top_k=3."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(0, 1, 3, 5))
        result = select_from_tensors(
            scan, top_k=3, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        expected = [3, 5, 0]
        actual = [s.feature_id for s in result]
        assert actual == expected, (
            f"Whitelist golden drift: expected {expected}, got {actual}"
        )


# ---------------------------------------------------------------------------
# 4. All-masked ⇒ empty
# ---------------------------------------------------------------------------


class TestAllMasked:
    def test_blacklist_all_returns_empty(self) -> None:
        """Blacklisting every scanned feature returns [] (no crash, no RNG error)."""
        scan = _make_scan_tensors(10, 3)
        mask = FeatureMask(
            mode=FeatureMaskMode.blacklist, ids=tuple(range(10))
        )
        result = select_from_tensors(
            scan, top_k=5, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        assert result == []

    def test_whitelist_none_matching_returns_empty(self) -> None:
        """Whitelisting ids that are NOT in the scan ⇒ no candidates ⇒ []."""
        scan = _make_scan_tensors(10, 3)  # feature_ids = 0..9
        mask = FeatureMask(
            mode=FeatureMaskMode.whitelist, ids=(20, 30, 40)
        )
        result = select_from_tensors(
            scan, top_k=5, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        assert result == []

    def test_empty_scan_with_mask_returns_empty(self) -> None:
        """Empty ScanResultTensors + any mask ⇒ []."""
        empty = ScanResultTensors(
            feature_ids=t.empty(0, dtype=t.int64),
            n_docs_activated=t.empty(0, dtype=t.int64),
            offsets=t.zeros(1, dtype=t.int64),
            doc_idx=t.empty(0, dtype=t.int64),
            max_activation=t.empty(0, dtype=t.float32),
            max_segment_idx=t.empty(0, dtype=t.int64),
            percentile=t.empty(0, dtype=t.float32),
            total_segments=t.empty(0, dtype=t.int64),
        )
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(0, 1))
        result = select_from_tensors(
            empty, top_k=5, docs_per_feature=2, mode=EvalMode.ABLATION, seed=42,
            feature_mask=mask,
        )
        assert result == []


# ---------------------------------------------------------------------------
# 5. assert_selections_within_mask
# ---------------------------------------------------------------------------


class _FakeSel:
    """Minimal stand-in for RawFeatureSelection / FeatureSelection."""

    def __init__(self, feature_id: int) -> None:
        self.feature_id = feature_id


class TestAssertSelectionsWithinMask:
    def test_none_mask_is_noop(self) -> None:
        """feature_mask=None ⇒ no-op regardless of selection content (I5)."""
        sels = [_FakeSel(1), _FakeSel(3), _FakeSel(5)]
        # Must not raise even if the ids would be disallowed under some mask
        assert_selections_within_mask(sels, None)

    def test_empty_selections_is_noop(self) -> None:
        """Empty selections ⇒ no-op regardless of mask."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(1, 2, 3))
        assert_selections_within_mask([], mask)

    def test_all_allowed_no_raise(self) -> None:
        """Selections entirely within allowed set ⇒ no raise."""
        sels = [_FakeSel(1), _FakeSel(3), _FakeSel(5)]
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(10, 11, 12))
        assert_selections_within_mask(sels, mask)

    def test_raises_on_blacklist_violation(self) -> None:
        """A selection containing a blacklisted feature_id ⇒ FeatureMaskError."""
        sels = [_FakeSel(1), _FakeSel(3), _FakeSel(5)]
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(3,))
        with pytest.raises(FeatureMaskError):
            assert_selections_within_mask(sels, mask)

    def test_raises_on_whitelist_violation(self) -> None:
        """A selection containing a non-whitelisted feature_id ⇒ FeatureMaskError."""
        sels = [_FakeSel(1), _FakeSel(3), _FakeSel(5)]
        # Whitelist only 1 and 3 — 5 is not allowed
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 3))
        with pytest.raises(FeatureMaskError):
            assert_selections_within_mask(sels, mask)

    def test_accepts_real_raw_feature_selection(self) -> None:
        """Works with actual RawFeatureSelection objects."""
        sels = [
            RawFeatureSelection(feature_id=0, n_docs_activated=3, selected_docs=[0, 1, 2]),
            RawFeatureSelection(feature_id=4, n_docs_activated=2, selected_docs=[0, 1]),
        ]
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(10, 11))
        # Should not raise (0 and 4 are not blacklisted)
        assert_selections_within_mask(sels, mask)

    def test_accepts_real_feature_selection(self) -> None:
        """Works with actual FeatureSelection objects (has .feature_id)."""
        sels = [
            FeatureSelection(
                feature_id=2,
                neuronpedia_description="test",
                n_docs_activated=5,
                selected_docs=[0, 1],
            )
        ]
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(3, 4))
        assert_selections_within_mask(sels, mask)  # 2 is not blacklisted


# ---------------------------------------------------------------------------
# 6. Config round-trip (I5/back-compat)
# ---------------------------------------------------------------------------


class TestConfigRoundTrip:
    def test_completion_steering_eval_yaml_has_feature_mask_null(self) -> None:
        """completion_steering_eval.yaml must have feature_mask: null at top level."""
        yaml_path = _CONFIGS_EVAL_DIR / "completion_steering_eval.yaml"
        assert yaml_path.is_file(), f"YAML not found: {yaml_path}"
        raw = yaml.safe_load(yaml_path.read_text())
        assert isinstance(raw, dict)
        assert "feature_mask" in raw, (
            "completion_steering_eval.yaml is missing 'feature_mask' key — "
            "H8 gap: feature_mask is unreachable for this eval"
        )
        assert raw.get("feature_mask") is None, (
            f"completion_steering_eval.yaml feature_mask must be null, "
            f"got {raw.get('feature_mask')!r}"
        )

    def test_completion_steering_eval_gemma3_yaml_has_feature_mask_null(self) -> None:
        """completion_steering_eval_gemma3.yaml must have feature_mask: null at top level."""
        yaml_path = _CONFIGS_EVAL_DIR / "completion_steering_eval_gemma3.yaml"
        assert yaml_path.is_file(), f"YAML not found: {yaml_path}"
        raw = yaml.safe_load(yaml_path.read_text())
        assert isinstance(raw, dict)
        assert "feature_mask" in raw, (
            "completion_steering_eval_gemma3.yaml is missing 'feature_mask' key"
        )
        assert raw.get("feature_mask") is None, (
            f"completion_steering_eval_gemma3.yaml feature_mask must be null, "
            f"got {raw.get('feature_mask')!r}"
        )

    def test_omegaconf_round_trip_feature_mask_none(self) -> None:
        """OmegaConf round-trip: feature_mask resolves as None for both YAMLs."""
        for name in [
            "completion_steering_eval.yaml",
            "completion_steering_eval_gemma3.yaml",
        ]:
            yaml_path = _CONFIGS_EVAL_DIR / name
            raw = yaml.safe_load(yaml_path.read_text())
            cfg = OmegaConf.create(raw)
            resolved = OmegaConf.to_container(cfg, resolve=False)
            assert isinstance(resolved, dict)
            assert resolved.get("feature_mask") is None, (
                f"{name}: OmegaConf round-trip feature_mask must be null"
            )


# ---------------------------------------------------------------------------
# 7. Pipeline auto-resolve
# ---------------------------------------------------------------------------


class TestPipelineAutoResolve:
    """Verify build_feature_pipeline_from_cfg_block wires feature_mask via cfg slot."""

    _FS_BLOCK = {
        "threshold_type": "feature_percentile_segment_topk",
        "threshold": 2048.9,
    }

    def test_cfg_with_blacklist_mask_resolves_pipeline_feature_mask(self) -> None:
        """A cfg block with feature_mask={mode,ids} → pipeline.feature_mask set."""
        n = 16
        cfg = OmegaConf.create(
            {
                "feature_selection": self._FS_BLOCK,
                "negative_selection": None,
                "feature_mask": {"mode": "blacklist", "ids": [0, 1, 2]},
            }
        )
        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_make_per_feature_stats(n),
            n_sae_features=n,
            num_buckets=4,
        )
        assert pipeline.feature_mask is not None, (
            "pipeline.feature_mask must not be None when cfg has a blacklist"
        )
        assert pipeline.feature_mask.mode == FeatureMaskMode.blacklist
        assert set(pipeline.feature_mask.ids) == {0, 1, 2}

    def test_cfg_with_whitelist_mask_resolves_pipeline_feature_mask(self) -> None:
        """A cfg block with whitelist feature_mask → pipeline.feature_mask set."""
        n = 16
        cfg = OmegaConf.create(
            {
                "feature_selection": self._FS_BLOCK,
                "negative_selection": None,
                "feature_mask": {"mode": "whitelist", "ids": [5, 10]},
            }
        )
        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_make_per_feature_stats(n),
            n_sae_features=n,
            num_buckets=4,
        )
        assert pipeline.feature_mask is not None
        assert pipeline.feature_mask.mode == FeatureMaskMode.whitelist
        assert set(pipeline.feature_mask.ids) == {5, 10}

    def test_cfg_null_mask_resolves_to_none(self) -> None:
        """feature_mask: null in cfg → pipeline.feature_mask is None (I5 back-compat)."""
        n = 16
        cfg = OmegaConf.create(
            {
                "feature_selection": self._FS_BLOCK,
                "negative_selection": None,
                "feature_mask": None,
            }
        )
        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_make_per_feature_stats(n),
            n_sae_features=n,
            num_buckets=4,
        )
        assert pipeline.feature_mask is None, (
            "pipeline.feature_mask must be None when cfg.feature_mask is null (I5)"
        )

    def test_cfg_missing_mask_resolves_to_none(self) -> None:
        """No feature_mask key in cfg → pipeline.feature_mask is None (back-compat)."""
        n = 16
        cfg = OmegaConf.create(
            {
                "feature_selection": self._FS_BLOCK,
                "negative_selection": None,
            }
        )
        pipeline = build_feature_pipeline_from_cfg_block(
            cfg,
            per_feature_stats=_make_per_feature_stats(n),
            n_sae_features=n,
            num_buckets=4,
        )
        assert pipeline.feature_mask is None
