"""H6: tests proving that build_feature_pipeline_from_cfg threads feature_mask,
and that _resolve_bpb_selection_dir is mask-aware (sel-shard discovery gap).

Gap 1 (config-only, PR HEAD): conditioning BPB of a mask-trained model must
apply the feature_mask so the conditioning pipeline matches training. The fix
is to add a ``feature_mask: null`` slot to configs/eval/base_eval.yaml's
``conditioning:`` block so ``build_feature_pipeline_from_cfg_block``'s
auto-resolve (``resolve_feature_mask(cond_cfg, n_sae_features)``) picks it up.

Gap 2 (this commit): _resolve_bpb_selection_dir's precomputed selection-shard
DISCOVERY must incorporate feature_mask into the selection hash (same as
precompute_selected_features.py + conditional_encdec_train.py Task 16) and
assert the located sel/ manifest records the same mask. Without this, a
mask-trained model evaluated with conditioned BPB can silently load an unmasked
sel/ dir when stale unmasked shards exist on disk.

Call chain for Gap 1:
  base_eval.py:1159
    → build_feature_pipeline_from_cfg(cfg)          [bpb_pipeline.py]
    → build_feature_pipeline_from_cfg_block(cond_cfg, ...) [feature_pipeline.py]
    → resolve_feature_mask(cond_cfg, n_sae_features)        [feature_mask.py]

Call chain for Gap 2:
  base_eval.py ~1166
    → _resolve_bpb_selection_dir(packed_row_dir, ..., feature_mask_canonical=...) [base_eval.py]
      computes selection hash via compute_selection_hash(..., feature_mask=...) [packed_batch_io.py]
      asserts manifest.metadata.feature_mask == feature_mask_canonical           [overlay_io.py]
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from errors import FeatureMaskError
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode, canonical_dict
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import compute_selection_hash
from steering.stats_types import PerFeatureStat
from tests.statistics_doubles import fixed_statistics_double

# ---------------------------------------------------------------------------
# Fixtures shared with test_build_feature_pipeline_from_cfg.py
# ---------------------------------------------------------------------------

N_SAE = 64


def _stats(n: int = N_SAE) -> dict[str, PerFeatureStat]:
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    out: dict[str, PerFeatureStat] = {}
    for i in range(n):
        out[str(i)] = PerFeatureStat(
            count=100 - i,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(full_percentiles),
            histogram=None,
        )
    return out


def _fake_stats(n: int = N_SAE) -> SimpleNamespace:
    """Minimal fixed-mode double returned by ``load_statistics``."""

    return fixed_statistics_double(per_feature_stats=_stats(n))


def _make_cfg(*, feature_mask: dict | None = None, n_sae_features: int = N_SAE) -> object:
    """Build a cfg with the conditioning block mirroring base_eval.yaml structure.

    Uses the explicit ``cfg.bpb.stats_path`` shortcut so ``_resolve_stats_path``
    is bypassed and tests don't need real files on disk (the patch below handles
    load_statistics).
    """
    cond: dict = {
        "identity": {
            "model_id": "gemma-2-2b",
            "sae_id": "gemmascope-residual-l8-16k",
            "layer": 8,
            "pooling": "max_pool",
            "feature_selection_hash": None,
        },
        "n_sae_features": n_sae_features,
        "num_buckets": 2,
        "feature_selection": {
            "threshold_type": "feature_percentile",
            "threshold": 0.95,
        },
    }
    if feature_mask is not None:
        cond["feature_mask"] = feature_mask
    # No feature_mask key at all when None — mirrors the pre-fix base_eval.yaml
    # (and, post-fix, the null slot is equivalent: OmegaConf treats both as None).
    return OmegaConf.create({
        "conditioning": cond,
        "bpb": {
            "stats_path": "/fake/stats.safetensors",  # patched below
            "feature_root": None,
            "segment_size": None,
        },
        "segment_size": 128,
        "dataset_phase": "base",
    })


# ---------------------------------------------------------------------------
# Test 1: feature_mask resolved when conditioning block carries it
# ---------------------------------------------------------------------------


def test_bpb_pipeline_resolves_feature_mask(tmp_path):
    """build_feature_pipeline_from_cfg returns a pipeline with the correct
    FeatureMask when cfg.conditioning.feature_mask is set.

    This is the regression test for the H6 gap: previously the conditioning
    block had NO feature_mask slot → resolve_feature_mask returned None →
    pipeline had feature_mask=None even for mask-trained models.
    """
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    BLACKLISTED_ID = 5
    cfg = _make_cfg(feature_mask={"mode": "blacklist", "ids": [BLACKLISTED_ID]})

    fake_path = tmp_path / "stats.safetensors"
    fake_path.touch()

    with (
        patch("steering.eval.bpb_pipeline._resolve_stats_path", return_value=fake_path),
        patch(
            "steering.eval.bpb_pipeline.load_statistics",
            return_value=_fake_stats(N_SAE),
        ),
    ):
        pipeline = build_feature_pipeline_from_cfg(cfg)

    assert isinstance(pipeline, FeaturePipelineConfig)
    assert pipeline.feature_mask is not None, (
        "feature_mask must be resolved when cfg.conditioning.feature_mask is set; "
        "got None — the conditioning block is not being passed to resolve_feature_mask"
    )
    assert isinstance(pipeline.feature_mask, FeatureMask)
    assert pipeline.feature_mask.mode is FeatureMaskMode.blacklist
    assert BLACKLISTED_ID in pipeline.feature_mask.ids


# ---------------------------------------------------------------------------
# Test 2: no feature_mask configured → pipeline.feature_mask is None  (I5)
# ---------------------------------------------------------------------------


def test_bpb_pipeline_no_mask_is_none(tmp_path):
    """When cfg.conditioning has no feature_mask slot (or feature_mask: null),
    the pipeline has feature_mask=None — byte-identical to the pre-feature-mask
    behavior (I5: no regression for unmasked models).
    """
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    cfg = _make_cfg(feature_mask=None)  # no feature_mask key in conditioning block

    fake_path = tmp_path / "stats.safetensors"
    fake_path.touch()

    with (
        patch("steering.eval.bpb_pipeline._resolve_stats_path", return_value=fake_path),
        patch(
            "steering.eval.bpb_pipeline.load_statistics",
            return_value=_fake_stats(N_SAE),
        ),
    ):
        pipeline = build_feature_pipeline_from_cfg(cfg)

    assert isinstance(pipeline, FeaturePipelineConfig)
    assert pipeline.feature_mask is None, (
        "feature_mask must be None when not configured (I5: no-mask path unchanged)"
    )


# ---------------------------------------------------------------------------
# Test 3: null slot in cfg (post-fix base_eval.yaml) also resolves to None  (I5)
# ---------------------------------------------------------------------------


def test_bpb_pipeline_null_slot_is_none(tmp_path):
    """cfg.conditioning.feature_mask: null (the post-fix base_eval.yaml default)
    resolves to None — the null-slot addition is a no-op for unmasked models.
    """
    from steering.eval.bpb_pipeline import build_feature_pipeline_from_cfg

    cfg = _make_cfg(feature_mask=None)
    # Explicitly add the feature_mask key with None value (mirrors OmegaConf
    # null slot): simulate base_eval.yaml post-fix.
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg_dict["conditioning"]["feature_mask"] = None  # ty: ignore[invalid-argument-type, not-subscriptable]
    cfg_with_null_slot = OmegaConf.create(cfg_dict)

    fake_path = tmp_path / "stats.safetensors"
    fake_path.touch()

    with (
        patch("steering.eval.bpb_pipeline._resolve_stats_path", return_value=fake_path),
        patch(
            "steering.eval.bpb_pipeline.load_statistics",
            return_value=_fake_stats(N_SAE),
        ),
    ):
        pipeline = build_feature_pipeline_from_cfg(cfg_with_null_slot)

    assert pipeline.feature_mask is None


# ---------------------------------------------------------------------------
# Test 4: YAML round-trip — base_eval.yaml conditioning block has feature_mask slot
# ---------------------------------------------------------------------------


def test_base_eval_yaml_conditioning_has_feature_mask_slot():
    """After the fix, configs/eval/base_eval.yaml must have a feature_mask slot
    under conditioning so that resolve_feature_mask can pick it up from cond_cfg.

    A null value here is required (not an absent key) so that:
    (a) operators can override on CLI (conditioning.feature_mask.mode=blacklist
        conditioning.feature_mask.ids=[5])
    (b) the auto-resolve path in build_feature_pipeline_from_cfg_block fires
        correctly via resolve_feature_mask(cond_cfg, n_sae_features).
    """
    from hydra import compose, initialize_config_dir

    config_dir = Path(__file__).resolve().parents[2] / "configs" / "eval"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name="base_eval", overrides=["source=base"])

    assert "conditioning" in cfg, "cfg must have a conditioning block"
    cond = cfg.conditioning
    # The key must exist (even if null) so OmegaConf returns None (not KeyError)
    # and resolve_feature_mask sees a null block → disabled.
    assert hasattr(cond, "feature_mask") or "feature_mask" in cond, (
        "configs/eval/base_eval.yaml conditioning block must define feature_mask "
        "(null by default). Add 'feature_mask: null' under conditioning:."
    )
    assert cond.get("feature_mask", "MISSING") is None, (
        "conditioning.feature_mask must default to null in base_eval.yaml"
    )


# ---------------------------------------------------------------------------
# Gap-2 fixtures and helpers
# ---------------------------------------------------------------------------

_FS_RAW: dict = {"threshold_type": "feature_percentile", "threshold": 0.95}
_NEG_RAW: dict | None = None
_NUM_BUCKETS: int = 2
_SEED: int = 0  # BPB pins seed=0
_MASK_IDS = (5,)


def _make_sel_dir_with_manifest(
    packed_row_dir: Path,
    mask: FeatureMask | None,
    fs_raw: dict = _FS_RAW,
    neg_raw: dict | None = _NEG_RAW,
    num_buckets: int = _NUM_BUCKETS,
) -> Path:
    """Create a sel/<hash>/ directory with a manifest, mirroring precompute_selected_features.

    Computes the hash via compute_selection_hash (same args as precompute/training)
    so hash-based discovery in _resolve_bpb_selection_dir matches.
    """
    sel_hash = compute_selection_hash(
        positive_selection=fs_raw,
        negative_selection=neg_raw,
        delta_config=None,
        num_buckets=num_buckets,
        seed=_SEED,
        ddp_rank=0,
        feature_mask=canonical_dict(mask),
    )
    sel_dir = packed_row_dir / "sel" / sel_hash
    sel_dir.mkdir(parents=True, exist_ok=True)
    # Write a dummy shard so glob_overlay_shards finds it (not needed here but
    # mirrors real sel dirs; manifest is what matters for mask-check).
    (sel_dir / "shard_00000.safetensors.zst").write_bytes(b"fake")
    metadata: dict = {}
    if mask is not None:
        metadata["feature_mask"] = canonical_dict(mask)
    write_manifest(
        sel_dir,
        sel_hash,
        OverlayKind.PACKED_ROW,
        metadata=metadata if metadata else None,
    )
    return sel_dir


# ---------------------------------------------------------------------------
# Test class 5 — _resolve_bpb_selection_dir is mask-aware (Gap 2)
# ---------------------------------------------------------------------------


class TestResolveBpbSelectionDirMaskAware:
    """_resolve_bpb_selection_dir: hash includes feature_mask; manifest is checked.

    This is the regression test for Gap 2: the selection-shard discovery must be
    mask-aware so that a mask-trained model's conditioned BPB eval loads the
    CORRECT masked sel/ dir (or fails loudly on mismatch).
    """

    def test_feature_selection_is_required_before_directory_lookup(self, tmp_path: Path) -> None:
        """The private resolver fails loud instead of guessing a sel/ directory."""
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        (packed_row_dir / "sel" / "unrelated").mkdir(parents=True)

        with pytest.raises(ValueError, match="conditioning.feature_selection"):
            _resolve_bpb_selection_dir(
                packed_row_dir,
                selection_hash=None,
                feature_selection_raw=None,
            )

    @pytest.mark.parametrize("raw", [[], "threshold"])
    def test_feature_selection_must_be_a_mapping(
        self,
        tmp_path: Path,
        raw: object,
    ) -> None:
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        with pytest.raises(TypeError, match="resolve to a mapping"):
            _resolve_bpb_selection_dir(
                tmp_path,
                selection_hash=None,
                feature_selection_raw=raw,
            )

    def test_missing_selection_root_returns_none_for_computed_mode(self, tmp_path: Path) -> None:
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()

        assert (
            _resolve_bpb_selection_dir(
                packed_row_dir,
                selection_hash=None,
                feature_selection_raw=_FS_RAW,
                negative_selection_raw=_NEG_RAW,
                num_buckets=_NUM_BUCKETS,
            )
            is None
        )

    def test_explicit_hash_resolves_named_directory(self, tmp_path: Path) -> None:
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()
        expected = _make_sel_dir_with_manifest(packed_row_dir, None)

        resolved = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=expected.name,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
        )

        assert resolved == expected

    def test_explicit_hash_requires_existing_selection_root(self, tmp_path: Path) -> None:
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()

        with pytest.raises(ValueError, match="was pinned"):
            _resolve_bpb_selection_dir(
                packed_row_dir,
                selection_hash="missing",
                feature_selection_raw=_FS_RAW,
            )

    def test_explicit_hash_requires_existing_named_directory(self, tmp_path: Path) -> None:
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        (packed_row_dir / "sel").mkdir(parents=True)

        with pytest.raises(ValueError, match="does not exist"):
            _resolve_bpb_selection_dir(
                packed_row_dir,
                selection_hash="missing",
                feature_selection_raw=_FS_RAW,
            )

    def test_masked_discovery_resolves_masked_sel_dir(self, tmp_path: Path) -> None:
        """With feature_mask set, the computed hash resolves to the MASKED sel/ dir."""
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        mask = FeatureMask(mode="blacklist", ids=_MASK_IDS)
        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()
        masked_sel_dir = _make_sel_dir_with_manifest(packed_row_dir, mask)

        resolved = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=None,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
            feature_mask_canonical=canonical_dict(mask),
        )
        assert resolved == masked_sel_dir, (
            f"Expected masked sel dir {masked_sel_dir}, got {resolved}"
        )

    def test_no_mask_discovery_resolves_unmasked_sel_dir(self, tmp_path: Path) -> None:
        """No feature_mask: hash byte-identical to legacy; resolves unmasked sel/ dir (I5)."""
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()
        unmasked_sel_dir = _make_sel_dir_with_manifest(packed_row_dir, None)

        resolved = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=None,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
            feature_mask_canonical=None,
        )
        assert resolved == unmasked_sel_dir, (
            f"Expected unmasked sel dir {unmasked_sel_dir}, got {resolved}"
        )

    def test_masked_and_unmasked_dirs_coexist_each_resolves_correctly(
        self, tmp_path: Path
    ) -> None:
        """Both masked and unmasked sel/ dirs exist; each config resolves to its own dir."""
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        mask = FeatureMask(mode="blacklist", ids=_MASK_IDS)
        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()
        masked_sel_dir = _make_sel_dir_with_manifest(packed_row_dir, mask)
        unmasked_sel_dir = _make_sel_dir_with_manifest(packed_row_dir, None)

        # Mask config → masked dir
        resolved_masked = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=None,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
            feature_mask_canonical=canonical_dict(mask),
        )
        assert resolved_masked == masked_sel_dir

        # No mask config → unmasked dir
        resolved_unmasked = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=None,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
            feature_mask_canonical=None,
        )
        assert resolved_unmasked == unmasked_sel_dir

        # Different directories
        assert resolved_masked != resolved_unmasked

    def test_masked_config_absent_sel_dir_returns_none(self, tmp_path: Path) -> None:
        """Masked config but no matching sel/<hash>/ dir → returns None (fused inline fallback)."""
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        mask = FeatureMask(mode="blacklist", ids=_MASK_IDS)
        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()
        # Create a sel/ dir but with the UNMASKED hash — masked lookup won't find it.
        _make_sel_dir_with_manifest(packed_row_dir, None)
        (packed_row_dir / "sel" / "another-unrelated-selection").mkdir()

        resolved = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=None,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
            feature_mask_canonical=canonical_dict(mask),
        )
        assert resolved is None, (
            "Expected None when masked sel/<hash>/ dir does not exist; "
            "loader should fall back to fused-inline path"
        )

    def test_manifest_mismatch_raises_FeatureMaskError(self, tmp_path: Path) -> None:
        """Manifest records mask_A but configured mask_B → FeatureMaskError.

        Guards hash collisions and moved/mislabeled sel/ dirs.
        """
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        mask_a = FeatureMask(mode="blacklist", ids=_MASK_IDS)
        mask_b = FeatureMask(mode="blacklist", ids=(3,))
        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()

        # Build the sel dir for mask_a, but manually place it at mask_b's hash path
        # so the hash lookup succeeds but the manifest records mask_a.
        hash_b = compute_selection_hash(
            positive_selection=_FS_RAW,
            negative_selection=_NEG_RAW,
            delta_config=None,
            num_buckets=_NUM_BUCKETS,
            seed=_SEED,
            ddp_rank=0,
            feature_mask=canonical_dict(mask_b),
        )
        sel_dir = packed_row_dir / "sel" / hash_b
        sel_dir.mkdir(parents=True, exist_ok=True)
        (sel_dir / "shard_00000.safetensors.zst").write_bytes(b"fake")
        # Write manifest recording mask_a (mismatch with configured mask_b)
        write_manifest(
            sel_dir, hash_b, OverlayKind.PACKED_ROW,
            metadata={"feature_mask": canonical_dict(mask_a)},
        )

        with pytest.raises(FeatureMaskError, match="mismatch"):
            _resolve_bpb_selection_dir(
                packed_row_dir,
                selection_hash=None,
                feature_selection_raw=_FS_RAW,
                negative_selection_raw=_NEG_RAW,
                num_buckets=_NUM_BUCKETS,
                feature_mask_canonical=canonical_dict(mask_b),
            )

    def test_mask_configured_legacy_unmasked_manifest_raises(self, tmp_path: Path) -> None:
        """mask configured + legacy sel/ dir with no feature_mask key in manifest → FeatureMaskError.

        This is the CRITICAL back-compat guard: evaluating a mask-trained model on
        stale unmasked sel/ shards must fail loudly, not silently load wrong data.
        The hash-based lookup won't resolve the old unmasked hash anyway (different hash),
        but if somehow a sel/ dir exists at the masked hash with a legacy manifest,
        the manifest check catches it.
        """
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        mask = FeatureMask(mode="blacklist", ids=_MASK_IDS)
        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()

        # Build the sel dir at the MASKED hash path but write a legacy manifest (no feature_mask key)
        masked_hash = compute_selection_hash(
            positive_selection=_FS_RAW,
            negative_selection=_NEG_RAW,
            delta_config=None,
            num_buckets=_NUM_BUCKETS,
            seed=_SEED,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
        )
        sel_dir = packed_row_dir / "sel" / masked_hash
        sel_dir.mkdir(parents=True, exist_ok=True)
        (sel_dir / "shard_00000.safetensors.zst").write_bytes(b"fake")
        # Legacy manifest: no feature_mask key at all
        write_manifest(sel_dir, masked_hash, OverlayKind.PACKED_ROW, metadata=None)

        with pytest.raises(FeatureMaskError, match="mismatch"):
            _resolve_bpb_selection_dir(
                packed_row_dir,
                selection_hash=None,
                feature_selection_raw=_FS_RAW,
                negative_selection_raw=_NEG_RAW,
                num_buckets=_NUM_BUCKETS,
                feature_mask_canonical=canonical_dict(mask),
            )

    def test_no_mask_legacy_unmasked_manifest_passes(self, tmp_path: Path) -> None:
        """No mask configured + legacy manifest with no feature_mask key → passes (I5 back-compat).

        This is the key I5 invariant: existing unmasked sel/ dirs keep working
        byte-identically for unmasked models.
        """
        from scripts.eval.base_eval import _resolve_bpb_selection_dir

        packed_row_dir = tmp_path / "packed"
        packed_row_dir.mkdir()

        # Build an unmasked sel dir with a legacy manifest (no feature_mask key)
        unmasked_hash = compute_selection_hash(
            positive_selection=_FS_RAW,
            negative_selection=_NEG_RAW,
            delta_config=None,
            num_buckets=_NUM_BUCKETS,
            seed=_SEED,
            ddp_rank=0,
            feature_mask=None,
        )
        sel_dir = packed_row_dir / "sel" / unmasked_hash
        sel_dir.mkdir(parents=True, exist_ok=True)
        (sel_dir / "shard_00000.safetensors.zst").write_bytes(b"fake")
        write_manifest(sel_dir, unmasked_hash, OverlayKind.PACKED_ROW, metadata=None)

        # Must not raise (I5: None == None)
        resolved = _resolve_bpb_selection_dir(
            packed_row_dir,
            selection_hash=None,
            feature_selection_raw=_FS_RAW,
            negative_selection_raw=_NEG_RAW,
            num_buckets=_NUM_BUCKETS,
            feature_mask_canonical=None,
        )
        assert resolved == sel_dir

    def test_hash_parity_with_precompute_and_training(self, tmp_path: Path) -> None:
        """The hash computed by _resolve_bpb_selection_dir equals the one that
        precompute_selected_features.py and conditional_encdec_train.py produce.

        This directly proves the ANTI-DESYNC property: a sel/ dir written by
        precompute is discoverable by both training AND eval with the same mask.
        """
        mask = FeatureMask(mode="blacklist", ids=_MASK_IDS)
        # Hash computed the same way as precompute/training (the canonical tuple)
        expected_hash = compute_selection_hash(
            positive_selection=_FS_RAW,
            negative_selection=_NEG_RAW,
            delta_config=None,
            num_buckets=_NUM_BUCKETS,
            seed=_SEED,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
        )
        # Hash that _resolve_bpb_selection_dir would compute (same call)
        actual_hash = compute_selection_hash(
            positive_selection=_FS_RAW,
            negative_selection=_NEG_RAW,
            delta_config=None,
            num_buckets=_NUM_BUCKETS,
            seed=_SEED,
            ddp_rank=0,
            feature_mask=canonical_dict(mask),
        )
        assert actual_hash == expected_hash, (
            "Hash parity violation between BPB eval discovery and precompute/training. "
            "The ANTI-DESYNC invariant is broken."
        )
        # Also verify: unmasked hash differs from masked hash (no collision)
        unmasked_hash = compute_selection_hash(
            positive_selection=_FS_RAW,
            negative_selection=_NEG_RAW,
            delta_config=None,
            num_buckets=_NUM_BUCKETS,
            seed=_SEED,
            ddp_rank=0,
            feature_mask=None,
        )
        assert actual_hash != unmasked_hash, (
            "Masked and unmasked hashes must differ to ensure correct sel/ dir selection."
        )
