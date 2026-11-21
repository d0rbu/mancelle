"""Tests for defense-in-depth: sel/ manifest records feature_mask + load-time assertion.

Invariant I4: A precomputed sel/<hash>/ directory must record the canonical
feature_mask in its manifest.  At discovery time the configured mask must be
ASSERTED to match the manifest's recorded mask; mismatch ⇒ FeatureMaskError.

Back-compat (I5): existing sel/ dirs with no feature_mask key + no configured
mask must keep working byte-identically (legacy assertion: None == None → pass).

Fixture mirrors _build_base_shard / _build_pipeline from
tests/steering/test_precompute_selected_features_mask.py (Task 10).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

from errors import FeatureClustersError, FeatureMaskError
from scripts.preprocess import precompute_selected_features as precompute_module
from scripts.preprocess.precompute_selected_features import main as precompute_main
from scripts.preprocess.precompute_selected_features import select_features_for_shard
from scripts.train import conditional_encdec_train as train_module
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    build_clustered_selection_shard_integrity,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode, canonical_dict
from steering.conditional.feature_pipeline import (
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.selection_artifact_validation import (
    validate_complete_selection_dir,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.feature_io import save_features
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    read_manifest,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    compute_selection_hash,
    load_packed_row_shard,
    load_selection_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.stats_io import save_statistics
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    SegmentMode,
    StorageStats,
)

# ---------------------------------------------------------------------------
# Constants (mirrored from test_precompute_selected_features_mask.py)
# ---------------------------------------------------------------------------

N_SAE_FEATURES = 64
_K = 7
_N_SEGS = 6
_N_FEATS_PER_SEG = 8
_SEED = 1


# ---------------------------------------------------------------------------
# Fixture helpers (mirrored from test_precompute_selected_features_mask.py)
# ---------------------------------------------------------------------------


def _make_per_feature_stats(n: int = N_SAE_FEATURES) -> dict[str, PerFeatureStat]:
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


def _build_base_shard(tmp_path: Path) -> tuple[PackedRowShard, Path]:
    T = 16
    n_rows = 2
    segs_per_row = _N_SEGS // n_rows  # 3

    all_fids: list[int] = []
    all_pcts: list[int] = []
    feat_counts: list[int] = []

    for seg_idx in range(_N_SEGS):
        for f in range(_N_FEATS_PER_SEG):
            fid = (seg_idx * _N_FEATS_PER_SEG + f) % N_SAE_FEATURES
            all_fids.append(fid)
            all_pcts.append(200)
        feat_counts.append(_N_FEATS_PER_SEG)

    seg_feat_offsets = [0]
    for c in feat_counts:
        seg_feat_offsets.append(seg_feat_offsets[-1] + c)

    row_seg_offsets = [0, segs_per_row, _N_SEGS]
    tokens = t.ones(n_rows, T + 1, dtype=t.uint16)

    shard = PackedRowShard(
        tokens=tokens,
        row_seg_offsets=t.tensor(row_seg_offsets, dtype=t.int32),
        seg_token_start=t.zeros(_N_SEGS, dtype=t.uint16),
        seg_token_end=t.full((_N_SEGS,), T, dtype=t.uint16),
        seg_doc_id=t.zeros(_N_SEGS, dtype=t.uint8),
        seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
        feature_ids=t.tensor(all_fids, dtype=t.uint16),
        feature_pcts=t.tensor(all_pcts, dtype=t.uint8),
    )

    shard_dir = tmp_path / "base"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")
    return shard, shard_path


def _build_pipeline(mask: FeatureMask | None, seed: int = _SEED):
    cfg = OmegaConf.create(
        {
            "feature_selection": {
                "distribution": "geometric",
                "mean": 4.0,
                "std": 4.0,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 3.0,
                "std": 3.0,
            },
        }
    )
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_make_per_feature_stats(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=8,
        seed=seed,
        feature_mask=mask,
    )


def _make_sel_dir(base_shard_path: Path, mask: FeatureMask | None, tmp_path: Path) -> Path:
    """Run select_features_for_shard + write_manifest for a sel/ dir.

    Returns the sel/<hash>/ directory (created by this helper).
    """
    feature_selection_raw = {"distribution": "geometric", "mean": 4.0, "std": 4.0}
    negative_selection_raw = {"distribution": "geometric", "mean": 3.0, "std": 3.0}
    selection_hash = compute_selection_hash(
        positive_selection=feature_selection_raw,
        negative_selection=negative_selection_raw,
        delta_config=None,
        num_buckets=8,
        seed=_SEED,
        ddp_rank=0,
        feature_mask=canonical_dict(mask),
    )
    packed_row_dir = base_shard_path.parent
    sel_dir = packed_row_dir / "sel" / selection_hash
    sel_dir.mkdir(parents=True, exist_ok=True)

    pipeline = _build_pipeline(mask)
    select_features_for_shard(base_shard_path, pipeline, sel_dir, "shard_00000")

    # Write manifest (the behavior we are testing — mirrors precompute_packed_rows).
    metadata: dict = {
        "packed_rows": {
            "packed_row_fingerprint": packed_row_fingerprint(packed_row_dir),
        },
    }
    if mask is not None:
        metadata["feature_mask"] = canonical_dict(mask)
    write_manifest(
        sel_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata=metadata,
        base_shard_paths=[base_shard_path],
    )
    return sel_dir


# ---------------------------------------------------------------------------
# Helper: assertion logic extracted (mirrors the implementation under test)
# ---------------------------------------------------------------------------


def _assert_manifest_mask_matches(
    sel_dir: Path,
    configured_mask: FeatureMask | None,
) -> None:
    """Assert that the manifest's feature_mask matches the configured mask.

    Mirrors the logic that conditional_encdec_train.py must implement.
    Raises FeatureMaskError on mismatch; passes silently when equal.
    """
    manifest = read_manifest(sel_dir)
    manifest_mask = manifest.get("metadata", {}).get("feature_mask") if manifest else None
    configured_canonical = canonical_dict(configured_mask)
    if manifest_mask != configured_canonical:
        raise FeatureMaskError(
            f"sel/ manifest feature_mask mismatch: "
            f"manifest has {manifest_mask!r}, configured {configured_canonical!r}"
        )


# ---------------------------------------------------------------------------
# Test class 1 — precompute writes the correct manifest metadata
# ---------------------------------------------------------------------------


class TestSelManifestWritten:
    """After running precompute, the sel/ manifest records the right feature_mask."""

    def test_blacklist_mask_in_manifest_metadata(self, tmp_path: Path) -> None:
        """After precomputing with a blacklist mask, manifest.metadata.feature_mask
        equals canonical_dict(mask)."""
        _, base_shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        sel_dir = _make_sel_dir(base_shard_path, mask, tmp_path)

        manifest = read_manifest(sel_dir)
        assert manifest is not None, "Manifest must be written"
        assert "metadata" in manifest, "Manifest must have metadata"
        assert manifest["metadata"].get("feature_mask") == canonical_dict(mask), (
            f"Manifest feature_mask {manifest['metadata'].get('feature_mask')!r} "
            f"!= canonical_dict(mask) {canonical_dict(mask)!r}"
        )

    def test_whitelist_mask_in_manifest_metadata(self, tmp_path: Path) -> None:
        """After precomputing with a whitelist mask, manifest.metadata.feature_mask
        equals canonical_dict(mask)."""
        _, base_shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(3, 5, _K, 10))
        sel_dir = _make_sel_dir(base_shard_path, mask, tmp_path)

        manifest = read_manifest(sel_dir)
        assert manifest is not None
        assert manifest.get("metadata", {}).get("feature_mask") == canonical_dict(mask)

    def test_no_mask_manifest_has_no_feature_mask_key(self, tmp_path: Path) -> None:
        """When no mask is configured, the manifest must NOT have a feature_mask key
        (back-compat: I5 — omit the key entirely, do NOT write feature_mask: null)."""
        _, base_shard_path = _build_base_shard(tmp_path)
        sel_dir = _make_sel_dir(base_shard_path, None, tmp_path)

        manifest = read_manifest(sel_dir)
        assert manifest is not None, "Manifest must be written even without a mask"
        # The key must be absent (not present as None)
        meta = manifest.get("metadata", {})
        assert "feature_mask" not in meta, (
            f"Manifest must NOT contain feature_mask key when mask=None, got metadata={meta!r}"
        )

    def test_manifest_has_checksum_and_kind(self, tmp_path: Path) -> None:
        """Basic schema sanity: manifest must have checksum and kind fields."""
        _, base_shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        sel_dir = _make_sel_dir(base_shard_path, mask, tmp_path)

        manifest = read_manifest(sel_dir)
        assert manifest is not None
        assert "checksum" in manifest
        assert "kind" in manifest


# ---------------------------------------------------------------------------
# Test class 2 — load-time assertion logic
# ---------------------------------------------------------------------------


class TestLoadTimeAssertion:
    """The discovery-time assert must catch mask mismatches and pass on match."""

    def test_matching_mask_passes(self, tmp_path: Path) -> None:
        """When configured mask == manifest mask, assertion passes (no error)."""
        _, base_shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        sel_dir = _make_sel_dir(base_shard_path, mask, tmp_path)

        # Must not raise
        _assert_manifest_mask_matches(sel_dir, mask)

    def test_mismatched_mask_raises_FeatureMaskError(self, tmp_path: Path) -> None:
        """When configured mask != manifest mask, FeatureMaskError is raised."""
        _, base_shard_path = _build_base_shard(tmp_path)
        mask_a = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        mask_b = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(3,))
        sel_dir = _make_sel_dir(base_shard_path, mask_a, tmp_path)

        with pytest.raises(FeatureMaskError):
            _assert_manifest_mask_matches(sel_dir, mask_b)

    def test_both_none_passes(self, tmp_path: Path) -> None:
        """Legacy path: manifest has no feature_mask key, configured mask is None
        → assertion passes (None == None — back-compat I5)."""
        _, base_shard_path = _build_base_shard(tmp_path)
        sel_dir = _make_sel_dir(base_shard_path, None, tmp_path)

        # Must not raise
        _assert_manifest_mask_matches(sel_dir, None)

    def test_manifest_mask_set_configured_none_raises(self, tmp_path: Path) -> None:
        """Manifest has a mask recorded but configured mask is None → mismatch ⇒ error.

        This guards the case where someone copies a masked sel/ dir to an unmasked
        run's packed_row_dir.
        """
        _, base_shard_path = _build_base_shard(tmp_path)
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        sel_dir = _make_sel_dir(base_shard_path, mask, tmp_path)

        with pytest.raises(FeatureMaskError):
            _assert_manifest_mask_matches(sel_dir, None)

    def test_manifest_mask_none_configured_set_raises(self, tmp_path: Path) -> None:
        """Legacy manifest (no feature_mask key) but configured mask is set → mismatch ⇒ error."""
        _, base_shard_path = _build_base_shard(tmp_path)
        # Build a sel/ dir WITHOUT a mask
        sel_dir = _make_sel_dir(base_shard_path, None, tmp_path)

        # Now assert with a configured mask — must raise
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(_K,))
        with pytest.raises(FeatureMaskError):
            _assert_manifest_mask_matches(sel_dir, mask)

    def test_legacy_manifest_no_key_no_mask_passes(self, tmp_path: Path) -> None:
        """A manually written legacy-style manifest (no metadata or feature_mask key at all)
        combined with no configured mask → passes (back-compat I5)."""
        _, base_shard_path = _build_base_shard(tmp_path)
        # Write a legacy manifest (no metadata block)
        legacy_sel_dir = base_shard_path.parent / "sel" / "legacyhash"
        legacy_sel_dir.mkdir(parents=True, exist_ok=True)
        # Write a shard file so glob_overlay_shards finds it
        pipeline = _build_pipeline(None)
        select_features_for_shard(base_shard_path, pipeline, legacy_sel_dir, "shard_00000")
        # Write a manifest without feature_mask (legacy style — no metadata key at all)
        write_manifest(
            legacy_sel_dir,
            "legacyhash",
            OverlayKind.PACKED_ROW,
            metadata=None,  # no metadata at all
        )

        # Back-compat: no mask, legacy manifest → must not raise
        _assert_manifest_mask_matches(legacy_sel_dir, None)


# ---------------------------------------------------------------------------
# Production-path infrastructure helpers (for TestSelManifestProductionPath)
# ---------------------------------------------------------------------------

_PROD_N_SAE = 64
_PROD_SEGMENT_SIZE = 128


def _make_production_extraction_stats(n_sae: int = _PROD_N_SAE) -> ExtractionStatistics:
    """Build a minimal ExtractionStatistics that satisfies save_statistics + load_statistics.

    segment_size=128 must match the s128 path component created by
    _make_production_feature_dir so verify_segment_size in main.__wrapped__ passes.
    """
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    per_feature_stats = {
        str(i): PerFeatureStat(
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
        for i in range(n_sae)
    }
    return ExtractionStatistics(
        metadata=MetadataStats(
            pooling_mode="max_pool",
            segment_mode=SegmentMode.fixed_tokens,
            segment_size=_PROD_SEGMENT_SIZE,
            context_window=1024,
            layer=8,
            n_sae_features=n_sae,
            generated_at="2025-01-01T00:00:00",
            activation_percentile=0.95,
        ),
        corpus=CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=10000,
            mean_tokens_per_doc=100.0,
            median_tokens_per_doc=100,
        ),
        features=SegmentFeatureStats(
            n_segments=100,
            n_features_seen=n_sae,
            mean_active_per_segment=2.0,
            median_active_per_segment=2,
            p90_active_per_segment=4,
            max_active_per_segment=8,
            min_active_per_segment=1,
            n_total_sae_features=n_sae,
            pct_features_active=1.0,
            count_percentiles={"p50": 50},
            magnitude_stats={},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        ),
        storage=StorageStats(
            total_bytes=1000,
            mean_bytes_per_doc=10.0,
            mean_bytes_per_token=0.1,
        ),
        per_feature_stats=per_feature_stats,
        extrapolation_10b=Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=1_000_000,
            estimated_storage_gb=1.0,
            estimated_wall_time_hours=1.0,
        ),
    )


def _make_production_feature_dir(root: Path) -> Path:
    """Create a feature_dir with an s128 path component, stats file, and a fake feature shard.

    The s128 component satisfies verify_segment_size's _parse_segment_size_from_path check.
    The fake feature shard (with segment_size=128 in its metadata) satisfies
    _read_safetensors_metadata in verify_segment_size.
    The stats file satisfies find_statistics_path + load_statistics in main.__wrapped__.
    """
    feature_dir = root / "s128" / "features"
    feature_dir.mkdir(parents=True)
    stats = _make_production_extraction_stats(_PROD_N_SAE)
    save_statistics(stats, feature_dir / "feature_statistics")
    # Create a minimal feature shard so glob_features finds it and verify_segment_size passes.
    shard_data = {
        "offsets": t.tensor([0, 4], dtype=t.int32),
        "feature_ids": t.tensor([0, 1, 2, 3], dtype=t.int64),
        "feature_mags": t.ones(4, dtype=t.float32),
        "segment_size": _PROD_SEGMENT_SIZE,
        "n_sae_features": _PROD_N_SAE,
    }
    save_features(shard_data, feature_dir / "shard_00000")
    return feature_dir


def _make_production_packed_row_dir(root: Path) -> Path:
    """Create a packed_row_dir with one base PackedRowShard.

    Returns the packed_row_dir path (glob_overlay_shards must find the shard there).
    """
    packed_row_dir = root / "packed_rows"
    packed_row_dir.mkdir(parents=True)
    n_rows, segs_per_row, n_feats_per_seg, T = 2, 3, 8, 16
    n_segs = n_rows * segs_per_row
    all_fids = [
        (seg_idx * n_feats_per_seg + f) % _PROD_N_SAE
        for seg_idx in range(n_segs)
        for f in range(n_feats_per_seg)
    ]
    seg_feat_offsets = list(range(0, n_segs * n_feats_per_seg + 1, n_feats_per_seg))
    shard = PackedRowShard(
        tokens=t.ones(n_rows, T + 1, dtype=t.uint16),
        row_seg_offsets=t.tensor([0, segs_per_row, n_segs], dtype=t.int32),
        seg_token_start=t.zeros(n_segs, dtype=t.uint16),
        seg_token_end=t.full((n_segs,), T, dtype=t.uint16),
        seg_doc_id=t.zeros(n_segs, dtype=t.uint8),
        seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
        feature_ids=t.tensor(all_fids, dtype=t.uint16),
        feature_pcts=t.full((len(all_fids),), 200, dtype=t.uint8),
    )
    save_packed_row_shard(shard, packed_row_dir / "shard_00000")
    return packed_row_dir


def _run_production_main(
    tmp_root: Path,
    mask: FeatureMask | None,
) -> Path:
    """Call main.__wrapped__ (the real production entrypoint, lines 253–404 in
    precompute_selected_features.py, including the manifest write at lines 386–402)
    and return the sel/<hash>/ directory it created.

    Entrypoint: scripts/preprocess/precompute_selected_features.main (the unwrapped
    production function).  Lines 386–402 are the ``write_manifest`` block at the
    end of main that records ``feature_mask`` in the manifest.  Calling
    main.__wrapped__ is the ONLY way to exercise those lines without modifying
    production code (they are in main, not in a separate helper).

    Why main.__wrapped__ genuinely runs lines 386–402:
      - main.__wrapped__ is the original Python function (the hydra decorator
        sets .__wrapped__ to the undecorated version).
      - Lines 386–402 execute unconditionally after the shard loop — there is
        no early-exit branch between the loop and the manifest write in the
        production code.
      - If the ``if feature_mask is not None:`` guard at line 394 were flipped
        (i.e., changed to ``if feature_mask is None:``), the no-mask test below
        would spuriously find a ``feature_mask`` key in the manifest, failing
        the assertion ``'feature_mask' not in meta``.  That regression would
        be caught by test_no_mask_manifest_omits_feature_mask_key.
    """
    feature_dir = _make_production_feature_dir(tmp_root / "feat")
    packed_row_dir = _make_production_packed_row_dir(tmp_root / "rows")
    cfg_dict: dict = {
        "packed_row_dir": str(packed_row_dir),
        "feature_dir": str(feature_dir),
        "segment_size": _PROD_SEGMENT_SIZE,
        "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
        "negative_selection": {"distribution": "geometric", "mean": 3.0, "std": 3.0},
        "feature_mask_freq_min": None,
        "feature_mask_freq_max": None,
        "num_buckets": 8,
        "seed": 1,
    }
    if mask is not None:
        cfg_dict["feature_mask"] = {"mode": mask.mode, "ids": list(mask.ids)}
    cfg = OmegaConf.create(cfg_dict)
    precompute_main.__wrapped__(cfg)
    # Return the sel/<hash>/ directory created by main
    sel_dirs = list((packed_row_dir / "sel").iterdir())
    assert len(sel_dirs) == 1, f"Expected exactly one sel/ dir, got: {sel_dirs}"
    return sel_dirs[0]


# ---------------------------------------------------------------------------
# Test class 3 — production-path manifest write (TestSelManifestProductionPath)
#
# These tests call main.__wrapped__ (the REAL production entrypoint) so that
# lines 386–402 of precompute_selected_features.py actually execute.  A bug
# in the manifest write block (e.g. accidentally writing feature_mask: null
# for the no-mask case) will be caught here but NOT by TestSelManifestWritten
# (which uses the re-implemented _make_sel_dir helper).
# ---------------------------------------------------------------------------


class TestSelManifestProductionPath:
    """Pin the REAL production manifest write at lines 386–402 of
    scripts/preprocess/precompute_selected_features.py:main.

    Three cases: blacklist, whitelist, no mask.  Each calls main.__wrapped__
    and reads the on-disk manifest via the real read_manifest (no mocking).
    """

    def test_blacklist_manifest_via_production_main(self, tmp_path: Path) -> None:
        """Blacklist mask → main writes manifest with feature_mask == canonical_dict(mask)."""
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(7,))
        sel_dir = _run_production_main(tmp_path, mask)
        manifest = read_manifest(sel_dir)
        assert manifest is not None, "Production main must write a manifest"
        meta = manifest.get("metadata") or {}
        assert "feature_mask" in meta, (
            f"Production manifest missing feature_mask key for blacklist mask; meta={meta!r}"
        )
        assert meta["feature_mask"] == canonical_dict(mask), (
            f"Production manifest feature_mask {meta['feature_mask']!r} "
            f"!= canonical_dict(mask) {canonical_dict(mask)!r}"
        )

    def test_whitelist_manifest_via_production_main(self, tmp_path: Path) -> None:
        """Whitelist mask → main writes manifest with feature_mask == canonical_dict(mask)."""
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=(1, 5, 10, 20))
        sel_dir = _run_production_main(tmp_path, mask)
        manifest = read_manifest(sel_dir)
        assert manifest is not None
        meta = manifest.get("metadata") or {}
        assert "feature_mask" in meta, (
            f"Production manifest missing feature_mask key for whitelist mask; meta={meta!r}"
        )
        assert meta["feature_mask"] == canonical_dict(mask), (
            f"Production manifest feature_mask {meta['feature_mask']!r} "
            f"!= canonical_dict(mask) {canonical_dict(mask)!r}"
        )

    def test_no_mask_manifest_omits_feature_mask_key(self, tmp_path: Path) -> None:
        """No mask → production main OMITS feature_mask key entirely (I5 back-compat).

        The production guard is:
            sel_metadata: dict = {}
            if feature_mask is not None:            # line 394
                sel_metadata['feature_mask'] = ...
            write_manifest(..., metadata=sel_metadata if sel_metadata else None, ...)

        When feature_mask is None, sel_metadata stays empty, so metadata=None is
        passed to write_manifest — the key is ABSENT, not written as null.
        This assertion would fail if the guard were accidentally inverted or if
        ``feature_mask: null`` were written instead of omitting the key.
        """
        sel_dir = _run_production_main(tmp_path, mask=None)
        manifest = read_manifest(sel_dir)
        assert manifest is not None, "Production main must write a manifest even without a mask"
        meta = manifest.get("metadata", {}) or {}
        assert "feature_mask" not in meta, (
            f"I5 violation: production manifest must NOT contain feature_mask key "
            f"when mask=None (would break legacy maskless sel/ dirs); "
            f"got metadata={meta!r}"
        )


class TestFeatureClustersProductionManifestPath:
    """Exercise clustered artifacts through the real Hydra-unwrapped main."""

    @staticmethod
    def _cfg_and_state(tmp_path: Path):
        feature_dir = _make_production_feature_dir(tmp_path / "feat")
        packed_row_dir = _make_production_packed_row_dir(tmp_path / "rows")
        cluster_path = tmp_path / "clusters.json"
        cluster_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "clusters": [[1, 3, 5], [8, 9]],
                }
            ),
            encoding="utf-8",
        )
        cfg = OmegaConf.create(
            {
                "packed_row_dir": str(packed_row_dir),
                "feature_dir": str(feature_dir),
                "segment_size": _PROD_SEGMENT_SIZE,
                "feature_selection": {
                    "threshold_type": "feature_percentile",
                    "threshold": 0.5,
                },
                "negative_selection": None,
                "feature_delta": None,
                "feature_mask_freq_min": None,
                "feature_mask_freq_max": None,
                "feature_mask": {"mode": "blacklist", "ids": [1]},
                "feature_clusters": {"path": str(cluster_path)},
                "num_buckets": 8,
                "seed": 1,
            }
        )
        resolved = resolve_feature_clusters(cfg, _PROD_N_SAE)
        assert resolved is not None
        state = FeatureClusterConsolidation.build(
            resolved,
            FeatureMask(FeatureMaskMode.blacklist, (1,)),
        )
        return cfg, packed_row_dir, state

    def test_main_writes_in_progress_then_validated_complete_manifest(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        cfg, packed_row_dir, state = self._cfg_and_state(tmp_path)
        writes: list[dict] = []
        real_write_manifest = precompute_module.write_manifest

        def _recording_write_manifest(*args, **kwargs):
            metadata = kwargs.get("metadata")
            assert isinstance(metadata, dict)
            writes.append(copy.deepcopy(metadata))
            return real_write_manifest(*args, **kwargs)

        monkeypatch.setattr(
            precompute_module,
            "write_manifest",
            _recording_write_manifest,
        )

        precompute_module.main.__wrapped__(cfg)

        assert [entry["feature_clusters"]["status"] for entry in writes] == [
            "in_progress",
            "in_progress",
            "complete",
        ]
        assert SELECTION_SHARD_INTEGRITY_KEY not in writes[0]
        assert SELECTION_SHARD_INTEGRITY_KEY in writes[1]
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest = read_manifest(sel_dir)
        assert manifest is not None
        metadata = manifest["metadata"]
        expected_cluster_metadata = state.metadata_dict() | {"status": "complete"}
        assert metadata["feature_clusters"] == expected_cluster_metadata

        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        selection = load_selection_shard(selection_path)
        state.assert_representatives_only(
            selection.sel_fids.to(t.int64),
            surface="production positive selection",
        )
        state.assert_representatives_only(
            selection.neg_fids.to(t.int64),
            surface="production negative selection",
        )
        assert 1 not in selection.sel_fids.tolist()
        assert 5 not in selection.sel_fids.tolist()
        assert 3 in selection.sel_fids.tolist()

        original_tensors = {
            key: value.clone()
            for key, value in selection.__dict__.items()
            if isinstance(value, t.Tensor)
        }
        writes.clear()
        precompute_module.main.__wrapped__(cfg)
        resumed = load_selection_shard(selection_path)
        assert [entry["feature_clusters"]["status"] for entry in writes] == [
            "in_progress",
            "complete",
        ]
        for key, expected in original_tensors.items():
            assert t.equal(getattr(resumed, key), expected), key

    def test_resume_rejects_invalid_existing_rows_before_manifest_rewrite(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        manifest_before = manifest_path.read_bytes()
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        selection = load_selection_shard(selection_path)
        selection.sel_fids[0] = 5
        save_selection_shard(selection, selection_path)

        with pytest.raises(
            FeatureClustersError,
            match="content fingerprint",
        ):
            precompute_module.main.__wrapped__(cfg)

        assert manifest_path.read_bytes() == manifest_before

    def test_resume_rejects_structurally_valid_content_drift_before_manifest_rewrite(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        manifest_before = manifest_path.read_bytes()
        selection = load_selection_shard(selection_path)
        selection.sel_buckets[0] = (int(selection.sel_buckets[0].item()) + 1) % 7
        save_selection_shard(selection, selection_path)
        mutated_bytes = selection_path.read_bytes()

        with pytest.raises(FeatureClustersError, match="content fingerprint"):
            precompute_module.main.__wrapped__(cfg)

        assert manifest_path.read_bytes() == manifest_before
        assert selection_path.read_bytes() == mutated_bytes

    def test_in_progress_resume_recomputes_unverified_selection_shards(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        expected_bytes = selection_path.read_bytes()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["feature_clusters"]["status"] = "in_progress"
        manifest["metadata"].pop("selection_shards", None)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        selection = load_selection_shard(selection_path)
        selection.sel_buckets[0] = (int(selection.sel_buckets[0].item()) + 1) % 7
        save_selection_shard(selection, selection_path)
        assert selection_path.read_bytes() != expected_bytes

        precompute_module.main.__wrapped__(cfg)

        assert selection_path.read_bytes() == expected_bytes
        completed = read_manifest(sel_dir)
        assert completed is not None
        assert completed["metadata"]["feature_clusters"]["status"] == "complete"
        assert "selection_shards" in completed["metadata"]

    def test_in_progress_resume_reuses_byte_certified_shard_by_default(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        expected_bytes = selection_path.read_bytes()
        expected_inode = selection_path.stat().st_ino

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["feature_clusters"]["status"] = "in_progress"
        manifest["metadata"][SELECTION_SHARD_INTEGRITY_KEY] = (
            build_clustered_selection_shard_integrity([selection_path])
        )
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        def forbidden_recompute(*_args, **_kwargs):
            raise AssertionError("a byte-certified shard must be resumed, not recomputed")

        monkeypatch.setattr(
            precompute_module,
            "select_features_for_shard",
            forbidden_recompute,
        )
        precompute_module.main.__wrapped__(cfg)

        assert selection_path.read_bytes() == expected_bytes
        assert selection_path.stat().st_ino == expected_inode
        completed = read_manifest(sel_dir)
        assert completed is not None
        assert completed["metadata"]["feature_clusters"]["status"] == "complete"

    def test_selection_resume_false_deliberately_recomputes_certified_shard(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        expected_bytes = selection_path.read_bytes()
        real_select = precompute_module.select_features_for_shard
        calls: list[str] = []

        def recording_select(*args, **kwargs):
            calls.append(args[3])
            return real_select(*args, **kwargs)

        cfg.selection_resume = False
        monkeypatch.setattr(
            precompute_module,
            "select_features_for_shard",
            recording_select,
        )
        precompute_module.main.__wrapped__(cfg)

        assert calls == ["shard_00000"]
        assert selection_path.read_bytes() == expected_bytes

    def test_selection_resume_false_repairs_corrupt_certified_shard(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit recompute must not be blocked by bytes it will replace."""
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        expected_bytes = selection_path.read_bytes()
        selection_path.write_bytes(b"corrupt-certified-selection-shard")

        real_select = precompute_module.select_features_for_shard
        calls: list[str] = []

        def recording_select(*args, **kwargs):
            calls.append(args[3])
            return real_select(*args, **kwargs)

        cfg.selection_resume = False
        monkeypatch.setattr(
            precompute_module,
            "select_features_for_shard",
            recording_select,
        )
        precompute_module.main.__wrapped__(cfg)

        assert calls == ["shard_00000"]
        assert selection_path.read_bytes() == expected_bytes
        completed = read_manifest(sel_dir)
        assert completed is not None
        assert completed["metadata"]["feature_clusters"]["status"] == "complete"

    def test_selection_resume_true_rejects_same_corrupt_certified_shard(
        self,
        tmp_path: Path,
    ) -> None:
        """The repair escape hatch must not weaken default resume integrity."""
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        manifest_before = manifest_path.read_bytes()
        selection_path.write_bytes(b"corrupt-certified-selection-shard")

        with pytest.raises(FeatureClustersError, match="content fingerprint"):
            precompute_module.main.__wrapped__(cfg)

        assert manifest_path.read_bytes() == manifest_before
        assert selection_path.read_bytes() == b"corrupt-certified-selection-shard"

    def test_resume_rejects_changed_packed_rows_before_recertifying_stale_shards(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        manifest_before = manifest_path.read_bytes()
        selection_before = selection_path.read_bytes()

        packed_path = next(iter(glob_overlay_shards(packed_row_dir)))
        packed = load_packed_row_shard(packed_path)
        packed.tokens[0, 0] = int(packed.tokens[0, 0].item()) + 1
        save_packed_row_shard(packed, packed_path)

        with pytest.raises(FeatureClustersError, match="packed-row fingerprint"):
            precompute_module.main.__wrapped__(cfg)

        # The failure precedes both the in-progress manifest rewrite and the
        # existing-shard skip, so stale output cannot be silently re-certified.
        assert manifest_path.read_bytes() == manifest_before
        assert selection_path.read_bytes() == selection_before

    def test_resume_rejects_wrong_selection_checksum_before_manifest_rewrite(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, _state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        manifest_path = sel_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checksum"] = "f" * 64
        corrupted = json.dumps(manifest, indent=2).encode() + b"\n"
        manifest_path.write_bytes(corrupted)

        with pytest.raises(FeatureClustersError, match="selection checksum"):
            precompute_module.main.__wrapped__(cfg)

        assert manifest_path.read_bytes() == corrupted

    def test_training_rejects_renamed_or_copied_selection_identity(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        expected_hash = sel_dir.name
        manifest_path = sel_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["checksum"] = "e" * 64
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match="manifest selection checksum"):
            validate_complete_selection_dir(
                packed_row_dir=packed_row_dir,
                selection_hash=expected_hash,
                configured_mask=canonical_dict(state.feature_mask),
                label="cluster identity regression",
                feature_clusters=state,
                delta_config=None,
            )

    def test_training_rejects_structurally_valid_selection_content_drift(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        sel_dir = next((packed_row_dir / "sel").iterdir())
        selection_path = next(iter(glob_overlay_shards(sel_dir)))
        selection = load_selection_shard(selection_path)
        selection.sel_pcts[0] = (int(selection.sel_pcts[0].item()) + 1) % 256
        save_selection_shard(selection, selection_path)

        with pytest.raises(FeatureClustersError, match="content fingerprint"):
            validate_complete_selection_dir(
                packed_row_dir=packed_row_dir,
                selection_hash=sel_dir.name,
                configured_mask=canonical_dict(state.feature_mask),
                label="cluster content regression",
                feature_clusters=state,
                delta_config=None,
            )

    def test_clustered_original_validation_resolves_and_certifies_selection_dir(
        self,
        tmp_path: Path,
    ) -> None:
        cfg, packed_row_dir, state = self._cfg_and_state(tmp_path)
        precompute_module.main.__wrapped__(cfg)
        expected = next((packed_row_dir / "sel").iterdir())

        actual = train_module._ensure_clustered_original_val_selection_dir(
            current=None,
            original_val_packed_row_dir=packed_row_dir,
            selection_hash=expected.name,
            configured_mask=canonical_dict(state.feature_mask),
            feature_clusters=state,
            splice_counterfactual=False,
        )

        assert actual == expected
        specs = train_module._build_validation_eval_specs(
            packed_row_dir=tmp_path / "active" / "train",
            active_val_packed_row_dir=tmp_path / "active" / "val",
            active_val_selection_dir=tmp_path / "active" / "val" / "sel" / expected.name,
            original_val_packed_row_dir=packed_row_dir,
            original_val_selection_dir=actual,
        )
        assert specs[0].metric_prefix == "val"
        assert specs[0].selection_dir == expected


# ---------------------------------------------------------------------------
# Task 4 — hash cache lives in packed_row_dir (NOT feature_dir)
# ---------------------------------------------------------------------------


def test_selected_features_caches_in_packed_row_dir_not_feature_dir(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    import steering.overlay_io as oio
    from scripts.preprocess.precompute_selected_features import main as precompute_main
    from steering.overlay_io import (
        BASE_SHARD_HASH_CACHE_FILENAME,
        compute_base_shard_content_hash,
        glob_overlay_shards,
        read_manifest,
    )

    feature_dir = _make_production_feature_dir(tmp_path / "feat")
    packed_row_dir = _make_production_packed_row_dir(tmp_path / "rows")
    cfg = OmegaConf.create(
        {
            "packed_row_dir": str(packed_row_dir),
            "feature_dir": str(feature_dir),
            "segment_size": _PROD_SEGMENT_SIZE,
            "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
            "negative_selection": {"distribution": "geometric", "mean": 3.0, "std": 3.0},
            "num_buckets": 8,
            "seed": 1,
        }
    )

    precompute_main.__wrapped__(cfg)
    assert (packed_row_dir / BASE_SHARD_HASH_CACHE_FILENAME).is_file(), (
        "cache must live next to the hashed packed-row shards"
    )
    assert not (feature_dir / BASE_SHARD_HASH_CACHE_FILENAME).exists(), (
        "selected_features must NOT cache against feature_dir (different name-set)"
    )

    sel_dir = next((packed_row_dir / "sel").iterdir())
    manifest = read_manifest(sel_dir)
    assert manifest is not None
    assert manifest["base_shard_content_hash"] == compute_base_shard_content_hash(
        glob_overlay_shards(packed_row_dir),
        cache_dir=None,
    )

    calls = {"n": 0}
    orig = oio._stream_base_hash
    monkeypatch.setattr(
        oio, "_stream_base_hash", lambda p: (calls.__setitem__("n", calls["n"] + 1), orig(p))[1]
    )
    precompute_main.__wrapped__(cfg)  # re-run; per-shard skip then manifest re-write
    assert calls["n"] == 0
