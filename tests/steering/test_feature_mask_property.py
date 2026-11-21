"""Property-based tests for the feature-mask feature (Task 17).

Uses the ``hypothesis`` library (available: 6.151.9).  All tests are kept
fast via ``@settings(max_examples=50)`` and deterministic seeds.

  P1 — ∀ random (seed, n_sae, mask mode, random valid id subset):
         apply_to_positive output contains NO blacklisted id / ONLY
         whitelisted ids; allowed_universe_mask consistent; and
         sample_inactive_features_batched emits only allowed ids.

  P2 — disabled-mask compute_selection_hash == the no-feature_mask-arg hash
         for random base payloads (I1 invariance); a set mask changes it;
         blacklist vs whitelist of same ids differ.

  P3 — canonical invariance: random permutations / different source formats
         (inline list vs a tmp .txt vs .yaml) of the SAME logical id set +
         mode ⇒ identical canonical_dict, feature_mask_dirname,
         feature_mask_leaf_from_cfg, and compute_selection_hash contribution.

  P4 — determinism: identical (config, seed, mask) ⇒ identical
         select_features_for_shard SelectionShard tensors (bytes) over
         repeated runs.
"""

from __future__ import annotations

import os
import random
import tempfile
from pathlib import Path

import torch as t
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st
from omegaconf import OmegaConf

from scripts.preprocess.precompute_selected_features import select_features_for_shard
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
    feature_mask_dirname,
    feature_mask_leaf_from_cfg,
    resolve_feature_mask,
)
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    sample_inactive_features_batched,
)
from steering.overlay_io import overlay_shard_path
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    compute_selection_hash,
    load_selection_shard,
    save_packed_row_shard,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Shared strategies
# ---------------------------------------------------------------------------

# Keep n_sae small for speed
_N_SAE_STRATEGY = st.integers(min_value=10, max_value=80)
_MODE_STRATEGY = st.sampled_from(["blacklist", "whitelist"])

_BASE_POSITIVE_SELECTION = {"threshold_type": "segment_topk", "threshold": 2048}


def _ids_strategy(n_sae: int, min_size: int = 1, max_size: int | None = None) -> st.SearchStrategy:
    """Draw a unique, sorted subset of [0, n_sae) of size [min_size, max_size]."""
    if max_size is None:
        max_size = max(1, n_sae - 1)  # can't whitelist ALL features (some must be inactive)
    max_size = min(max_size, n_sae - 1)
    return st.lists(
        st.integers(min_value=0, max_value=n_sae - 1),
        min_size=min_size,
        max_size=max_size,
        unique=True,
    ).map(sorted)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_per_feature_stats(n: int) -> dict[str, PerFeatureStat]:
    """Build a tiny stats dict with deterministic percentiles."""
    full_percentiles = {f"p{k}": k * 0.01 + 0.5 for k in range(1, 100)}
    full_percentiles["p95"] = 1.45
    full_percentiles["p98"] = 1.48
    out: dict[str, PerFeatureStat] = {}
    for i in range(n):
        out[str(i)] = PerFeatureStat(
            count=100 - i % 100,
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


def _make_tiny_shard(n_sae: int, seed: int = 42) -> PackedRowShard:
    """Build a minimal PackedRowShard for property tests."""
    t.manual_seed(seed)
    n_rows, segs_per_row, feats_per_seg, ctx = 2, 2, 8, 16
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg
    seg_size = ctx // segs_per_row
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = (
                ctx if s == segs_per_row - 1 else (s + 1) * seg_size
            )
    return PackedRowShard(
        tokens=t.randint(1, 100, (n_rows, ctx + 1), dtype=t.uint16),
        row_seg_offsets=t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32),
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=t.zeros(total_segs, dtype=t.uint8),
        seg_feat_offsets=t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32),
        # Real packed rows list each segment's ACTIVE features once — they come from
        # ``(pooled > 0).nonzero()`` in extraction, so per-segment feature ids are
        # unique. Draw without replacement per segment to match that invariant (the
        # precompute percentile lookup asserts exactly one entry per selected id).
        feature_ids=t.cat([
            t.randperm(n_sae)[:feats_per_seg] for _ in range(total_segs)
        ]).to(t.uint16),
        feature_pcts=t.randint(100, 255, (total_feats,), dtype=t.uint8),
    )


# ---------------------------------------------------------------------------
# P1 — apply_to_positive / allowed_universe_mask / sample_inactive_features_batched
# ---------------------------------------------------------------------------


class TestP1ApplyAndSampleInvariants:
    """P1: for random (seed, n_sae, mode, ids):
         apply_to_positive respects the mask;
         allowed_universe_mask is consistent;
         sample_inactive_features_batched emits only allowed ids.
    """

    @given(
        n_sae=_N_SAE_STRATEGY,
        mode=_MODE_STRATEGY,
        ids=st.integers(min_value=10, max_value=80).flatmap(
            lambda n: _ids_strategy(n, min_size=1, max_size=min(5, n - 1))
        ),
    )
    @settings(max_examples=50, derandomize=True)
    def test_apply_to_positive_blacklist(
        self, n_sae: int, mode: str, ids: list[int]
    ) -> None:
        """apply_to_positive with blacklist: no blacklisted id in output."""
        # Bound ids to n_sae
        ids_bounded = [i for i in ids if i < n_sae]
        if not ids_bounded:
            return  # vacuous
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=tuple(ids_bounded))
        fids = t.arange(0, min(n_sae, 20), dtype=t.int64)
        mags = t.ones(len(fids), dtype=t.float32)
        pcts = t.full((len(fids),), 128, dtype=t.uint8)
        out_fids, _, _ = mask.apply_to_positive(fids, mags, pcts)
        blacklisted = set(mask.ids)
        bad = set(out_fids.tolist()) & blacklisted
        assert not bad, f"apply_to_positive blacklist: {bad} in output"

    @given(
        n_sae=_N_SAE_STRATEGY,
        ids=st.integers(min_value=10, max_value=80).flatmap(
            lambda n: _ids_strategy(n, min_size=1, max_size=min(5, n - 1))
        ),
    )
    @settings(max_examples=50, derandomize=True)
    def test_apply_to_positive_whitelist(
        self, n_sae: int, ids: list[int]
    ) -> None:
        """apply_to_positive with whitelist: output ⊆ whitelist."""
        ids_bounded = [i for i in ids if i < n_sae]
        if not ids_bounded:
            return
        mask = FeatureMask(mode=FeatureMaskMode.whitelist, ids=tuple(ids_bounded))
        fids = t.arange(0, min(n_sae, 20), dtype=t.int64)
        mags = t.ones(len(fids), dtype=t.float32)
        pcts = t.full((len(fids),), 128, dtype=t.uint8)
        out_fids, _, _ = mask.apply_to_positive(fids, mags, pcts)
        allowed = set(mask.ids)
        bad = set(out_fids.tolist()) - allowed
        assert not bad, f"apply_to_positive whitelist: {bad} not in whitelist"

    @given(
        n_sae=_N_SAE_STRATEGY,
        mode=_MODE_STRATEGY,
        ids=st.integers(min_value=10, max_value=80).flatmap(
            lambda n: _ids_strategy(n, min_size=1, max_size=min(5, n - 1))
        ),
    )
    @settings(max_examples=50, derandomize=True)
    def test_allowed_universe_mask_consistent(
        self, n_sae: int, mode: str, ids: list[int]
    ) -> None:
        """allowed_universe_mask shape and values are consistent with the mask mode."""
        ids_bounded = [i for i in ids if i < n_sae]
        if not ids_bounded:
            return
        mask = FeatureMask(mode=FeatureMaskMode(mode), ids=tuple(ids_bounded))
        univ = mask.allowed_universe_mask(n_sae)
        assert univ.shape == (n_sae,), f"allowed_universe_mask shape {univ.shape} != ({n_sae},)"
        mask_set = set(mask.ids)
        for i in range(n_sae):
            expected = (i not in mask_set) if mode == "blacklist" else (i in mask_set)
            assert univ[i].item() is expected, (
                f"allowed_universe_mask[{i}] wrong for mode={mode}, ids={mask_set}"
            )

    @given(
        n_sae=_N_SAE_STRATEGY,
        mode=_MODE_STRATEGY,
        ids=st.integers(min_value=10, max_value=80).flatmap(
            lambda n: _ids_strategy(n, min_size=1, max_size=min(4, n - 1))
        ),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=200, derandomize=True)
    def test_sample_inactive_features_batched_I4(
        self, n_sae: int, mode: str, ids: list[int], seed: int
    ) -> None:
        """sample_inactive_features_batched emits only allowed ids under any mask."""
        ids_bounded = [i for i in ids if i < n_sae]
        if not ids_bounded:
            return
        mask = FeatureMask(mode=FeatureMaskMode(mode), ids=tuple(ids_bounded))
        cfg = NegativeSamplingSelection(mean=3.0, std=3.0, min_k=1, max_k=4)
        rng = t.Generator().manual_seed(seed)
        results = sample_inactive_features_batched(
            [t.tensor([], dtype=t.int64)],
            n_sae,
            cfg,
            rng,
            feature_mask=mask,
        )
        if not results or results[0].numel() == 0:
            return  # empty result is fine
        out = set(results[0].tolist())
        if mode == "blacklist":
            forbidden = set(mask.ids)
            bad = out & forbidden
            assert not bad, f"P1 batched blacklist: {bad} in output (n_sae={n_sae})"
        else:
            whitelist = set(mask.ids)
            bad = out - whitelist
            assert not bad, f"P1 batched whitelist: {bad} not in whitelist (n_sae={n_sae})"


# ---------------------------------------------------------------------------
# P2 — compute_selection_hash I1 invariance
# ---------------------------------------------------------------------------


class TestP2SelectionHashInvariance:
    """P2: disabled-mask hash == no-arg hash (I1); set mask changes it; bl vs wl differ."""

    @given(
        seed=st.integers(min_value=0, max_value=1000),
        num_buckets=st.integers(min_value=4, max_value=32),
        ddp_rank=st.integers(min_value=0, max_value=7),
    )
    @settings(max_examples=50, derandomize=True)
    def test_disabled_mask_equals_no_arg(
        self, seed: int, num_buckets: int, ddp_rank: int
    ) -> None:
        """feature_mask=None must produce the same hash as omitting the argument."""
        h_no_arg = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, num_buckets, seed, ddp_rank
        )
        h_none = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, num_buckets, seed, ddp_rank,
            feature_mask=None,
        )
        assert h_no_arg == h_none, (
            f"P2 I1 violation: feature_mask=None changed hash for seed={seed}, "
            f"num_buckets={num_buckets}, ddp_rank={ddp_rank}"
        )

    @given(
        seed=st.integers(min_value=0, max_value=1000),
        ids=_ids_strategy(50, min_size=1, max_size=5),
    )
    @settings(max_examples=50, derandomize=True)
    def test_set_mask_changes_hash(self, seed: int, ids: list[int]) -> None:
        """A set mask produces a different hash than no mask."""
        h_base = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, 8, seed, 0
        )
        h_bl = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, 8, seed, 0,
            feature_mask={"mode": "blacklist", "ids": ids},
        )
        assert h_base != h_bl, (
            f"P2: set blacklist mask did not change hash (seed={seed}, ids={ids})"
        )

    @given(
        seed=st.integers(min_value=0, max_value=1000),
        ids=_ids_strategy(50, min_size=1, max_size=5),
    )
    @settings(max_examples=50, derandomize=True)
    def test_blacklist_vs_whitelist_differ(self, seed: int, ids: list[int]) -> None:
        """Blacklist and whitelist of same ids produce different hashes."""
        h_bl = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, 8, seed, 0,
            feature_mask={"mode": "blacklist", "ids": ids},
        )
        h_wl = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, 8, seed, 0,
            feature_mask={"mode": "whitelist", "ids": ids},
        )
        assert h_bl != h_wl, (
            f"P2: blacklist and whitelist of same ids produce same hash "
            f"(seed={seed}, ids={ids})"
        )


# ---------------------------------------------------------------------------
# P3 — canonical invariance across source formats and permutations
# ---------------------------------------------------------------------------


class TestP3CanonicalInvariance:
    """P3: same logical (mode, ids) ⇒ identical canonical_dict, dirname, leaf, hash.

    Covers:
      - Permutations of the id list (sort-invariance)
      - Inline list vs .txt file vs .yaml file
    """

    @given(
        n_sae=_N_SAE_STRATEGY,
        mode=_MODE_STRATEGY,
        ids=st.integers(min_value=10, max_value=60).flatmap(
            lambda n: _ids_strategy(n, min_size=1, max_size=min(6, n - 1))
        ),
        seed=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=50, derandomize=True)
    def test_permutation_canonical_invariance(
        self, n_sae: int, mode: str, ids: list[int], seed: int
    ) -> None:
        """Permuting the id list in the cfg produces identical canonical outputs."""
        ids_bounded = [i for i in ids if i < n_sae]
        if len(ids_bounded) < 2:
            return
        rng = random.Random(seed)
        perm = list(ids_bounded)
        rng.shuffle(perm)

        cfg_sorted = {"feature_mask": {"mode": mode, "ids": sorted(ids_bounded)}}
        cfg_perm = {"feature_mask": {"mode": mode, "ids": perm}}

        m_sorted = resolve_feature_mask(cfg_sorted, n_sae)
        m_perm = resolve_feature_mask(cfg_perm, n_sae)

        assert m_sorted is not None and m_perm is not None
        assert canonical_dict(m_sorted) == canonical_dict(m_perm), (
            "P3: canonical_dict differs between sorted and permuted ids"
        )
        assert feature_mask_dirname(m_sorted) == feature_mask_dirname(m_perm), (
            "P3: feature_mask_dirname differs between sorted and permuted ids"
        )

        # feature_mask_leaf_from_cfg also invariant
        leaf_sorted = feature_mask_leaf_from_cfg(cfg_sorted)
        leaf_perm = feature_mask_leaf_from_cfg(cfg_perm)
        assert leaf_sorted == leaf_perm, (
            "P3: feature_mask_leaf_from_cfg differs between sorted and permuted ids"
        )

        # compute_selection_hash contribution is also invariant
        h_sorted = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, 8, 0, 0,
            feature_mask=canonical_dict(m_sorted),
        )
        h_perm = compute_selection_hash(
            _BASE_POSITIVE_SELECTION, None, None, 8, 0, 0,
            feature_mask=canonical_dict(m_perm),
        )
        assert h_sorted == h_perm, (
            "P3: compute_selection_hash differs between sorted and permuted ids"
        )

    @given(
        n_sae=_N_SAE_STRATEGY,
        mode=_MODE_STRATEGY,
        ids=st.integers(min_value=10, max_value=60).flatmap(
            lambda n: _ids_strategy(n, min_size=1, max_size=min(4, n - 1))
        ),
    )
    @settings(max_examples=40, derandomize=True)
    def test_format_invariance_inline_vs_txt_vs_yaml(
        self, n_sae: int, mode: str, ids: list[int]
    ) -> None:
        """Same ids via inline list, .txt, and .yaml produce identical canonical outputs."""
        ids_bounded = [i for i in ids if i < n_sae]
        if not ids_bounded:
            return

        cfg_inline = {"feature_mask": {"mode": mode, "ids": sorted(ids_bounded)}}
        mask_inline = resolve_feature_mask(cfg_inline, n_sae)
        assert mask_inline is not None

        with tempfile.TemporaryDirectory() as tmp:
            # .txt format
            txt_path = os.path.join(tmp, "ids.txt")
            with open(txt_path, "w") as f:
                f.write("# generated by test\n")
                for i in sorted(ids_bounded):
                    f.write(f"{i}\n")
            cfg_txt = {"feature_mask": {"mode": mode, "path": txt_path}}
            mask_txt = resolve_feature_mask(cfg_txt, n_sae)
            assert mask_txt is not None

            # .yaml format
            yaml_path = os.path.join(tmp, "ids.yaml")
            with open(yaml_path, "w") as f:
                yaml.dump(sorted(ids_bounded), f)
            cfg_yaml = {"feature_mask": {"mode": mode, "path": yaml_path}}
            mask_yaml = resolve_feature_mask(cfg_yaml, n_sae)
            assert mask_yaml is not None

            # All three must agree
            cd_inline = canonical_dict(mask_inline)
            cd_txt = canonical_dict(mask_txt)
            cd_yaml = canonical_dict(mask_yaml)

            assert cd_inline == cd_txt, "P3: canonical_dict differs: inline vs txt"
            assert cd_inline == cd_yaml, "P3: canonical_dict differs: inline vs yaml"

            dn_inline = feature_mask_dirname(mask_inline)
            dn_txt = feature_mask_dirname(mask_txt)
            dn_yaml = feature_mask_dirname(mask_yaml)

            assert dn_inline == dn_txt, "P3: dirname differs: inline vs txt"
            assert dn_inline == dn_yaml, "P3: dirname differs: inline vs yaml"


# ---------------------------------------------------------------------------
# P4 — determinism: same (config, seed, mask) ⇒ identical SelectionShard bytes
# ---------------------------------------------------------------------------


class TestP4Determinism:
    """P4: select_features_for_shard is deterministic under fixed seed + mask.

    Two independent runs (separate tmp dirs) with identical (shard, pipeline, seed, mask)
    must produce bit-identical SelectionShard tensors.
    """

    @given(
        mode=_MODE_STRATEGY,
        ids=_ids_strategy(64, min_size=1, max_size=4),
        seed=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=20, derandomize=True, deadline=None)
    def test_select_features_deterministic(
        self, mode: str, ids: list[int], seed: int
    ) -> None:
        """Two identical precompute runs produce byte-identical SelectionShard tensors."""
        mask = FeatureMask(mode=FeatureMaskMode(mode), ids=tuple(ids))
        n_sae = 64

        cfg = OmegaConf.create({
            "feature_selection": {"distribution": "geometric", "mean": 4.0, "std": 4.0},
            "negative_selection": {"distribution": "geometric", "mean": 3.0, "std": 3.0},
        })

        shard = _make_tiny_shard(n_sae, seed=seed)

        results: list[SelectionShard] = []
        for _run in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                shard_dir = tmp_path / "base"
                shard_dir.mkdir()
                shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

                pipeline = build_feature_pipeline_from_cfg_block(
                    cfg,
                    per_feature_stats=_make_per_feature_stats(n_sae),
                    n_sae_features=n_sae,
                    num_buckets=8,
                    seed=seed,
                    feature_mask=mask,
                )
                out_dir = tmp_path / "sel"
                out_dir.mkdir()
                select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
                sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
                results.append(sel)

        sel_a, sel_b = results
        assert t.equal(sel_a.sel_fids, sel_b.sel_fids), (
            f"P4: sel_fids not identical across runs (mode={mode}, ids={ids}, seed={seed})"
        )
        assert t.equal(sel_a.sel_buckets, sel_b.sel_buckets), (
            f"P4: sel_buckets not identical across runs (mode={mode}, ids={ids}, seed={seed})"
        )
        assert t.equal(sel_a.neg_fids, sel_b.neg_fids), (
            f"P4: neg_fids not identical across runs (mode={mode}, ids={ids}, seed={seed})"
        )
        assert t.equal(sel_a.sel_offsets, sel_b.sel_offsets), (
            "P4: sel_offsets not identical across runs"
        )
        assert t.equal(sel_a.neg_offsets, sel_b.neg_offsets), (
            "P4: neg_offsets not identical across runs"
        )

    def test_determinism_no_mask(self) -> None:
        """Sanity: two identical no-mask precompute runs are also deterministic."""
        n_sae = 32
        seed = 5

        cfg = OmegaConf.create({
            "feature_selection": {"distribution": "geometric", "mean": 3.0, "std": 3.0},
        })
        shard = _make_tiny_shard(n_sae, seed=seed)

        results: list[SelectionShard] = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as tmp:
                tmp_path = Path(tmp)
                shard_dir = tmp_path / "base"
                shard_dir.mkdir()
                shard_path = save_packed_row_shard(shard, shard_dir / "shard_00000")

                pipeline = build_feature_pipeline_from_cfg_block(
                    cfg,
                    per_feature_stats=_make_per_feature_stats(n_sae),
                    n_sae_features=n_sae,
                    num_buckets=4,
                    seed=seed,
                    feature_mask=None,
                )
                out_dir = tmp_path / "sel"
                out_dir.mkdir()
                select_features_for_shard(shard_path, pipeline, out_dir, "shard_00000")
                sel = load_selection_shard(overlay_shard_path(out_dir, "shard_00000"))
                results.append(sel)

        sel_a, sel_b = results
        assert t.equal(sel_a.sel_fids, sel_b.sel_fids), (
            "P4 (no mask): sel_fids not identical across runs"
        )
        assert t.equal(sel_a.neg_fids, sel_b.neg_fids), (
            "P4 (no mask): neg_fids not identical across runs"
        )
