"""Regenerate the committed clustered-selection golden (Task 2, A5).

This is a deterministic capture utility, NOT a test. It builds a small,
fully seed-derived synthetic packed-row shard covering: a multi-member
cluster with a mask-shifted representative ([3, 7, 11], mask blacklists 3
so the representative shifts from 3 to 7), a singleton-after-mask cluster
([20, 25], mask blacklists 25 leaving only 20), a plain multi-member cluster
with no masking ([40, 41, 42]), non-cluster features (50, 51, 55, 60), and
13 segments (>= 12) interleaving three fully-empty segments and one segment
that becomes empty only after masking ([3, 25], both masked out) -- so both
"raw-empty" and "empty-after-mask" RNG short-circuits are exercised. Runs
the fixture through ``scripts.preprocess.precompute_selected_features.
select_features_for_shard`` under a threshold selection config, a sampling
selection config with negative sampling on and a SHARED positive/negative
``torch.Generator`` (matching production convention), and (review fix
batch, lens-2 Important) a THIRD case, ``threshold_with_negatives``: a
threshold selection config combined with an INDEPENDENTLY-seeded
NegativeBundle over its own smaller >= 4-segment fixture -- mirroring
feature_pipeline.py's OTHER documented convention ("threshold path gives
NegativeBundle its own separate RNG"), which neither of the other two cases
exercises. Writes the resulting CSR selection arrays plus per-segment
``torch.Generator.get_state()`` SHA-256 digests to a committed JSON.

Regeneration procedure (per the A5 plan, Task 2 Step 1): this script must
be run ON MAIN, at the commit that still has the pre-delegation inline
clustered implementation (``ec6413fe`` at authoring time -- verify with
``git log --oneline -1 scripts/preprocess/precompute_selected_features.py``
if main has moved), NOT on a branch that already delegates to
``select_clustered_segment``. Running it on a delegating branch would pin
the new path against itself, which proves nothing. ``tests`` is not an
installed package, so the repo root must be on ``PYTHONPATH``:

    PYTHONPATH=. uv run python tests/steering/goldens/gen_cluster_selection_golden.py

It is import-safe (the run happens inside ``build_and_run``/``capture_golden``)
so ``build_and_run`` can be imported by the consuming test
(tests/steering/test_cluster_segment_selection_golden.py) to replay the
identical fixture through whichever implementation is checked out, without
that import re-running the capture or touching the committed golden file.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import torch as t

from scripts.preprocess.precompute_selected_features import select_features_for_shard
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.overlay_io import overlay_shard_path
from steering.packed_batch_io import (
    PackedRowShard,
    load_selection_shard,
    save_packed_row_shard,
)

GOLDEN_PATH = Path(__file__).parent / "cluster_selection_golden.json"

N_SAE_FEATURES = 64
SAMPLING_SEED = 20260721

# Cluster config: multi-member with a mask-shifted representative ([3,7,11],
# 3 masked -> rep shifts to 7), singleton-after-mask ([20,25], 25 masked ->
# only 20 survives), and a plain multi-member cluster with no masking
# ([40,41,42]).
CLUSTER_GROUPS: list[list[int]] = [[3, 7, 11], [20, 25], [40, 41, 42]]
MASK_BLACKLIST_IDS: tuple[int, ...] = (3, 25)

# 13 segments (>= 12): each (raw_feature_ids, raw_percentile_uint8_bytes).
# Interleaves 3 raw-empty segments (indices 2, 5, 10) and one
# empty-after-mask segment (index 8: both ids masked out) among segments
# mixing cluster survivors and non-cluster features.
SEGMENTS: list[tuple[list[int], list[int]]] = [
    ([7, 11, 50], [180, 140, 90]),
    ([3, 7, 50], [250, 160, 95]),
    ([], []),
    ([20, 25, 51], [200, 210, 60]),
    ([40, 41, 42, 55], [70, 220, 130, 40]),
    ([], []),
    ([7, 11, 20, 40, 41, 60], [110, 130, 240, 30, 200, 170]),
    ([50, 51, 55, 60], [20, 230, 45, 250]),
    ([3, 25], [255, 255]),
    ([7, 41, 50], [90, 190, 200]),
    ([], []),
    ([11, 42, 51, 60], [150, 60, 90, 210]),
    ([20, 40, 55], [130, 175, 60]),
]
assert len(SEGMENTS) >= 12

# Third case (review fix batch, lens-2 Important): threshold selection PLUS
# an independently-seeded NegativeBundle -- the "threshold" case above has no
# negatives at all, and "sampling" case's negatives share the SamplingBundle's
# rng, so neither exercises feature_pipeline.build_feature_pipeline's OTHER
# documented convention: "Threshold path: NegativeBundle gets its own
# separate RNG" (steering/conditional/feature_pipeline.py). Independently
# seeded from SAMPLING_SEED so it cannot be mistaken for a derived/shared
# stream. >= 4 segments, including one multi-member cluster surviving
# post-mask ([3, 7, 11] -> [7, 11]), one fully unmasked multi-member cluster
# ([40, 41, 42]), one post-mask-empty segment ([3, 25], both masked out),
# and one singleton-after-mask cluster ([20, 25] -> [20]).
THRESHOLD_NEGATIVE_SAMPLING_SEED = 20260722
THRESHOLD_WITH_NEGATIVES_SEGMENTS: list[tuple[list[int], list[int]]] = [
    ([7, 11, 50], [200, 150, 80]),
    ([3, 25], [255, 255]),
    ([40, 41, 42, 51], [90, 210, 60, 130]),
    ([20, 55], [175, 45]),
]
assert len(THRESHOLD_WITH_NEGATIVES_SEGMENTS) >= 4


def _shard_from_segments(segments: list[tuple[list[int], list[int]]]) -> PackedRowShard:
    n_segs = len(segments)
    seg_feat_offsets = [0]
    flat_ids: list[int] = []
    flat_pcts: list[int] = []
    for raw_ids, raw_pcts in segments:
        flat_ids.extend(raw_ids)
        flat_pcts.extend(raw_pcts)
        seg_feat_offsets.append(len(flat_ids))
    return PackedRowShard(
        tokens=t.zeros(n_segs, 9, dtype=t.uint16),
        row_seg_offsets=t.arange(n_segs + 1, dtype=t.int32),
        seg_token_start=t.zeros(n_segs, dtype=t.uint16),
        seg_token_end=t.full((n_segs,), 8, dtype=t.uint16),
        seg_doc_id=t.zeros(n_segs, dtype=t.uint8),
        seg_feat_offsets=t.tensor(seg_feat_offsets, dtype=t.int32),
        feature_ids=t.tensor(flat_ids, dtype=t.uint16),
        feature_pcts=t.tensor(flat_pcts, dtype=t.uint8),
    )


def _build_clusters(root: Path, mask: FeatureMask | None) -> FeatureClusterConsolidation:
    path = root / "clusters.json"
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": CLUSTER_GROUPS}), encoding="utf-8"
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}}, N_SAE_FEATURES
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _threshold_pipeline(mask: FeatureMask | None) -> FeaturePipelineConfig:
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.3),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(N_SAE_FEATURES, 3, dtype=t.float64),
        negative_bundle=None,
        effective_buckets=4,
        feature_mask=mask,
    )


def _sampling_pipeline(mask: FeatureMask | None) -> tuple[FeaturePipelineConfig, t.Generator]:
    """Sampling + negative sampling, sharing ONE generator (production
    convention: 'sampling path shares rng between SamplingBundle and
    NegativeBundle', see feature_pipeline.build_feature_pipeline)."""
    rng = t.Generator().manual_seed(SAMPLING_SEED)
    table = t.zeros(N_SAE_FEATURES, 99, dtype=t.float64)
    pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.normal,
                mean=2.0,
                std=0.01,
                min_k=1,
                max_k=3,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=table,
            rng=rng,
        ),
        boundaries=t.zeros(N_SAE_FEATURES, 3, dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal, mean=2.0, std=0.01, min_k=1, max_k=3,
            ),
            n_sae_features=N_SAE_FEATURES,
            rng=rng,
            feature_mask=mask,
        ),
        effective_buckets=4,
        feature_mask=mask,
    )
    return pipeline, rng


def _threshold_with_negatives_pipeline(
    mask: FeatureMask | None,
) -> tuple[FeaturePipelineConfig, t.Generator | None]:
    """Threshold selection + an INDEPENDENTLY-seeded NegativeBundle (mirrors
    feature_pipeline.build_feature_pipeline's threshold-path convention:
    'threshold path gives NegativeBundle its own separate RNG' -- unlike the
    sampling path's shared-generator convention captured by
    _sampling_pipeline above). The negative generator is seeded from
    THRESHOLD_NEGATIVE_SAMPLING_SEED, a distinct constant from SAMPLING_SEED,
    so it is never mistaken for a derived or shared stream."""
    neg_rng = t.Generator().manual_seed(THRESHOLD_NEGATIVE_SAMPLING_SEED)
    pipeline = FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.feature_percentile, 0.3),
            precomputed_thresholds=None,
        ),
        boundaries=t.zeros(N_SAE_FEATURES, 3, dtype=t.float64),
        negative_bundle=NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.normal, mean=2.0, std=0.01, min_k=1, max_k=3,
            ),
            n_sae_features=N_SAE_FEATURES,
            rng=neg_rng,
            feature_mask=mask,
        ),
        effective_buckets=4,
        feature_mask=mask,
    )
    return pipeline, neg_rng


def _state_digest(rng: t.Generator | None) -> str | None:
    if rng is None:
        return None
    state = rng.get_state()
    return hashlib.sha256(state.numpy().tobytes()).hexdigest()


def _run_case(
    root: Path,
    case_name: str,
    *,
    mask: FeatureMask | None,
    clusters: FeatureClusterConsolidation,
    pipeline_factory,
    segments: list[tuple[list[int], list[int]]],
) -> dict:
    # --- Whole-shard call: the canonical CSR arrays for this case. ---
    whole_pipeline, whole_rng = pipeline_factory()
    whole_base_dir = root / case_name / "whole" / "base"
    whole_base_dir.mkdir(parents=True)
    whole_shard_path = save_packed_row_shard(
        _shard_from_segments(segments), whole_base_dir / "shard_00000"
    )
    whole_out_dir = root / case_name / "whole" / "sel"
    whole_out_dir.mkdir(parents=True)
    select_features_for_shard(
        whole_shard_path, whole_pipeline, whole_out_dir, "shard_00000", feature_clusters=clusters
    )
    whole_selection = load_selection_shard(overlay_shard_path(whole_out_dir, "shard_00000"))
    whole_final_digest = _state_digest(whole_rng)

    # --- Per-segment calls: one 1-segment shard per SEGMENTS entry, reusing
    # a FRESH but identically-seeded pipeline across calls so its rng
    # advances cumulatively exactly as it would inside a real multi-segment
    # loop -- this exposes the intermediate per-segment RNG state that a
    # single whole-shard call cannot. ---
    per_seg_pipeline, per_seg_rng = pipeline_factory()
    per_segment: list[dict] = []
    for seg_idx, (raw_ids, raw_pcts) in enumerate(segments):
        seg_root = root / case_name / f"seg_{seg_idx:03d}"
        seg_base_dir = seg_root / "base"
        seg_base_dir.mkdir(parents=True)
        seg_shard_path = save_packed_row_shard(
            _shard_from_segments([(raw_ids, raw_pcts)]), seg_base_dir / "shard_00000"
        )
        seg_out_dir = seg_root / "sel"
        seg_out_dir.mkdir(parents=True)
        select_features_for_shard(
            seg_shard_path,
            per_seg_pipeline,
            seg_out_dir,
            "shard_00000",
            feature_clusters=clusters,
        )
        seg_selection = load_selection_shard(overlay_shard_path(seg_out_dir, "shard_00000"))
        per_segment.append(
            {
                "segment": seg_idx,
                "positive_ids": seg_selection.sel_fids.tolist(),
                "positive_buckets": seg_selection.sel_buckets.tolist(),
                "positive_percentiles_uint8": seg_selection.sel_pcts.tolist(),
                "negative_ids": seg_selection.neg_fids.tolist(),
                "rng_state_sha256": _state_digest(per_seg_rng),
            }
        )

    # Internal self-consistency: the per-segment replay must reconstruct the
    # exact same whole-shard CSR arrays and final RNG state as the direct
    # whole-shard call. This is a hard invariant of THIS fixture harness
    # (not something the golden JSON records) -- a failure here means the
    # fixture's per-segment/whole-shard equivalence assumption is broken,
    # not that select_features_for_shard itself is wrong.
    replay_offsets = [0]
    replay_fids: list[int] = []
    replay_buckets: list[int] = []
    replay_pcts: list[int] = []
    replay_neg_offsets = [0]
    replay_neg_fids: list[int] = []
    for seg in per_segment:
        replay_fids.extend(seg["positive_ids"])
        replay_buckets.extend(seg["positive_buckets"])
        replay_pcts.extend(seg["positive_percentiles_uint8"])
        replay_offsets.append(len(replay_fids))
        replay_neg_fids.extend(seg["negative_ids"])
        replay_neg_offsets.append(len(replay_neg_fids))
    assert replay_offsets == whole_selection.sel_offsets.tolist(), case_name
    assert replay_fids == whole_selection.sel_fids.tolist(), case_name
    assert replay_buckets == whole_selection.sel_buckets.tolist(), case_name
    assert replay_pcts == whole_selection.sel_pcts.tolist(), case_name
    assert replay_neg_offsets == whole_selection.neg_offsets.tolist(), case_name
    assert replay_neg_fids == whole_selection.neg_fids.tolist(), case_name
    if whole_final_digest is not None:
        assert per_segment[-1]["rng_state_sha256"] == whole_final_digest, case_name

    return {
        "whole_shard": {
            "sel_offsets": whole_selection.sel_offsets.tolist(),
            "sel_fids": whole_selection.sel_fids.tolist(),
            "sel_buckets": whole_selection.sel_buckets.tolist(),
            "sel_pcts": whole_selection.sel_pcts.tolist(),
            "neg_offsets": whole_selection.neg_offsets.tolist(),
            "neg_fids": whole_selection.neg_fids.tolist(),
        },
        "per_segment": per_segment,
        "final_rng_state_sha256": whole_final_digest,
    }


def build_and_run(root: Path) -> dict:
    """Build the fixture and run all three selection cases under whatever
    ``select_features_for_shard`` implementation is currently checked out.

    Pure function of ``root`` (a caller-supplied scratch directory) and the
    fixed module-level constants above -- no wall-clock, no unseeded random.
    Used both by ``capture_golden`` (writes the committed JSON) and by
    tests/steering/test_cluster_segment_selection_golden.py (replays the
    identical fixture on THIS branch and compares against the golden).
    """
    mask = FeatureMask(FeatureMaskMode.blacklist, MASK_BLACKLIST_IDS)
    clusters = _build_clusters(root, mask)

    def threshold_factory() -> tuple[FeaturePipelineConfig, t.Generator | None]:
        return _threshold_pipeline(mask), None

    def sampling_factory() -> tuple[FeaturePipelineConfig, t.Generator | None]:
        return _sampling_pipeline(mask)

    def threshold_with_negatives_factory() -> tuple[FeaturePipelineConfig, t.Generator | None]:
        return _threshold_with_negatives_pipeline(mask)

    return {
        "schema_version": 1,
        "n_sae_features": N_SAE_FEATURES,
        "cluster_groups": CLUSTER_GROUPS,
        "mask_blacklist_ids": list(MASK_BLACKLIST_IDS),
        "cases": {
            "threshold": _run_case(
                root,
                "threshold",
                mask=mask,
                clusters=clusters,
                pipeline_factory=threshold_factory,
                segments=SEGMENTS,
            ),
            "sampling": _run_case(
                root,
                "sampling",
                mask=mask,
                clusters=clusters,
                pipeline_factory=sampling_factory,
                segments=SEGMENTS,
            ),
            "threshold_with_negatives": _run_case(
                root,
                "threshold_with_negatives",
                mask=mask,
                clusters=clusters,
                pipeline_factory=threshold_with_negatives_factory,
                segments=THRESHOLD_WITH_NEGATIVES_SEGMENTS,
            ),
        },
    }


def capture_golden(golden_path: Path = GOLDEN_PATH) -> Path:
    """Build the fixture, run it, and write the committed golden JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        result = build_and_run(Path(tmp))
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return golden_path


if __name__ == "__main__":
    written = capture_golden()
    print(f"wrote golden JSON: {written} ({written.stat().st_size} bytes)")
