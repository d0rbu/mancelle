"""Canonical feature_selection_hash for PR-C cache identity.

Wraps :func:`steering.packed_batch_io.compute_selection_hash` with
``seed=0`` / ``ddp_rank=0`` (those are run-specific, not pipeline-specific).
Both ``build_eval_context_cache`` and PR-C target-cache callers use this to
populate :class:`steering.eval.context_cache_io.CacheIdentity.feature_selection_hash`.
Operators stop computing the hash by hand.

S-07-11 P0 fix (audit 2026-05-05): the wrapper now hashes the ``feature_delta``
config alongside ``stats_source_sha256``. Pre-fix code only hashed
``stats_source_sha256`` so two pipelines that differed only in
``feature_delta.mode`` (``prev_segment`` vs ``cum_max``) or
``feature_delta.neg_delta_ratio`` collided on the eval-context-cache identity.

H7 fix (audit 2026-05-18): the wrapper now accepts ``feature_mask`` so that a
masked eval-context-cache build produces a DIFFERENT identity hash from an
unmasked build (cache non-collision). When ``feature_mask`` is ``None`` (or
omitted), the hash is byte-identical to the pre-H7 value — existing unmasked
caches remain valid (cache-identity I1).

Backwards compatibility: when ``feature_delta`` is ``None`` (or omitted), the
emitted ``delta_config`` is ``{"stats_source_sha256": ...}`` — byte-identical to
the pre-fix layout — so existing eval caches built without a feature_delta
block remain valid. Caches that WERE built against a feature_delta config
(currently nothing in production wires feature_delta into the eval-cache
path; build_feature_pipeline_from_cfg uses ``feature_delta=None`` per
``steering/eval/bpb_pipeline.py:234``) need to be re-baselined.
"""

from __future__ import annotations

from steering.packed_batch_io import compute_selection_hash


def compute_canonical_feature_selection_hash(
    *,
    positive_selection: dict,
    negative_selection: dict | None,
    num_buckets: int,
    stats_source_sha256: str,
    feature_delta: dict | None = None,
    feature_mask: dict | None = None,
    feature_clusters: dict | None = None,
    splice_batch_size: object = None,
) -> str:
    """Return a 64-char hex SHA-256 hash of the given pipeline parameters.

    Args:
        positive_selection: Plain-dict form of ``feature_selection`` config.
        negative_selection: Plain-dict form of ``negative_selection`` config
            (or ``None`` when negatives are disabled).
        num_buckets: Discretization bucket count.
        stats_source_sha256: Hex digest of the source statistics file.
        feature_delta: Plain-dict form of ``feature_delta`` config (or
            ``None`` when delta conditioning is disabled). When ``None``, the
            emitted ``delta_config`` matches the pre-S-07-11-fix layout for
            backwards compatibility with on-disk caches.
        feature_mask: Canonical plain-dict form of the feature mask (i.e.
            ``canonical_dict(feature_mask)`` — ``{mode, ids}`` — or ``None``
            when no mask is applied). When ``None`` the hash is byte-identical
            to the pre-H7 value (cache-identity I1 preserved).
        feature_clusters: Canonical path-independent cluster descriptor, or
            ``None`` when static consolidation is disabled.
        splice_batch_size: Validated splice-selection execution bound. Null/1
            preserves the legacy payload; values above one are conditionally
            hashed only for ``splice_counterfactual`` delta mode.

    Returns:
        64-character lowercase hex SHA-256 string.

    ANTI-DESYNC: this cache-identity hash is the eval-context-cache counterpart
    of the packed-row-directory ``compute_selection_hash`` calls at:
      1. scripts/preprocess/precompute_selected_features.py
      2. scripts/train/conditional_encdec_train.py:discover_packed_selection
      3. scripts/eval/base_eval.py:_resolve_bpb_selection_dir
      4. steering/eval/feature_selection_hash.py (THIS function — cache identity)
    All four must agree on feature-mask, cluster, and splice-batching semantics.
    Change ALL 4 or none.
    """
    if feature_delta is None:
        delta_config: dict = {"stats_source_sha256": stats_source_sha256}
    else:
        # Merge feature_delta keys with stats_source_sha256 into a single
        # dict; the underlying compute_selection_hash uses
        # json.dumps(sort_keys=True) so insertion order is irrelevant.
        delta_config = {
            **dict(feature_delta),
            "stats_source_sha256": stats_source_sha256,
        }
    return compute_selection_hash(
        positive_selection=positive_selection,
        negative_selection=negative_selection,
        delta_config=delta_config,
        num_buckets=num_buckets,
        seed=0,
        ddp_rank=0,
        feature_mask=feature_mask,
        feature_clusters=feature_clusters,
        splice_batch_size=splice_batch_size,
    )
