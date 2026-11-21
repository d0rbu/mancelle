"""Roofline general-validation BPB helpers.

Roofline precomputed-bag methods are per-prefix in the concept-margin path. For
generic validation there is no concept prefix, so the comparable oracle is
per-document conditioning: derive the validation document's own output-window
signal, optionally GCG-optimize it against the document's masked LM loss, and
score that same document under the resulting bag.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch as t
from beartype import beartype
from loguru import logger

from errors import FeatureClustersError, FeatureMaskError
from steering.conditional.discretization import percentile_to_bucket
from steering.conditional.feature_gcg.config import FeatureGcgConfig
from steering.conditional.feature_gcg.feature_conditioning import FeatureConditioning
from steering.conditional.feature_gcg.gradient import FeatureGradMember
from steering.conditional.feature_gcg.objective import FeatureLmItem, lm_loss_on_masked_tokens
from steering.conditional.feature_gcg.optimize import optimize_feature_conditioning_universal
from steering.conditional.feature_gcg.partition import (
    PartitionBounds,
    is_active,
    partition_bounds_from_model,
)
from steering.conditional.percentile_table import (
    compute_percentile_checksum,
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    parse_negative_selection,
    sample_inactive_features_batched,
)
from steering.eval.axbench_bpb.context_cache import threshold_and_discretize_features
from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.eval.axbench_bpb.vocab_restriction import (
    restrict_segment_conditions_to_vocab,
    restrict_to_vocab,
)
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.feature_diffmean import (
    PrefixedCompletion,
    _max_pooled_sae_activations,
)
from steering.eval.cluster_feature_contract import GENERIC_ROOFLINE_SIGNAL_VERSION
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import NonNegativeIntLike, PositiveIntLike

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract

GENERAL_VAL_SIGNAL_POLICY = "top_k_by_partition_full_percentile_v1"


@dataclass(frozen=True)
class GeneralValSignalBudget:
    """Sparse roofline output-signal budget resolved from the target checkpoint."""

    k_pos: NonNegativeIntLike
    k_neg: NonNegativeIntLike
    positive_bucket: NonNegativeIntLike
    negative_bucket: NonNegativeIntLike

    def __post_init__(self) -> None:
        if self.positive_bucket == self.negative_bucket:
            raise AxBenchBpbAdvError(
                "roofline general_val: positive_bucket and negative_bucket "
                f"must differ, got {self.positive_bucket}"
            )

    def metadata(
        self,
        *,
        cluster_contract: ClusterFeatureContract | None = None,
    ) -> dict[str, str]:
        """Return signal identity, conditionally bound to the cluster contract.

        Identity-equivalent and consolidating clustered contracts share
        ``GENERIC_ROOFLINE_SIGNAL_VERSION`` even though the former uses the
        legacy magnitude/bucket currency and the latter uses quantized
        normalized percentiles. Their simultaneously stamped feature-contract
        fingerprints distinguish those semantics in persisted bag identity.
        """

        metadata = {
            "signal_policy": (
                GENERAL_VAL_SIGNAL_POLICY
                if cluster_contract is None
                else GENERIC_ROOFLINE_SIGNAL_VERSION
            ),
            "signal_k_pos": str(int(self.k_pos)),
            "signal_k_neg": str(int(self.k_neg)),
            "positive_bucket": str(int(self.positive_bucket)),
            "negative_bucket": str(int(self.negative_bucket)),
        }
        if cluster_contract is not None:
            metadata["feature_contract_fingerprint"] = (
                cluster_contract.feature_contract_fingerprint
            )
        return metadata


GeneralValSignalBudget.__init__ = beartype(GeneralValSignalBudget.__init__)


@dataclass(frozen=True)
class GeneralValWarmStartCaps:
    """Exclusive GCG partition caps for one roofline warm-start bag."""

    max_k_pos: PositiveIntLike
    max_k_neg: PositiveIntLike


GeneralValWarmStartCaps.__init__ = beartype(GeneralValWarmStartCaps.__init__)


def _segment_condition_dict(seg: ContextFeatureSegment) -> dict[int, int]:
    return {int(fid): int(bucket) for fid, bucket in zip(seg.feature_ids, seg.buckets, strict=True)}


def _clustered_signal_seed(
    *,
    cluster_contract: ClusterFeatureContract,
    corpus_signature: str,
    sample_id: str,
) -> int:
    """Return the private per-document negative-selection seed."""

    payload = [
        int(cluster_contract.selection.preselection_seed),
        cluster_contract.feature_contract_fingerprint,
        corpus_signature,
        sample_id,
        GENERIC_ROOFLINE_SIGNAL_VERSION,
    ]
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def _validate_clustered_signal_inputs(
    *,
    mags: t.Tensor,
    percentile_table: t.Tensor,
    extraction_threshold: float,
    signal_budget: GeneralValSignalBudget,
    allowed_ids: t.Tensor | None,
    cluster_contract: ClusterFeatureContract,
    corpus_signature: str | None,
    sample_id: str,
) -> tuple[t.Tensor, str]:
    """Validate the exact clustered per-document signal contract."""

    if not isinstance(corpus_signature, str) or not corpus_signature:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output-signal derivation requires a "
            "non-empty general-validation corpus_signature"
        )
    if not sample_id:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output-signal derivation requires a "
            "non-empty sample_id"
        )
    n_sae_features = int(cluster_contract.n_sae_features)
    if mags.ndim != 1 or int(mags.numel()) != n_sae_features:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output-window activation width does "
            "not match the validated original SAE width: "
            f"shape={tuple(mags.shape)}, n_sae_features={n_sae_features}"
        )
    if not bool(t.isfinite(mags).all()) or bool((mags < 0).any()):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output-window SAE activations must "
            "be finite and non-negative"
        )
    if float(extraction_threshold) != 0.0:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output-signal derivation requires "
            "extraction_threshold=0.0 so raw SAE presence matches clustered "
            "training preselection"
        )
    table = percentile_table.detach().to(device="cpu", dtype=t.float64)
    if table.ndim != 2 or tuple(table.shape) != (n_sae_features, 99):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered percentile table must have shape "
            f"({n_sae_features}, 99), got {tuple(table.shape)}"
        )
    actual_checksum = compute_percentile_checksum(table)
    expected_checksum = cluster_contract.statistics.percentile_table_checksum
    if actual_checksum != expected_checksum:
        raise AxBenchBpbAdvError(
            "roofline general_val: percentile table does not match the clustered "
            "feature contract statistics: "
            f"expected={expected_checksum}, actual={actual_checksum}"
        )

    consolidation = cluster_contract.consolidation
    if consolidation.n_sae_features != n_sae_features:
        raise AxBenchBpbAdvError(
            "roofline general_val: cluster consolidation original SAE width "
            "does not match the feature contract"
        )
    if consolidation.feature_mask != cluster_contract.feature_mask:
        raise AxBenchBpbAdvError(
            "roofline general_val: cluster consolidation mask does not match the "
            "validated feature contract"
        )
    num_buckets = int(cluster_contract.selection.num_buckets)
    effective_positive_buckets = int(
        cluster_contract.selection.effective_positive_buckets
    )
    if (
        effective_positive_buckets < 1
        or effective_positive_buckets != num_buckets - 1
    ):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered checkpoint bucket contract must "
            "reserve exactly the final bucket for inactive features; "
            f"num_buckets={num_buckets}, "
            f"effective_positive_buckets={effective_positive_buckets}"
        )
    if signal_budget.negative_bucket != num_buckets - 1 or not (
        0 <= signal_budget.positive_bucket < effective_positive_buckets
    ):
        raise AxBenchBpbAdvError(
            "roofline general_val: resolved signal-budget buckets disagree with "
            "the clustered checkpoint bucket contract"
        )

    checkpoint_vocab = cluster_contract.checkpoint.feature_vocab
    if checkpoint_vocab is None:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output-signal derivation requires "
            "the checkpoint's persistent compacted FeatureVocab"
        )
    expected_allowed_ids = t.tensor(checkpoint_vocab.original_ids, dtype=t.int64)
    if allowed_ids is not None:
        if (
            allowed_ids.ndim != 1
            or allowed_ids.dtype is t.bool
            or allowed_ids.is_floating_point()
            or allowed_ids.is_complex()
        ):
            raise AxBenchBpbAdvError(
                "roofline general_val: supplied clustered allowed_ids must be a "
                "one-dimensional integer tensor"
            )
        actual_allowed_ids = allowed_ids.detach().to(device="cpu", dtype=t.int64)
        if not t.equal(actual_allowed_ids, expected_allowed_ids):
            raise AxBenchBpbAdvError(
                "roofline general_val: supplied allowed_ids do not exactly match "
                "the clustered checkpoint FeatureVocab V"
            )
    return table, corpus_signature


def _derive_clustered_real_output_signal(
    *,
    mags: t.Tensor,
    boundaries: t.Tensor,
    percentile_table: t.Tensor,
    extraction_threshold: float,
    signal_budget: GeneralValSignalBudget,
    allowed_ids: t.Tensor | None,
    cluster_contract: ClusterFeatureContract,
    corpus_signature: str | None,
    sample_id: str,
) -> FeatureConditioning:
    """Materialize one versioned clustered per-document roofline signal."""

    table, validated_corpus_signature = _validate_clustered_signal_inputs(
        mags=mags,
        percentile_table=percentile_table,
        extraction_threshold=extraction_threshold,
        signal_budget=signal_budget,
        allowed_ids=allowed_ids,
        cluster_contract=cluster_contract,
        corpus_signature=corpus_signature,
        sample_id=sample_id,
    )
    consolidation = cluster_contract.consolidation
    all_fids = t.arange(cluster_contract.n_sae_features, dtype=t.int64)
    raw_presence = mags > 0
    raw_active_ids = all_fids[raw_presence]
    raw_active_mags = mags[raw_presence]
    dropped_active_ids = t.empty(0, dtype=t.int64)

    # Mirror the clustered context-cache path: score only active (fid, mag)
    # pairs. A masked active feature whose statistics row is all zero cannot
    # affect the final selection, so drop it before scoring and log the
    # independently configured corpus mismatch. An allowed stat-less feature
    # remains fatal because it could affect the selected signal.
    if raw_active_ids.numel() > 0:
        stat_less = (table[raw_active_ids] == 0).all(dim=1)
        masked_out = ~cluster_contract.feature_mask.keep_mask(raw_active_ids)
        # Both conjuncts are load-bearing: masked stat-less features may be
        # dropped, while allowed stat-less features must remain fatal; keeping
        # the latter also preserves the representative/presence equality bind.
        drop = stat_less & masked_out
        if bool(drop.any()):
            dropped_active_ids = raw_active_ids[drop]
            logger.error(
                "roofline general_val: masked feature(s) "
                f"{dropped_active_ids.tolist()} fired in the eval extraction corpus "
                "but have no percentile statistics (an all-zero percentile-table "
                "row); excluding them before clustered percentile scoring since "
                "they are masked out regardless -- the extraction and statistics "
                "corpora are independently configured, so this is a legitimate "
                "(if noteworthy) mismatch, not a fatal error"
            )
            keep = ~drop
            raw_active_ids = raw_active_ids[keep]
            raw_active_mags = raw_active_mags[keep]

    if consolidation.is_identity_after_mask:
        # Schema-v1 groups cannot literally be singletons. The supported
        # identity-equivalent case is a cluster file whose groups each retain
        # at most one mask-allowed member. Once the full clustered contract is
        # validated, enter the exact legacy signal body so enabling a
        # numerically inert partition cannot change the selected bag through
        # percentile quantization or the cluster-only negative sampler.
        checkpoint_vocab = cluster_contract.checkpoint.feature_vocab
        if checkpoint_vocab is None:  # validated above; retain local type narrowing
            raise AxBenchBpbAdvError(
                "roofline general_val: validated identity-equivalent clustered "
                "signal unexpectedly lacks FeatureVocab"
            )
        # A dropped ID is mask-disallowed, and the legacy selector below is
        # restricted to checkpoint_vocab.original_ids, so zeroing it cannot
        # change the selected bag. Zero a clone only to keep the legacy
        # mag>0 predicate from querying its missing statistics row.
        identity_mags = mags
        if dropped_active_ids.numel() > 0:
            identity_mags = mags.clone()
            identity_mags[dropped_active_ids] = 0
        try:
            signal = _derive_legacy_real_output_signal(
                mags=identity_mags,
                boundaries=boundaries,
                percentile_table=table,
                extraction_threshold=extraction_threshold,
                signal_budget=signal_budget,
                allowed_ids=t.tensor(
                    checkpoint_vocab.original_ids,
                    dtype=t.int64,
                ),
                sample_id=sample_id,
            )
        except ValueError as exc:
            raise AxBenchBpbAdvError(
                "roofline general_val: clustered active output feature lacks "
                f"percentile statistics: {exc}"
            ) from exc
        feature_ids = t.tensor(
            [feature_id for feature_id, _bucket in signal.entries],
            dtype=t.int64,
        )
        try:
            consolidation.assert_representatives_only(
                feature_ids,
                surface="identity-equivalent clustered roofline general-val signal",
            )
            cluster_contract.feature_mask.assert_no_masked(feature_ids)
        except (FeatureClustersError, FeatureMaskError) as exc:
            raise AxBenchBpbAdvError(
                "roofline general_val: identity-equivalent clustered output-signal "
                f"invariant failed: {exc}"
            ) from exc
        return signal

    try:
        percentile_ranks = magnitude_to_percentile(
            raw_active_ids,
            raw_active_mags,
            table,
        )
    except ValueError as exc:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered active output feature lacks "
            f"percentile statistics: {exc}"
        ) from exc
    raw_active_percentiles = uint8_to_percentile(
        percentile_to_uint8(percentile_ranks)
    )
    try:
        masked_ids, masked_percentiles = cluster_contract.feature_mask.apply_to_positive(
            raw_active_ids,
            raw_active_percentiles,
        )
        representative_ids, representative_percentiles = (
            consolidation.consolidate_segment(masked_ids, masked_percentiles)
        )
        projected_presence = consolidation.project_allowed_presence(raw_presence)
    except (FeatureClustersError, FeatureMaskError) as exc:
        raise AxBenchBpbAdvError(
            f"roofline general_val: clustered output-signal projection failed: {exc}"
        ) from exc
    present_representatives = t.nonzero(
        projected_presence,
        as_tuple=False,
    ).squeeze(1).to(t.int64)
    if not t.equal(representative_ids, present_representatives):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered active-percentile projection and "
            "projected raw presence disagree"
        )

    positive_order = sorted(
        range(int(representative_ids.numel())),
        key=lambda idx: (
            -float(representative_percentiles[idx].item()),
            int(representative_ids[idx].item()),
        ),
    )[: signal_budget.k_pos]
    positive_index = t.tensor(positive_order, dtype=t.int64)
    positive_ids = representative_ids[positive_index]
    positive_percentiles = representative_percentiles[positive_index]
    positive_buckets = percentile_to_bucket(
        positive_percentiles,
        int(cluster_contract.selection.effective_positive_buckets),
    )

    try:
        raw_negative_config = json.loads(
            cluster_contract.selection.negative_selection_json
        )
        if not isinstance(raw_negative_config, dict):
            raise ValueError("negative selection is not a JSON object")
        negative_config = parse_negative_selection(raw_negative_config)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered checkpoint negative-selection "
            f"contract is invalid: {exc}"
        ) from exc
    if negative_config is None:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered checkpoint negative selection is disabled"
        )
    # k_neg is the resolved per-document budget. Fix the sampler's draw to that
    # count; its existing pool clamp makes the realized count at most k_neg.
    fixed_negative_config = replace(
        negative_config,
        min_k=signal_budget.k_neg,
        max_k=signal_budget.k_neg + 1,
    )
    generator = t.Generator(device="cpu")
    generator.manual_seed(
        _clustered_signal_seed(
            cluster_contract=cluster_contract,
            corpus_signature=validated_corpus_signature,
            sample_id=sample_id,
        )
    )
    sampled_negative = sample_inactive_features_batched(
        [present_representatives],
        cluster_contract.n_sae_features,
        fixed_negative_config,
        generator,
        feature_mask=consolidation.effective_negative_mask(),
    )
    if len(sampled_negative) != 1:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered batched negative sampler returned "
            f"{len(sampled_negative)} rows for one document"
        )
    negative_ids = sampled_negative[0].to(dtype=t.int64, device="cpu")
    negative_bucket = int(cluster_contract.selection.num_buckets) - 1
    negative_buckets = t.full_like(negative_ids, negative_bucket)

    feature_ids = t.cat((positive_ids, negative_ids))
    buckets = t.cat((positive_buckets, negative_buckets))
    try:
        consolidation.assert_representatives_only(
            feature_ids,
            surface="clustered roofline general-val signal",
        )
        cluster_contract.feature_mask.assert_no_masked(feature_ids)
    except (FeatureClustersError, FeatureMaskError) as exc:
        raise AxBenchBpbAdvError(
            f"roofline general_val: clustered output-signal invariant failed: {exc}"
        ) from exc
    if int(t.unique(feature_ids).numel()) != int(feature_ids.numel()):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered output signal contains duplicate "
            "representatives"
        )
    if negative_ids.numel() > signal_budget.k_neg:
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered negative sampler exceeded k_neg"
        )
    if negative_ids.numel() and bool(projected_presence[negative_ids].any()):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered negative sampler selected a "
            "representative with a mask-allowed active raw member"
        )
    effective_positive_buckets = int(
        cluster_contract.selection.effective_positive_buckets
    )
    if positive_buckets.numel() and bool(
        ((positive_buckets < 0) | (positive_buckets >= effective_positive_buckets)).any()
    ):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered positive bucket is outside the "
            "effective positive bucket range"
        )
    if negative_buckets.numel() and not bool(
        (negative_buckets == negative_bucket).all()
    ):
        raise AxBenchBpbAdvError(
            "roofline general_val: clustered negative was not assigned the "
            "reserved final bucket"
        )
    if feature_ids.numel() == 0:
        raise AxBenchBpbAdvError(
            "roofline general_val: sparse real output-window signal for doc "
            f"{sample_id!r} selected zero features under "
            f"k_pos={signal_budget.k_pos} k_neg={signal_budget.k_neg}; refusing "
            "to invent an empty roofline conditioning bag"
        )
    return FeatureConditioning.from_dict(
        {
            int(fid): int(bucket)
            for fid, bucket in zip(
                feature_ids.tolist(),
                buckets.tolist(),
                strict=True,
            )
        }
    )


def derive_real_output_signal(
    doc: Any,
    *,
    base_model: Any,
    base_sae: Any,
    sae_layer: int,
    boundaries: t.Tensor,
    percentile_table: t.Tensor,
    extraction_threshold: float,
    signal_budget: GeneralValSignalBudget,
    tokenizer: Any,
    device: t.device,
    allowed_ids: t.Tensor | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
    corpus_signature: str | None = None,
) -> FeatureConditioning:
    """Extract the doc's real output-window signal as a feature-conditioning bag.

    The SAE forward reuses the DiffMean extraction seam: Gemma-tokenize
    ``prefix_text + decoded_output`` in one pass, prepend one Gemma BOS, and
    max-pool only the output span. The resulting magnitudes are filtered,
    bucketed exactly like context-cache segments, then ranked by the same full
    per-feature percentile table used by fair feature methods and sparsified to
    their positive/negative feature-count budget.

    ``allowed_ids`` restricts that sparsification to a compacted target
    checkpoint's allowed SAE feature vocabulary ``V`` (see
    :func:`steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids`):
    an id outside ``V`` is masked out of the candidate pool in
    :func:`_select_signal_indices` before ranking, so it can never be selected
    however high its percentile. ``None`` (the default, an uncompacted
    checkpoint) reproduces the unrestricted selection byte-for-byte. If ``V``
    excludes every surviving candidate, the existing zero-selected guard below
    raises rather than inventing an empty bag.
    """

    output_text = str(tokenizer.decode(list(doc.output_tokens)))
    acts = _max_pooled_sae_activations(
        [PrefixedCompletion(prefix_text=str(doc.prefix_text), completion_text=output_text)],
        model=base_model,
        sae=base_sae,
        sae_layer=sae_layer,
        device=device,
    )
    if acts.shape[0] != 1:
        raise AxBenchBpbAdvError(
            "roofline general_val: expected one activation row for doc "
            f"{doc.sample_id!r}, got shape {tuple(acts.shape)}"
        )
    mags = acts[0].detach().to("cpu", dtype=t.float32)
    if cluster_contract is not None:
        return _derive_clustered_real_output_signal(
            mags=mags,
            boundaries=boundaries,
            percentile_table=percentile_table,
            extraction_threshold=extraction_threshold,
            signal_budget=signal_budget,
            allowed_ids=allowed_ids,
            cluster_contract=cluster_contract,
            corpus_signature=corpus_signature,
            sample_id=str(doc.sample_id),
        )

    # Exact legacy branch: no cluster validation, quantized percentile
    # roundtrip, projected presence, or private negative-selection RNG is
    # entered.
    return _derive_legacy_real_output_signal(
        mags=mags,
        boundaries=boundaries,
        percentile_table=percentile_table,
        extraction_threshold=extraction_threshold,
        signal_budget=signal_budget,
        allowed_ids=allowed_ids,
        sample_id=doc.sample_id,
    )


def _derive_legacy_real_output_signal(
    *,
    mags: t.Tensor,
    boundaries: t.Tensor,
    percentile_table: t.Tensor,
    extraction_threshold: float,
    signal_budget: GeneralValSignalBudget,
    allowed_ids: t.Tensor | None,
    sample_id: object,
) -> FeatureConditioning:
    """Execute the pre-cluster real-output signal algorithm exactly."""

    all_fids = t.arange(mags.numel(), dtype=t.int64)
    try:
        feature_ids, bucket_values = threshold_and_discretize_features(
            all_fids,
            mags,
            boundaries=boundaries.to("cpu"),
            extraction_threshold=extraction_threshold,
        )
    except AxBenchBpbError as exc:
        raise AxBenchBpbAdvError(
            "roofline general_val: real output-window signal for doc "
            f"{sample_id!r} could not be thresholded/discretized: {exc}; "
            "refusing to invent an empty roofline conditioning bag"
        ) from exc
    kept_fids = t.tensor(feature_ids, dtype=t.int64)
    kept_buckets = t.tensor(bucket_values, dtype=t.int64)
    kept_mags = mags[kept_fids]
    percentiles = magnitude_to_percentile(
        kept_fids,
        kept_mags,
        percentile_table.to("cpu"),
    )
    selected_indices = _select_signal_indices(
        kept_fids,
        kept_buckets,
        percentiles,
        signal_budget=signal_budget,
        allowed_ids=allowed_ids,
    )
    if not selected_indices:
        raise AxBenchBpbAdvError(
            "roofline general_val: sparse real output-window signal for doc "
            f"{sample_id!r} selected zero features under "
            f"k_pos={signal_budget.k_pos} k_neg={signal_budget.k_neg}; refusing "
            "to invent an empty roofline conditioning bag"
        )
    selected = t.tensor(selected_indices, dtype=t.long)
    selected_fids = kept_fids[selected]
    selected_buckets = kept_buckets[selected]
    if allowed_ids is not None:
        # Fail-loud belt-and-suspenders: the candidate-pool masking in
        # _select_signal_indices should make this unreachable. Reuses Task 1's
        # restrict_to_vocab (the same kept/dropped split used at the
        # model-load boundary) rather than a bespoke check.
        _kept, dropped = restrict_to_vocab(selected_fids, allowed_ids)
        if dropped.numel() > 0:
            raise AxBenchBpbAdvError(
                "roofline general_val: internal invariant violated -- selected "
                f"feature id(s) {[int(x) for x in dropped.tolist()]} fall "
                f"outside the allowed vocabulary V for doc {sample_id!r}; "
                "this should be unreachable after masking non-V ids out of the "
                "candidate selection."
            )
    return FeatureConditioning.from_dict(
        {
            int(fid): int(bucket)
            for fid, bucket in zip(
                selected_fids.tolist(),
                selected_buckets.tolist(),
                strict=True,
            )
        }
    )


def _select_signal_indices(
    fids: t.Tensor,
    buckets: t.Tensor,
    percentiles: t.Tensor,
    *,
    signal_budget: GeneralValSignalBudget,
    allowed_ids: t.Tensor | None = None,
) -> list[int]:
    """Return deterministic sparse indices for the positive/negative partitions.

    ``allowed_ids`` (``V``) restricts candidates in BOTH partitions to ids in
    ``V``: an id outside ``V`` is excluded from ranking entirely (masked out of
    the candidate pool in :func:`_top_signal_indices_for_bucket`), so it can
    never be selected however high its percentile. ``None`` reproduces the
    unrestricted (current, full-``n_sae``) selection byte-for-byte.
    """

    selected = [
        *_top_signal_indices_for_bucket(
            fids,
            buckets,
            percentiles,
            bucket=signal_budget.positive_bucket,
            limit=signal_budget.k_pos,
            allowed_ids=allowed_ids,
        ),
        *_top_signal_indices_for_bucket(
            fids,
            buckets,
            percentiles,
            bucket=signal_budget.negative_bucket,
            limit=signal_budget.k_neg,
            allowed_ids=allowed_ids,
        ),
    ]
    return sorted(selected)


def _top_signal_indices_for_bucket(
    fids: t.Tensor,
    buckets: t.Tensor,
    percentiles: t.Tensor,
    *,
    bucket: int,
    limit: int,
    allowed_ids: t.Tensor | None = None,
) -> list[int]:
    """Return the top percentile-ranked signal indices for one bucket.

    ``allowed_ids`` (``V``), when given, additionally restricts candidates to
    ids in ``V`` -- an id outside ``V`` is never a candidate in the first
    place, so it can never be picked no matter how high its percentile.
    ``None`` is a no-op (unrestricted, current behavior). If ``V`` has fewer
    in-bucket members than ``limit``, every available in-``V`` candidate is
    returned with no padding and no crash.
    """

    if fids.ndim != 1 or buckets.ndim != 1 or percentiles.ndim != 1:
        raise AxBenchBpbAdvError(
            "roofline general_val: signal selection tensors must be one-dimensional "
            f"(fids={tuple(fids.shape)}, buckets={tuple(buckets.shape)}, "
            f"percentiles={tuple(percentiles.shape)})"
        )
    if fids.numel() != buckets.numel() or fids.numel() != percentiles.numel():
        raise AxBenchBpbAdvError(
            "roofline general_val: signal selection tensors must have equal "
            f"lengths (fids={fids.numel()}, buckets={buckets.numel()}, "
            f"percentiles={percentiles.numel()})"
        )
    if limit <= 0:
        return []
    allowed_set: set[int] | None = (
        None if allowed_ids is None else {int(fid) for fid in allowed_ids.tolist()}
    )
    candidates = [
        idx
        for idx, value in enumerate(buckets.tolist())
        if int(value) == int(bucket)
        and (allowed_set is None or int(fids[idx].item()) in allowed_set)
    ]
    candidates.sort(
        key=lambda idx: (
            -float(percentiles[idx].item()),
            int(fids[idx].item()),
        )
    )
    return candidates[:limit]


def build_general_val_doc_member(
    doc: Any,
    *,
    segments: tuple[ContextFeatureSegment, ...],
    tokenizer: Any,
    output_bag: FeatureConditioning,
    max_seq_len: int,
    device: t.device,
    allowed_ids: t.Tensor | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureGradMember:
    """Build the one-document LM-loss member under the score-time encdec layout.

    ``allowed_ids`` restricts the FIXED per-segment context bags (built from
    *segments*, the context cache's pre-extracted, threshold-filtered raw SAE
    activations -- NOT pre-selected against a compacted checkpoint's vocabulary
    ``V`` the way *output_bag* already is) to ``V``, mirroring
    ``steering.eval.axbench_bpb.teacher_force._build_encdec_seg_map_and_encoder``'s
    context-cache filter (``teacher_force.py:306-340``) via
    ``vocab_restriction.restrict_segment_conditions_to_vocab``: an id outside
    ``V`` is dropped, with ONE aggregated ``logger.warning`` naming the total
    dropped count across every segment (not one per segment), and a segment
    fully emptied by the filter raises ``AxBenchBpbError``. *output_bag* (the
    generation-slot bag) is NEVER restricted here -- it is the method's own
    bag, already restricted upstream of wherever it is threaded in. ``None``
    (the default; an uncompacted checkpoint) is a strict no-op, byte-identical
    to before this parameter existed. See
    ``steering.eval.axbench_bpb.vocab_restriction.feature_vocab_allowed_ids``.
    """

    if not segments:
        raise AxBenchBpbAdvError(
            "roofline general_val: doc "
            f"{doc.sample_id!r} has no context feature segments; every roofline "
            "general-val doc must condition on the same prefix segments as scoring"
        )
    if cluster_contract is not None:
        if allowed_ids is None:
            raise AxBenchBpbAdvError(
                "roofline general_val clustered member construction requires "
                "the exact checkpoint feature vocabulary"
            )
        contract_vocab = cluster_contract.checkpoint.feature_vocab
        resolved_allowed = tuple(sorted(int(value) for value in allowed_ids.tolist()))
        if contract_vocab is None or resolved_allowed != contract_vocab.original_ids:
            raise AxBenchBpbAdvError(
                "roofline general_val clustered member checkpoint vocabulary "
                "does not match the validated feature contract"
            )
        for segment_index, segment in enumerate(segments):
            segment_ids = t.tensor(segment.feature_ids, dtype=t.int64)
            try:
                cluster_contract.consolidation.assert_representatives_only(
                    segment_ids,
                    surface=(
                        "roofline general_val fixed context segment "
                        f"{segment_index}"
                    ),
                )
                cluster_contract.feature_mask.assert_no_masked(segment_ids)
            except (FeatureClustersError, FeatureMaskError) as exc:
                raise AxBenchBpbAdvError(
                    "roofline general_val fixed context must already be exact "
                    f"representative-only clustered cache content: {exc}"
                ) from exc
        output_ids = t.tensor(list(output_bag.feature_ids()), dtype=t.int64)
        try:
            cluster_contract.consolidation.assert_representatives_only(
                output_ids,
                surface="roofline general_val output bag",
            )
            cluster_contract.feature_mask.assert_no_masked(output_ids)
        except (FeatureClustersError, FeatureMaskError) as exc:
            raise AxBenchBpbAdvError(
                f"roofline general_val output bag violates cluster contract: {exc}"
            ) from exc
    prefix_tokens = [int(token_id) for token_id in doc.prefix_tokens]
    output_tokens = [int(token_id) for token_id in doc.output_tokens]
    n_prefix = len(prefix_tokens)
    n_output = len(output_tokens)
    if n_prefix + n_output > int(max_seq_len):
        raise AxBenchBpbAdvError(
            "roofline general_val: doc "
            f"{doc.sample_id!r} has prefix+output length {n_prefix}+{n_output}="
            f"{n_prefix + n_output}, exceeding encdec max_seq_len={int(max_seq_len)}. "
            "The optimizer does not mirror score-time left truncation, so this "
            "would optimize a different layout than the BPB scorer."
        )
    all_tokens = prefix_tokens + output_tokens
    dec_input = t.tensor([all_tokens[:-1]], dtype=t.long, device=device)
    dec_target = t.tensor([all_tokens[1:]], dtype=t.long, device=device)
    if n_prefix > 1:
        dec_target[0, : n_prefix - 1] = -1

    generation_slot = len(segments)
    full_seg_map = build_multi_segment_seg_map(
        tokenizer=tokenizer,
        prefix_tokens=prefix_tokens,
        prefix_segments=segments,
        n_output=n_output,
        generation_slot=generation_slot,
    )
    seg_map = full_seg_map[:, :-1].to(device)
    expected_width = len(all_tokens) - 1
    if seg_map.shape[1] != expected_width:
        raise AxBenchBpbAdvError(
            "roofline general_val: doc "
            f"{doc.sample_id!r} seg_map width {seg_map.shape[1]} != shifted token "
            f"width {expected_width}; the optimizer and scorer layouts would diverge"
        )

    raw_segment_dicts = [_segment_condition_dict(seg) for seg in segments]
    restricted_segment_dicts, total_dropped = restrict_segment_conditions_to_vocab(
        raw_segment_dicts, allowed_ids,
    )
    if cluster_contract is not None and total_dropped != 0:
        raise AxBenchBpbAdvError(
            "roofline general_val clustered fixed context required read-time "
            f"filtering of {total_dropped} IDs; the cache is stale or corrupt"
        )
    if total_dropped > 0:
        logger.warning(
            "roofline general_val: doc "
            f"{doc.sample_id!r} dropped {total_dropped} context feature id(s) "
            f"outside the checkpoint's compacted vocabulary V, across "
            f"{len(segments)} segment(s)."
        )
    segment_conditions = [
        *[FeatureConditioning.from_dict(bag) for bag in restricted_segment_dicts],
        output_bag,
    ]
    return FeatureGradMember(
        dec_input=dec_input,
        dec_target=dec_target,
        segment_conditions=segment_conditions,
        seg_map=seg_map,
        generation_slot=generation_slot,
    )


def _lm_objective_factory(
    *, device: t.device, max_forward_batch: int, max_grad_batch: int
) -> Callable[[Any], Any]:
    def factory(model: Any) -> Any:
        return lm_loss_on_masked_tokens(
            model,
            device=device,
            max_forward_batch=max_forward_batch,
            max_grad_batch=max_grad_batch,
        )

    return factory


def _warm_start_partition_counts(
    warm: FeatureConditioning, bounds: PartitionBounds
) -> tuple[int, int]:
    if not bounds.has_negative:
        return len(warm), 0
    n_pos = sum(
        1
        for _fid, bucket in warm.entries
        if is_active(int(bucket), int(bounds.num_buckets))
    )
    return n_pos, len(warm) - n_pos


def general_val_warm_start_partition_caps(
    model: Any,
    warm: FeatureConditioning,
    base_config: FeatureGcgConfig,
    *,
    cluster_contract: ClusterFeatureContract | None = None,
) -> GeneralValWarmStartCaps:
    """Return exclusive GCG partition caps large enough for a real-signal warm bag.

    Concept-path feature-GCG warm starts are top-k bags and naturally fit the
    package defaults. Roofline general-val warm starts are also sparse, but the
    user may override the feature budget above the package defaults. The
    optimizer still needs a valid starting point; the cap identity is derived
    from the warm bag itself and persisted with optimized per-doc bags.
    """

    bounds = (
        partition_bounds_from_model(model, base_config)
        if cluster_contract is None
        else partition_bounds_from_model(
            model,
            base_config,
            cluster_contract=cluster_contract,
        )
    )
    n_pos, n_neg = _warm_start_partition_counts(warm, bounds)
    return GeneralValWarmStartCaps(
        max_k_pos=max(int(base_config.max_k_pos), n_pos + 1),
        max_k_neg=max(int(base_config.max_k_neg), n_neg + 1),
    )


def general_val_doc_gcg_config(
    model: Any,
    warm: FeatureConditioning,
    *,
    reg_lambda: float,
    ours: OursGcgConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureGcgConfig:
    base_config = FeatureGcgConfig(
        reg_lambda=float(reg_lambda),
        max_steps=ours.max_steps,
        minibatch_size=ours.minibatch_size,
        master_seed=ours.master_seed,
        max_forward_batch=ours.max_forward_batch,
        max_grad_batch=ours.max_grad_batch,
    )
    warm_start_caps = (
        general_val_warm_start_partition_caps(model, warm, base_config)
        if cluster_contract is None
        else general_val_warm_start_partition_caps(
            model,
            warm,
            base_config,
            cluster_contract=cluster_contract,
        )
    )
    if (
        warm_start_caps.max_k_pos == int(base_config.max_k_pos)
        and warm_start_caps.max_k_neg == int(base_config.max_k_neg)
    ):
        return base_config
    return FeatureGcgConfig(
        reg_lambda=base_config.reg_lambda,
        max_steps=base_config.max_steps,
        candidate_batch_size=base_config.candidate_batch_size,
        top_k=base_config.top_k,
        temperature=base_config.temperature,
        master_seed=base_config.master_seed,
        minibatch_size=base_config.minibatch_size,
        max_forward_batch=base_config.max_forward_batch,
        max_grad_batch=base_config.max_grad_batch,
        min_k_pos=base_config.min_k_pos,
        max_k_pos=warm_start_caps.max_k_pos,
        min_k_neg=base_config.min_k_neg,
        max_k_neg=warm_start_caps.max_k_neg,
    )


def penalized_lm_objective_value(
    model: Any,
    member: FeatureGradMember,
    bag: FeatureConditioning,
    *,
    reg_lambda: float,
    ours: OursGcgConfig,
    device: t.device,
    cluster_contract: ClusterFeatureContract | None = None,
) -> float:
    """Return token-mean LM loss plus the feature-count penalty for metadata."""

    # #720: NO compaction guard here. The ROOFLINE_GCG general-val derive now restricts the
    # doc's output-window signal bag to the checkpoint's vocabulary V before this forward-only
    # LM-loss measurement conditions the encoder on it, so `_to_compact` sees only ids in V —
    # no OOV. On an uncompacted encoder this is unchanged (V is the full range).
    if cluster_contract is not None:
        for condition_index, conditioning in enumerate(member.segment_conditions):
            ids = t.tensor(list(conditioning.feature_ids()), dtype=t.int64)
            try:
                cluster_contract.consolidation.assert_representatives_only(
                    ids,
                    surface=(
                        "roofline general_val objective segment conditioning "
                        f"{condition_index}"
                    ),
                )
                cluster_contract.feature_mask.assert_no_masked(ids)
            except (FeatureClustersError, FeatureMaskError) as exc:
                raise AxBenchBpbAdvError(
                    "roofline general_val objective reached encoder packing with "
                    f"invalid clustered conditioning: {exc}"
                ) from exc
        bag_ids = t.tensor(list(bag.feature_ids()), dtype=t.int64)
        try:
            cluster_contract.consolidation.assert_representatives_only(
                bag_ids,
                surface="roofline general_val objective bag",
            )
            cluster_contract.feature_mask.assert_no_masked(bag_ids)
        except (FeatureClustersError, FeatureMaskError) as exc:
            raise AxBenchBpbAdvError(
                "roofline general_val objective reached encoder packing with "
                f"invalid clustered conditioning: {exc}"
            ) from exc
    objective = lm_loss_on_masked_tokens(
        model,
        device=device,
        max_forward_batch=ours.max_forward_batch,
    )
    [loss] = objective.losses([FeatureLmItem(member=member, bag=bag)])
    return float(loss) + float(reg_lambda) * len(bag)


def optimize_general_val_doc_bag(
    model: Any,
    member: FeatureGradMember,
    *,
    warm: FeatureConditioning,
    reg_lambda: float,
    ours: OursGcgConfig,
    device: t.device,
    cluster_contract: ClusterFeatureContract | None = None,
) -> FeatureConditioning:
    """Optimize one validation document's generation-slot bag with feature-GCG."""

    gcg_cfg = (
        general_val_doc_gcg_config(
            model,
            warm,
            reg_lambda=float(reg_lambda),
            ours=ours,
        )
        if cluster_contract is None
        else general_val_doc_gcg_config(
            model,
            warm,
            reg_lambda=float(reg_lambda),
            ours=ours,
            cluster_contract=cluster_contract,
        )
    )
    optimize_kwargs = (
        {} if cluster_contract is None else {"cluster_contract": cluster_contract}
    )
    return optimize_feature_conditioning_universal(
        model,
        [member],
        warm_start=warm,
        objective_factory=_lm_objective_factory(
            device=device,
            max_forward_batch=ours.max_forward_batch,
            max_grad_batch=ours.max_grad_batch,
        ),
        config=gcg_cfg,
        **optimize_kwargs,
    )


def validate_general_val_bag_metadata(
    *,
    metadata: Mapping[str, str],
    sample_id: str,
    family: str,
    reg_lambda: float,
    corpus_signature: str,
    ours: OursGcgConfig,
    signal_budget: GeneralValSignalBudget,
    config_metadata: Mapping[str, str] | None = None,
    warm_start_caps: GeneralValWarmStartCaps | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    """Fail loudly if a persisted roofline general-val bag is not the requested one.

    Validates the FULL identity the bag was derived/optimized under, not just
    the path components (sample_id, family, reg_lambda). A cached bag whose
    validation corpus, GCG hyperparameters, extraction settings, SAE identity,
    statistics source, or bucket boundaries differ from the current run belongs
    to a DIFFERENT problem; reusing it would ship a wrong steerability-frontier
    x with no error. Both the write site and this validator must list the same
    fields, so drift is caught on resume rather than silently reused.
    """

    expected = general_val_bag_identity_metadata(
        sample_id=sample_id,
        family=family,
        reg_lambda=reg_lambda,
        corpus_signature=corpus_signature,
        ours=ours,
        signal_budget=signal_budget,
        config_metadata=config_metadata,
        warm_start_caps=warm_start_caps,
        cluster_contract=cluster_contract,
    )
    for key, expected_value in expected.items():
        actual = metadata.get(key)
        if actual != expected_value:
            raise AxBenchBpbAdvError(
                "roofline general_val: persisted bag metadata mismatch for "
                f"{key!r}: expected {expected_value!r}, got {actual!r} "
                f"(sample_id={sample_id!r}, family={family!r}, reg_lambda={reg_lambda!r}). "
                "A cached bag optimized under a different corpus/config must not be "
                "reused; clear general_val_bags/ or score under a fresh run_tag."
            )


def general_val_bag_identity_metadata(
    *,
    sample_id: str,
    family: str,
    reg_lambda: float,
    corpus_signature: str,
    ours: OursGcgConfig,
    signal_budget: GeneralValSignalBudget,
    config_metadata: Mapping[str, str] | None = None,
    warm_start_caps: GeneralValWarmStartCaps | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> dict[str, str]:
    """Return the persisted-bag identity shared by writers and validators."""

    expected = {
        "sample_id": str(sample_id),
        "mode_tag": str(family),
        "reg_lambda": str(float(reg_lambda)),
        "general_val_corpus": str(corpus_signature),
        "max_steps": str(int(ours.max_steps)),
        "minibatch_size": str(int(ours.minibatch_size)),
        "master_seed": str(int(ours.master_seed)),
        "max_forward_batch": str(int(ours.max_forward_batch)),
    }
    if config_metadata is not None:
        expected.update(dict(config_metadata))
    expected.update(signal_budget.metadata(cluster_contract=cluster_contract))
    if warm_start_caps is not None:
        expected["max_k_pos"] = str(int(warm_start_caps.max_k_pos))
        expected["max_k_neg"] = str(int(warm_start_caps.max_k_neg))
    return expected


__all__ = [
    "GENERAL_VAL_SIGNAL_POLICY",
    "GeneralValSignalBudget",
    "GeneralValWarmStartCaps",
    "build_general_val_doc_member",
    "derive_real_output_signal",
    "general_val_bag_identity_metadata",
    "general_val_doc_gcg_config",
    "general_val_warm_start_partition_caps",
    "optimize_general_val_doc_bag",
    "penalized_lm_objective_value",
    "validate_general_val_bag_metadata",
]
