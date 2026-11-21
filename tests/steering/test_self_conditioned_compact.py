"""Compact feature-prediction tests for self-conditioned generation.

The prediction head of a compact checkpoint emits rows in compact-vocabulary
order, while every public feature id and every encoder input remains an
original SAE id. These tests make row index and original id deliberately
different, then compare the uncompacted path against the pre-change algorithm
exactly across every inference pool mode and both selection families.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import nanochat.flash_attention as _fa_mod
from errors import FeatureVocabError
from nanochat.checkpoint_manager import ModelPhase, build_encdec_model
from nanochat.common import get_base_dir
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import Linear
from steering.conditional.discretization import (
    bucket_percentile_key_grid,
    build_boundaries_from_stats,
    discretize_activations,
    percentile_to_bucket,
)
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionConfig,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.feature_prediction import pool_fp_logits
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    append_packed_encoder_segment,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    EmptySelfConditioningSegmentError,
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondState,
)
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    ThresholdSelectionConfig,
    active_bucket_count,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.percentile_table import (
    build_percentile_table,
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import (
    ThresholdConfig,
    ThresholdType,
    apply_threshold,
    build_thresholds_from_stats,
    extract_threshold_percentile,
    feature_percentile_key,
)
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import ContextFeatureSegment, SelfConditionedTarget
from steering.refined_types import PositiveInt
from steering.segmentation import token_to_char_offsets
from steering.stats_types import PerFeatureStat
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import build_tiny_model
from tests.steering._self_conditioned_state import build_test_self_cond_state

N_SAE = 32
ENCODER_BUCKETS = 4
COMPACT_ORIGINAL_IDS = (8, 10, 20, 25, 27, 30)
COMPACT_VOCAB = FeatureVocab(N_SAE, COMPACT_ORIGINAL_IDS)
COMPACT_WIDTH = len(COMPACT_ORIGINAL_IDS)
_CPU = t.device("cpu")


def _assert_packed_equal(
    actual: PackedEncoderOutput,
    expected: PackedEncoderOutput,
) -> None:
    assert t.equal(actual.values, expected.values)
    assert t.equal(actual.layout.flat_indices, expected.layout.flat_indices)
    assert t.equal(actual.layout.valid_mask, expected.layout.valid_mask)
    assert t.equal(actual.layout.batch_indices, expected.layout.batch_indices)
    assert t.equal(actual.layout.segment_indices, expected.layout.segment_indices)
    assert actual.layout.batch_size == expected.layout.batch_size
    assert actual.layout.n_segments == expected.layout.n_segments
    assert actual.layout.feature_count == expected.layout.feature_count


def _segment_values(packed: PackedEncoderOutput, segment_index: int) -> t.Tensor:
    owned = packed.layout.valid_mask & (
        packed.layout.segment_indices == segment_index
    )
    return packed.values[0, owned]


@pytest.fixture(autouse=True)
def _force_sdpa() -> Iterator[None]:
    """The CPU test model uses SDPA rather than flex-attention compilation."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _model(
    feature_vocab: FeatureVocab | None,
    *,
    num_buckets: int = ENCODER_BUCKETS,
):
    t.manual_seed(0)
    return build_tiny_model(
        n_sae_features=N_SAE,
        num_buckets=num_buckets,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.FULL_SEGMENT,
        ),
        feature_vocab=feature_vocab,
    )


def _threshold(
    threshold_type: ThresholdType = ThresholdType.segment_topk,
    threshold: float = 3,
    precomputed: t.Tensor | None = None,
) -> ThresholdSelectionConfig:
    return ThresholdSelectionConfig(
        selection=ThresholdSelection(threshold_type, threshold),
        precomputed_thresholds=precomputed,
    )


def _sampling(
    seed: int,
    *,
    mode: SamplingMode = SamplingMode.weighted,
) -> SamplingBundle:
    rng = t.Generator(device="cpu").manual_seed(seed)
    return SamplingBundle(
        selection=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=COMPACT_WIDTH,
            mode=mode,
        ),
        percentile_table=t.zeros(N_SAE, 1, dtype=t.float64),
        rng=rng,
    )


def _config(
    selection: ThresholdSelectionConfig | SamplingBundle,
    *,
    pool_mode: FeaturePredLossMode = FeaturePredLossMode.LAST_TOKEN,
    n_sae_features: int = N_SAE,
    num_buckets: int = ENCODER_BUCKETS,
    feature_mask: FeatureMask | None = None,
    selection_producer: FeatureSelectionProducer = (
        FeatureSelectionProducer.FULL_PERCENTILE_RANK
    ),
    bucket_producer: FeatureBucketProducer = FeatureBucketProducer.UNIFORM_PERCENTILE,
    cluster_consolidation: FeatureClusterConsolidation | None = None,
    fp_logit_bias: t.Tensor | None = None,
) -> SelfCondConfig:
    return SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(4),
        pool_mode=pool_mode,
        selection=selection,
        selection_producer=selection_producer,
        bucket_producer=bucket_producer,
        num_buckets=PositiveInt.parse(num_buckets),
        n_sae_features=PositiveInt.parse(n_sae_features),
        feature_mask=feature_mask,
        cluster_consolidation=cluster_consolidation,
        fp_logit_bias=fp_logit_bias,
    )


def _consolidation(
    tmp_path: Path,
    clusters: list[list[int]],
    *,
    name: str = "clusters.json",
    mask: FeatureMask | None = None,
    n_sae_features: int = N_SAE,
) -> FeatureClusterConsolidation:
    """Build a real consolidation from a static cluster file, as production does."""
    path = tmp_path / name
    path.write_text(
        json.dumps({"schema_version": 1, "clusters": clusters}),
        encoding="utf-8",
    )
    resolved = resolve_feature_clusters(
        {"feature_clusters": {"path": str(path)}},
        n_sae_features,
    )
    assert resolved is not None
    return FeatureClusterConsolidation.build(resolved, mask)


def _provider(model, cfg: SelfCondConfig) -> SelfConditionedFeatureProvider:
    return SelfConditionedFeatureProvider(model=model, cfg=cfg, device=_CPU)


def _logits_for_percentiles(percentiles: Sequence[float], *, steps: int = 1) -> t.Tensor:
    values = t.tensor(percentiles, dtype=t.float64)
    logits = t.log(values / (1.0 - values)).to(t.float32)
    return logits.view(1, 1, -1).expand(1, steps, -1).contiguous()


def _identity_percentile_stat() -> PerFeatureStat:
    """Statistics whose pXX magnitude is the float32 value XX/100."""
    percentiles = {
        f"p{percentile}": float(t.tensor(percentile / 100, dtype=t.float32))
        for percentile in range(1, 100)
    }
    return PerFeatureStat(
        count=100,
        pct_segments_active=1.0,
        mean_mag=0.5,
        mean_mag_incl_zeros=0.5,
        std_mag=0.25,
        min_mag=0.01,
        max_mag=0.99,
        median_mag=0.5,
        percentiles=percentiles,
    )


def _p33_through_p66_plateau_stat() -> PerFeatureStat:
    """Identity-like statistics with one wide exact-value plateau."""
    base = _identity_percentile_stat()
    percentiles = dict(base.percentiles)
    for percentile in range(33, 67):
        percentiles[f"p{percentile}"] = 0.5
    return PerFeatureStat(
        count=base.count,
        pct_segments_active=base.pct_segments_active,
        mean_mag=base.mean_mag,
        mean_mag_incl_zeros=base.mean_mag_incl_zeros,
        std_mag=base.std_mag,
        min_mag=base.min_mag,
        max_mag=base.max_mag,
        median_mag=base.median_mag,
        percentiles=percentiles,
    )


def test_documented_selfcond_config_uses_the_production_producer_enums() -> None:
    """Keep the load-bearing docs code block aligned with the import surface."""
    repo_root = Path(__file__).resolve().parents[2]
    doc = (repo_root / "docs/pipelines/self-conditioned-generation.md").read_text()

    assert "FeaturePredTargetCurrency" not in doc
    assert "FeatureSelectionProducer," in doc
    assert "FeatureBucketProducer," in doc
    assert "selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK" in doc
    assert "bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE" in doc
    assert "feature_vocab.to_compact(original_ids)" in doc
    assert "controlled_ids = full_bias.nonzero().flatten()" in doc
    assert "feature_vocab.select_rows(full_bias)" in doc
    assert "pool over decoder-token positions" in doc
    assert "add fp_logit_bias in float32" in doc
    assert "finite float64 value that overflows float32 is" in doc
    assert "no addition, conversion, allocation, version change, or RNG use" in doc

    configuration_doc = (repo_root / "docs/reference/configuration.md").read_text()
    file_reference = (repo_root / "docs/reference/file-reference.md").read_text()
    assert "SelfCondConfig.fp_logit_bias" in configuration_doc
    assert "strict head-row float32 logit bias" in file_reference


@contextmanager
def _capture_segments(
    provider: SelfConditionedFeatureProvider,
) -> Iterator[list[dict[int, int]]]:
    """Capture the original-id/bucket dictionary handed to the encoder."""
    captured: list[dict[int, int]] = []
    original_populated = provider._build_segment_inputs

    def capture_populated(features: dict[int, int]):
        captured.append(dict(features))
        return original_populated(features)

    provider._build_segment_inputs = capture_populated  # ty: ignore[invalid-assignment]
    try:
        yield captured
    finally:
        del provider._build_segment_inputs


@contextmanager
def _capture_selection_inputs(
    provider: SelfConditionedFeatureProvider,
) -> Iterator[list[tuple[t.Tensor, t.Tensor]]]:
    """Capture ids/scores at the common post-sigmoid selection boundary."""
    captured: list[tuple[t.Tensor, t.Tensor]] = []
    original = provider._apply_selection

    def capture(feature_ids: t.Tensor, scores: t.Tensor):
        captured.append((feature_ids.detach().clone(), scores.detach().clone()))
        return original(feature_ids, scores)

    provider._apply_selection = capture  # ty: ignore[invalid-assignment]
    try:
        yield captured
    finally:
        del provider._apply_selection


def _legacy_uncompacted_step(
    provider: SelfConditionedFeatureProvider,
    state: SelfCondState,
    fp_logits_tail: t.Tensor,
) -> tuple[dict[int, int], SelfCondState]:
    """The exact valid-input ``step`` algorithm before compact support."""
    pooled = pool_fp_logits(fp_logits_tail, provider.cfg.pool_mode)
    percentiles = t.sigmoid(pooled.detach().to(t.float32))
    feature_ids = t.arange(
        provider.cfg.n_sae_features,
        dtype=t.int32,
        device=percentiles.device,
    )
    kept_ids, kept_pcts = provider._apply_selection(feature_ids, percentiles[0])
    kept_buckets = percentile_to_bucket(kept_pcts, provider.cfg.num_buckets)
    segment = {
        int(fid): int(bucket)
        for fid, bucket in zip(
            kept_ids.tolist(),
            kept_buckets.tolist(),
            strict=True,
        )
    }
    if not segment:
        raise EmptySelfConditioningSegmentError(
            "legacy oracle selected zero features"
        )
    fids, mags, mask = provider._build_segment_inputs(segment)
    new_out = provider._encode_one_segment(fids, mags, mask)
    merged_out = append_packed_encoder_segment(state.encoder_out, new_out)
    return segment, SelfCondState(
        encoder_out=merged_out,
        target_slot_idx=state.target_slot_idx + 1,
    )


@pytest.mark.parametrize(
    "pool_mode",
    [
        FeaturePredLossMode.LAST_TOKEN,
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.FULL_SEGMENT,
    ],
)
@pytest.mark.parametrize("selection_kind", ["threshold", "sampling"])
def test_normalized_producer_matches_legacy_outputs_and_rng(
    pool_mode: FeaturePredLossMode,
    selection_kind: str,
) -> None:
    """The packed/normalized producer retains the complete legacy algorithm."""
    model = _model(None)
    if selection_kind == "threshold":
        reference_selection = _threshold()
        actual_selection = _threshold()
    else:
        reference_selection = _sampling(91)
        actual_selection = _sampling(91)

    reference = _provider(model, _config(reference_selection, pool_mode=pool_mode))
    actual = _provider(model, _config(actual_selection, pool_mode=pool_mode))
    sampling_rng_before: t.Tensor | None = None
    if selection_kind == "sampling":
        assert isinstance(reference.cfg.selection, SamplingBundle)
        assert isinstance(actual.cfg.selection, SamplingBundle)
        sampling_rng_before = reference.cfg.selection.rng.get_state().clone()
        assert t.equal(
            sampling_rng_before,
            actual.cfg.selection.rng.get_state(),
        )
    reference_state = build_test_self_cond_state(reference, {3: 1})
    actual_state = build_test_self_cond_state(actual, {3: 1})
    _assert_packed_equal(reference_state.encoder_out, actual_state.encoder_out)

    logits_rng = t.Generator().manual_seed(17)
    fp_logits = t.where(
        t.randn(1, 3, N_SAE, generator=logits_rng) >= 0,
        1.0,
        -1.0,
    )
    fixed_percentiles = t.sigmoid(
        pool_fp_logits(fp_logits, pool_mode).detach().to(t.float32)
    )[0]
    legacy_buckets = percentile_to_bucket(fixed_percentiles, ENCODER_BUCKETS)
    actual_buckets = actual._bucket_percentiles_like_training(fixed_percentiles)
    assert t.equal(legacy_buckets, actual_buckets)
    global_before = t.random.get_rng_state().clone()
    expected_segment, expected_state = _legacy_uncompacted_step(
        reference,
        reference_state,
        fp_logits,
    )
    global_after_reference = t.random.get_rng_state().clone()

    t.random.set_rng_state(global_before)
    with _capture_segments(actual) as captured:
        actual_state_after = actual.step(actual_state, fp_logits)
    global_after_actual = t.random.get_rng_state().clone()

    assert captured == [expected_segment]
    _assert_packed_equal(actual_state_after.encoder_out, expected_state.encoder_out)
    assert actual_state_after.target_slot_idx == expected_state.target_slot_idx
    assert t.equal(global_after_actual, global_after_reference)
    assert t.equal(global_after_actual, global_before)
    if selection_kind == "sampling":
        assert isinstance(reference.cfg.selection, SamplingBundle)
        assert isinstance(actual.cfg.selection, SamplingBundle)
        reference_rng_after = reference.cfg.selection.rng.get_state()
        actual_rng_after = actual.cfg.selection.rng.get_state()
        assert t.equal(reference_rng_after, actual_rng_after)
        assert sampling_rng_before is not None
        assert not t.equal(reference_rng_after, sampling_rng_before), (
            "the sampling parity check is vacuous unless the sampler consumes RNG"
        )


@pytest.mark.parametrize("num_buckets", [1, 3, 4, 10])
def test_normalized_producer_bucket_assignment_is_exact_legacy_parity(
    num_buckets: int,
) -> None:
    """Pin packed selection-shard bucketing, including every exact boundary."""
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=max(2, num_buckets)),
        _config(
            _threshold(threshold=COMPACT_WIDTH),
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        ),
    )
    scores = t.tensor(
        [0.0, 0.1, 0.25, 1 / 3, 0.5, 2 / 3, 0.75, 0.95, 1.0],
        dtype=t.float32,
    )

    assert t.equal(
        provider._bucket_percentiles_like_training(scores),
        percentile_to_bucket(scores, num_buckets),
    )


def test_compact_argmax_uses_original_id_active_bucket_and_real_encoder() -> None:
    """One case falsifies compact-row, full-bucket, and fake-encoder shortcuts."""
    model = _model(COMPACT_VOCAB)
    active_buckets = active_bucket_count(
        ENCODER_BUCKETS,
        reserves_negative_bucket=True,
    )
    provider = _provider(
        model,
        _config(_threshold(threshold=1), num_buckets=active_buckets),
    )
    state = build_test_self_cond_state(provider, {COMPACT_ORIGINAL_IDS[0]: 1})
    percentiles = [0.10, 0.20, 0.95, 0.30, 0.70, 0.40]
    with _capture_segments(provider) as captured:
        result = provider.step(state, _logits_for_percentiles(percentiles))

    selected_id = COMPACT_ORIGINAL_IDS[2]
    selected_bucket = active_buckets - 1
    assert selected_bucket == active_buckets - 1
    assert captured == [{selected_id: selected_bucket}]
    assert 2 not in captured[0], "compact row 2 is not original SAE id 20"
    assert selected_bucket != ENCODER_BUCKETS - 1, (
        "predicted-active features must not enter the reserved inactive bucket"
    )

    fids, mags, mask = provider._build_segment_inputs(captured[0])
    assert fids[0, 0, 0].item() == selected_id
    expected_out = model.encoder(fids, mags, mask)
    assert t.equal(
        _segment_values(result.encoder_out, result.encoder_out.layout.n_segments - 1),
        _segment_values(expected_out, 0),
    )


def test_threshold_shifted_p95_bucket_matches_training_boundary_search() -> None:
    """p95 after a p90 filter is bucket 5, not uniform-grid bucket 9."""
    target_row = 2
    target_id = COMPACT_ORIGINAL_IDS[target_row]
    num_buckets = 10
    threshold = 0.9
    boundaries = build_boundaries_from_stats(
        {str(target_id): _identity_percentile_stat()},
        n_sae_features=N_SAE,
        num_buckets=num_buckets,
        threshold_percentile=threshold,
    )
    training_bucket = int(
        discretize_activations(
            t.tensor([target_id], dtype=t.int32),
            t.tensor([0.95], dtype=t.float32),
            boundaries,
        )[0]
    )
    assert training_bucket == 5
    legacy_uniform_bucket = int(percentile_to_bucket(t.tensor([0.95]), num_buckets)[0])
    assert legacy_uniform_bucket == 9
    assert legacy_uniform_bucket != training_bucket

    precomputed = t.full((N_SAE,), threshold, dtype=t.float64)
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=num_buckets),
        _config(
            _threshold(
                ThresholdType.feature_percentile,
                threshold=threshold,
                precomputed=precomputed,
            ),
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.RAW_MAGNITUDE,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    percentiles = [0.10] * COMPACT_WIDTH
    percentiles[target_row] = 0.95
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(state, _logits_for_percentiles(percentiles))

    assert captured == [{target_id: training_bucket}]


def test_unshifted_three_bucket_grid_matches_training_p66_boundary() -> None:
    """The historical B=3 grid is p33/p66, not idealized exact thirds."""
    target_row = 2
    target_id = COMPACT_ORIGINAL_IDS[target_row]
    num_buckets = 3
    boundaries = build_boundaries_from_stats(
        {str(target_id): _identity_percentile_stat()},
        n_sae_features=N_SAE,
        num_buckets=num_buckets,
    )
    training_bucket = int(
        discretize_activations(
            t.tensor([target_id], dtype=t.int32),
            t.tensor([0.66], dtype=t.float32),
            boundaries,
        )[0]
    )
    assert training_bucket == 2

    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=num_buckets),
        _config(
            _threshold(threshold=1),
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    percentiles = [0.10] * COMPACT_WIDTH
    percentiles[target_row] = 0.66
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(state, _logits_for_percentiles(percentiles))

    assert captured == [{target_id: training_bucket}]


@pytest.mark.parametrize(
    ("num_buckets", "threshold_percentile", "key", "higher_bucket"),
    [
        (10, 0.9, 95, 5),
        (3, None, 66, 2),
    ],
)
def test_continuous_cutpoint_uses_training_right_boundary_semantics(
    num_buckets: int,
    threshold_percentile: float | None,
    key: int,
    higher_bucket: int,
) -> None:
    if threshold_percentile is None:
        selection = _threshold(threshold=1)
    else:
        selection = _threshold(
            ThresholdType.feature_percentile,
            threshold=threshold_percentile,
            precomputed=t.full(
                (N_SAE,),
                threshold_percentile,
                dtype=t.float64,
            ),
        )
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=num_buckets),
        _config(
            selection,
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    rank = 99.5 if key == 99 else float(key)
    cutpoint = t.tensor(rank / 100.0, dtype=t.float32)
    below = t.nextafter(cutpoint, t.tensor(-t.inf))
    above = t.nextafter(cutpoint, t.tensor(t.inf))

    buckets = provider._bucket_percentiles_like_training(
        t.stack((below, cutpoint, above))
    )

    assert buckets.tolist() == [
        higher_bucket - 1,
        higher_bucket,
        higher_bucket,
    ]


def test_p99_endpoint_target_matches_training_below_and_at_boundary() -> None:
    """Exact p99 maps to continuous rank 99.5 without a uint8 roundtrip."""
    target_id = COMPACT_ORIGINAL_IDS[2]
    stat = _identity_percentile_stat()
    stats = {str(target_id): stat}
    num_buckets = 10
    threshold = 0.9
    boundaries = build_boundaries_from_stats(
        stats,
        n_sae_features=N_SAE,
        num_buckets=num_buckets,
        threshold_percentile=threshold,
    )
    percentile_table = build_percentile_table(stats, N_SAE)
    at_p99 = t.tensor(stat.percentiles["p99"], dtype=t.float32)
    below_p99 = t.nextafter(at_p99, t.tensor(-t.inf))
    feature_ids = t.tensor([target_id, target_id], dtype=t.int32)
    magnitudes = t.stack((below_p99, at_p99))

    training_buckets = discretize_activations(
        feature_ids,
        magnitudes,
        boundaries,
    )
    target_ranks = magnitude_to_percentile(
        feature_ids,
        magnitudes,
        percentile_table,
    )
    continuous_targets = (target_ranks / 100.0).to(t.float32)
    assert training_buckets.tolist() == [8, 9]

    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=num_buckets),
        _config(
            _threshold(
                ThresholdType.feature_percentile,
                threshold=threshold,
                precomputed=t.full((N_SAE,), threshold, dtype=t.float64),
            ),
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    predicted_buckets = provider._bucket_percentiles_like_training(continuous_targets)

    assert t.equal(predicted_buckets, training_buckets)


@pytest.mark.parametrize(
    ("num_buckets", "threshold_percentile", "key", "higher_bucket"),
    [
        (4, None, 25, 1),
        (10, 0.9, 95, 5),
        (10, 0.9, 99, 9),
    ],
)
def test_continuous_rank_matches_raw_training_boundaries(
    num_buckets: int,
    threshold_percentile: float | None,
    key: int,
    higher_bucket: int,
) -> None:
    """p25/p95/p99 map directly into one continuous rank currency."""
    target_id = COMPACT_ORIGINAL_IDS[2]
    stat = _identity_percentile_stat()
    stats = {str(target_id): stat}
    raw_magnitude = t.tensor([stat.percentiles[f"p{key}"]], dtype=t.float32)
    feature_ids = t.tensor([target_id], dtype=t.int32)
    boundaries = build_boundaries_from_stats(
        stats,
        n_sae_features=N_SAE,
        num_buckets=num_buckets,
        threshold_percentile=threshold_percentile,
    )
    training_bucket = discretize_activations(
        feature_ids,
        raw_magnitude,
        boundaries,
    )
    raw_rank = magnitude_to_percentile(
        feature_ids,
        raw_magnitude,
        build_percentile_table(stats, N_SAE),
    )
    expected_rank = 99.5 if key == 99 else float(key)
    assert raw_rank.tolist() == [expected_rank]

    fp_target = (raw_rank / 100.0).to(t.float32)

    if threshold_percentile is None:
        selection = _threshold(threshold=COMPACT_WIDTH)
    else:
        selection = _threshold(
            ThresholdType.feature_percentile,
            threshold=threshold_percentile,
            precomputed=t.full(
                (N_SAE,),
                threshold_percentile,
                dtype=t.float64,
            ),
        )
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=num_buckets),
        _config(
            selection,
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    predicted_bucket = provider._bucket_percentiles_like_training(fp_target)

    assert training_bucket.tolist() == [higher_bucket]
    assert t.equal(predicted_bucket, training_bucket)
    cutpoint = provider._bucket_percentile_cutpoints[higher_bucket - 1]
    assert t.equal(fp_target, cutpoint.reshape(1))
    immediately_below = t.nextafter(cutpoint, t.tensor(-t.inf))
    immediately_above = t.nextafter(cutpoint, t.tensor(t.inf))
    neighboring_buckets = provider._bucket_percentiles_like_training(
        t.stack((immediately_below, cutpoint, immediately_above))
    )
    assert neighboring_buckets.tolist() == [
        higher_bucket - 1,
        higher_bucket,
        higher_bucket,
    ]


@pytest.mark.parametrize(
    ("num_buckets", "threshold_percentile"),
    [
        (3, None),
        (4, None),
        (10, 0.9),
    ],
)
def test_every_persisted_key_boundary_and_continuous_neighbors(
    num_buckets: int,
    threshold_percentile: float | None,
) -> None:
    if threshold_percentile is None:
        selection = _threshold(threshold=COMPACT_WIDTH)
    else:
        selection = _threshold(
            ThresholdType.feature_percentile,
            threshold=threshold_percentile,
            precomputed=t.full(
                (N_SAE,),
                threshold_percentile,
                dtype=t.float64,
            ),
        )
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=num_buckets),
        _config(
            selection,
            num_buckets=num_buckets,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    keys = bucket_percentile_key_grid(
        num_buckets,
        threshold_percentile=threshold_percentile,
    )
    ranks = t.tensor(
        [99.5 if key == 99 else float(key) for key in keys],
        dtype=t.float32,
    )
    continuous_cutpoints = ranks / 100.0
    below = t.nextafter(
        continuous_cutpoints,
        t.full_like(continuous_cutpoints, -t.inf),
    )
    above = t.nextafter(
        continuous_cutpoints,
        t.full_like(continuous_cutpoints, t.inf),
    )
    expected_lower = list(range(num_buckets - 1))
    expected_higher = list(range(1, num_buckets))

    assert provider._bucket_percentiles_like_training(below).tolist() == expected_lower
    assert (
        provider._bucket_percentiles_like_training(continuous_cutpoints).tolist()
        == expected_higher
    )
    assert provider._bucket_percentiles_like_training(above).tolist() == expected_higher

    target_id = COMPACT_ORIGINAL_IDS[0]
    stat = _identity_percentile_stat()
    boundaries = build_boundaries_from_stats(
        {str(target_id): stat},
        n_sae_features=N_SAE,
        num_buckets=num_buckets,
        threshold_percentile=threshold_percentile,
    )
    raw_boundary_magnitudes = t.tensor(
        [stat.percentiles[f"p{key}"] for key in keys],
        dtype=t.float32,
    )
    training_buckets = discretize_activations(
        t.full((len(keys),), target_id, dtype=t.int32),
        raw_boundary_magnitudes,
        boundaries,
    )
    assert training_buckets.tolist() == expected_higher


def test_raw_plateau_exact_boundary_matches_training_right_tie() -> None:
    """A p33..p66 plateau must jump across both equal B=3 boundaries."""
    target_id = COMPACT_ORIGINAL_IDS[2]
    stat = _p33_through_p66_plateau_stat()
    stats = {str(target_id): stat}
    boundaries = build_boundaries_from_stats(
        stats,
        n_sae_features=N_SAE,
        num_buckets=3,
    )
    feature_ids = t.tensor([target_id], dtype=t.int32)
    exact_magnitude = t.tensor([0.5], dtype=t.float32)
    training_bucket = discretize_activations(
        feature_ids,
        exact_magnitude,
        boundaries,
    )
    rank = magnitude_to_percentile(
        feature_ids,
        exact_magnitude,
        build_percentile_table(stats, N_SAE),
    )
    assert training_bucket.tolist() == [2]
    assert rank.tolist() == [66.0]
    target = (rank / 100).to(t.float32)
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=3),
        _config(
            _threshold(threshold=COMPACT_WIDTH),
            num_buckets=3,
            selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )

    assert t.equal(
        provider._bucket_percentiles_like_training(target),
        training_bucket,
    )


@pytest.mark.parametrize(
    "bucket_producer",
    [
        FeatureBucketProducer.UNIFORM_PERCENTILE,
        FeatureBucketProducer.RAW_STATS_BOUNDARIES,
    ],
)
def test_single_active_bucket_never_uses_the_reserved_row(
    bucket_producer: FeatureBucketProducer,
) -> None:
    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=2),
        _config(
            _threshold(threshold=1),
            num_buckets=1,
            bucket_producer=bucket_producer,
        ),
    )
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(
            state,
            _logits_for_percentiles([0.1, 0.2, 0.95, 0.3, 0.4, 0.5]),
        )

    assert captured == [{COMPACT_ORIGINAL_IDS[2]: 0}]


@pytest.mark.parametrize(
    ("threshold_type", "threshold", "expected_key", "selection_producer"),
    [
        (
            ThresholdType.feature_percentile,
            0.95,
            95,
            FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        ),
        (
            ThresholdType.feature_percentile,
            0.95,
            95,
            FeatureSelectionProducer.RAW_MAGNITUDE,
        ),
        (
            ThresholdType.feature_percentile,
            0.29,
            28,
            FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        ),
        (
            ThresholdType.feature_percentile,
            0.29,
            28,
            FeatureSelectionProducer.RAW_MAGNITUDE,
        ),
        (
            ThresholdType.feature_percentile_segment_topk,
            2048.57,
            56,
            FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        ),
    ],
)
def test_feature_threshold_selection_matches_declared_producer(
    threshold_type: ThresholdType,
    threshold: float,
    expected_key: int,
    selection_producer: FeatureSelectionProducer,
) -> None:
    """Pin direct packed cutoffs vs legacy raw-statistics key projection."""
    target_id = COMPACT_ORIGINAL_IDS[2]
    stat = _identity_percentile_stat()
    stats = {str(target_id): stat}
    threshold_cfg = ThresholdConfig(threshold_type, threshold)
    assert feature_percentile_key(threshold_cfg) == expected_key
    raw_thresholds = build_thresholds_from_stats(
        stats,
        N_SAE,
        threshold_type,
        threshold,
    )
    raw_magnitude = t.tensor([stat.percentiles[f"p{expected_key}"]])
    raw_ids = t.tensor([target_id], dtype=t.int32)
    kept_raw_ids, _ = apply_threshold(
        threshold_cfg,
        raw_ids,
        raw_magnitude,
        raw_thresholds,
    )
    assert kept_raw_ids.tolist() == [target_id]

    raw_rank = magnitude_to_percentile(
        raw_ids,
        raw_magnitude,
        build_percentile_table(stats, N_SAE),
    )
    expected_rank = 99.5 if expected_key == 99 else float(expected_key)
    assert raw_rank.tolist() == [expected_rank]
    fp_target = (raw_rank / 100.0).to(t.float32)

    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _threshold(
                threshold_type,
                threshold=threshold,
                precomputed=raw_thresholds,
            ),
            selection_producer=selection_producer,
        ),
    )
    kept_ids, _ = provider._apply_selection(raw_ids, fp_target)
    assert provider._selection_percentile_thresholds is not None
    cutoff = provider._selection_percentile_thresholds[target_id]
    if selection_producer is FeatureSelectionProducer.FULL_PERCENTILE_RANK:
        expected_cutoff = t.tensor(
            extract_threshold_percentile(threshold_cfg),
            dtype=t.float32,
        )
    else:
        expected_cutoff = fp_target[0]
    assert t.equal(cutoff, expected_cutoff)
    expected_keep = bool(fp_target[0] >= expected_cutoff)
    assert kept_ids.tolist() == ([target_id] if expected_keep else [])
    if selection_producer is FeatureSelectionProducer.RAW_MAGNITUDE:
        assert expected_keep
        assert t.equal(fp_target, cutoff.reshape(1))
        below = t.nextafter(cutoff, t.tensor(-t.inf))
        kept_below, _ = provider._apply_selection(raw_ids, below.reshape(1))
        assert kept_below.numel() == 0


def test_compact_per_feature_threshold_availability_uses_original_ids() -> None:
    model = _model(COMPACT_VOCAB)
    percentiles = [0.10, 0.30, 0.85, 0.50, 0.70, 0.95]
    correct_row = 2
    correct_id = COMPACT_ORIGINAL_IDS[correct_row]
    wrong_row = 0
    thresholds = t.full((N_SAE,), float("inf"), dtype=t.float64)
    thresholds[correct_id] = 123.0
    thresholds[wrong_row] = 456.0
    provider = _provider(
        model,
        _config(
            _threshold(
                ThresholdType.feature_percentile,
                threshold=0.5,
                precomputed=thresholds,
            )
        ),
    )
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(state, _logits_for_percentiles(percentiles))
    # Finite raw values are availability only; normalized producer uses its
    # configured p50 cutoff and uniform packed-row buckets.
    expected_bucket = 3
    assert captured == [{correct_id: expected_bucket}]
    assert wrong_row not in captured[0]


@pytest.mark.parametrize(
    ("precomputed", "error_type", "match"),
    [
        (None, ValueError, "requires.*precomputed_thresholds"),
        (t.zeros(N_SAE + 1), ValueError, "must have shape"),
        (t.zeros(N_SAE, dtype=t.int64), TypeError, "floating-point dtype"),
        (
            t.cat((t.tensor([float("nan")]), t.zeros(N_SAE - 1))),
            ValueError,
            r"finite training thresholds or \+inf",
        ),
        (
            t.cat((t.tensor([float("-inf")]), t.zeros(N_SAE - 1))),
            ValueError,
            r"finite training thresholds or \+inf",
        ),
    ],
)
def test_feature_threshold_availability_tensor_fails_loudly(
    precomputed: t.Tensor | None,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        _provider(
            _model(COMPACT_VOCAB),
            _config(
                _threshold(
                    ThresholdType.feature_percentile,
                    threshold=0.5,
                    precomputed=precomputed,
                ),
            ),
        )


# The cluster {8, 9} keeps 8 (the lowest mask-surviving member) as its
# representative and consolidates 9 away, so only 8 owns a compact head row.
_CLUSTER_REPRESENTATIVE = COMPACT_ORIGINAL_IDS[0]
_CLUSTER_SIBLING = _CLUSTER_REPRESENTATIVE + 1


def _sibling_only_thresholds() -> t.Tensor:
    """Training thresholds whose ONLY statistics row is the consolidated sibling."""
    assert _CLUSTER_SIBLING not in COMPACT_ORIGINAL_IDS
    thresholds = t.full((N_SAE,), float("inf"), dtype=t.float64)
    thresholds[_CLUSTER_SIBLING] = 123.0
    return thresholds


def _feature_percentile_selection(precomputed: t.Tensor) -> ThresholdSelectionConfig:
    return _threshold(
        ThresholdType.feature_percentile,
        threshold=0.5,
        precomputed=precomputed,
    )


def test_cluster_representative_is_selectable_from_a_sibling_statistics_row(
    tmp_path: Path,
) -> None:
    """Clustered training reads sibling statistics BEFORE consolidation.

    A representative whose own row is ``+inf`` is still a legitimately trained,
    selectable feature when any mask-surviving member of its cluster has
    statistics, because clustered selection looked those siblings up in
    original-id space and max-folded them into the representative.
    """
    consolidation = _consolidation(tmp_path, [[_CLUSTER_REPRESENTATIVE, _CLUSTER_SIBLING]])
    thresholds = _sibling_only_thresholds()
    assert t.isinf(thresholds[_CLUSTER_REPRESENTATIVE])
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _feature_percentile_selection(thresholds),
            cluster_consolidation=consolidation,
        ),
    )
    percentiles = [0.85, 0.10, 0.10, 0.10, 0.10, 0.10]
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(state, _logits_for_percentiles(percentiles))

    assert captured == [{_CLUSTER_REPRESENTATIVE: 3}]


def test_cluster_derived_checkpoints_must_declare_their_consolidation(
    tmp_path: Path,
) -> None:
    """Without the consolidation the same representative is permanently suppressed.

    This is the documented caller contract, not an accepted defect: availability
    can only be folded over cluster members when the config carries the
    checkpoint's own consolidation.
    """
    consolidation = _consolidation(tmp_path, [[_CLUSTER_REPRESENTATIVE, _CLUSTER_SIBLING]])
    thresholds = _sibling_only_thresholds()
    percentiles = [0.85, 0.10, 0.10, 0.10, 0.10, 0.10]

    unaware = _provider(
        _model(COMPACT_VOCAB),
        _config(_feature_percentile_selection(thresholds)),
    )
    state = build_test_self_cond_state(unaware)
    with pytest.raises(EmptySelfConditioningSegmentError, match="zero features"):
        unaware.step(state, _logits_for_percentiles(percentiles))

    aware = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _feature_percentile_selection(thresholds),
            cluster_consolidation=consolidation,
        ),
    )
    aware_state = build_test_self_cond_state(aware)
    with _capture_segments(aware) as aware_captured:
        aware.step(aware_state, _logits_for_percentiles(percentiles))
    assert aware_captured == [{_CLUSTER_REPRESENTATIVE: 3}]


def test_availability_fold_leaves_unclustered_candidates_untouched(
    tmp_path: Path,
) -> None:
    """Only cluster members contribute; unclustered rows keep their own state."""
    available_unclustered = COMPACT_ORIGINAL_IDS[2]
    missing_unclustered = COMPACT_ORIGINAL_IDS[3]
    consolidation = _consolidation(tmp_path, [[_CLUSTER_REPRESENTATIVE, _CLUSTER_SIBLING]])
    thresholds = _sibling_only_thresholds()
    thresholds[available_unclustered] = 45.0
    assert t.isinf(thresholds[missing_unclustered])

    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _feature_percentile_selection(thresholds),
            cluster_consolidation=consolidation,
        ),
    )
    cutoffs = provider._selection_percentile_thresholds
    assert cutoffs is not None
    finite_ids = set(t.nonzero(t.isfinite(cutoffs), as_tuple=False).flatten().tolist())
    # The representative gains its sibling's availability; the sibling itself is
    # consolidated away and no longer selectable in its own right.
    assert finite_ids == {_CLUSTER_REPRESENTATIVE, available_unclustered}

    percentiles = [0.85, 0.10, 0.85, 0.90, 0.10, 0.10]
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(state, _logits_for_percentiles(percentiles))
    assert captured == [{_CLUSTER_REPRESENTATIVE: 3, available_unclustered: 3}]
    assert missing_unclustered not in captured[0]


def test_cluster_consolidation_rejects_an_uncompacted_checkpoint(
    tmp_path: Path,
) -> None:
    consolidation = _consolidation(tmp_path, [[_CLUSTER_REPRESENTATIVE, _CLUSTER_SIBLING]])
    with pytest.raises(ValueError, match="uncompacted checkpoint"):
        _provider(
            _model(None),
            _config(_threshold(), cluster_consolidation=consolidation),
        )


def test_cluster_consolidation_rejects_a_different_original_sae_width(
    tmp_path: Path,
) -> None:
    wider = N_SAE + 8
    consolidation = _consolidation(
        tmp_path,
        [[_CLUSTER_REPRESENTATIVE, _CLUSTER_SIBLING]],
        n_sae_features=wider,
    )
    with pytest.raises(ValueError, match=rf"(?s)different original SAE width.*{wider}.*{N_SAE}"):
        _provider(
            _model(COMPACT_VOCAB),
            _config(_threshold(), cluster_consolidation=consolidation),
        )


def test_cluster_consolidation_rejects_a_non_representative_candidate(
    tmp_path: Path,
) -> None:
    """Two vocabulary rows in one cluster cannot both be surviving representatives."""
    consolidated_away_candidate = COMPACT_ORIGINAL_IDS[1]
    consolidation = _consolidation(
        tmp_path,
        [[_CLUSTER_REPRESENTATIVE, consolidated_away_candidate]],
    )
    with pytest.raises(
        ValueError,
        match=(
            r"(?s)does not describe this checkpoint's effective vocabulary.*"
            rf"{consolidated_away_candidate}.*{_CLUSTER_REPRESENTATIVE}"
        ),
    ):
        _provider(
            _model(COMPACT_VOCAB),
            _config(_threshold(), cluster_consolidation=consolidation),
        )


@pytest.mark.parametrize("with_consolidation", [False, True])
def test_statistics_free_threshold_tensor_fails_loudly(
    tmp_path: Path,
    with_consolidation: bool,
) -> None:
    """An all-``+inf`` table can never select anything; folding cannot rescue it."""
    consolidation = (
        _consolidation(tmp_path, [[_CLUSTER_REPRESENTATIVE, _CLUSTER_SIBLING]])
        if with_consolidation
        else None
    )
    with pytest.raises(
        ValueError,
        match=r"every per-feature threshold in precomputed_thresholds is \+inf",
    ):
        _provider(
            _model(COMPACT_VOCAB),
            _config(
                _feature_percentile_selection(t.full((N_SAE,), float("inf"), dtype=t.float64)),
                cluster_consolidation=consolidation,
            ),
        )


def test_one_statistics_row_is_enough_to_construct() -> None:
    """Near-miss control: the all-``+inf`` guard is not rejecting sparse tables."""
    only_id = COMPACT_ORIGINAL_IDS[2]
    thresholds = t.full((N_SAE,), float("inf"), dtype=t.float64)
    thresholds[only_id] = 7.0
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(_feature_percentile_selection(thresholds)),
    )
    cutoffs = provider._selection_percentile_thresholds
    assert cutoffs is not None
    assert t.nonzero(t.isfinite(cutoffs), as_tuple=False).flatten().tolist() == [only_id]


def test_uint8_roundtripped_boundary_target_takes_the_lower_bucket() -> None:
    """Pin the deliberate CONTINUOUS-cutpoint contract (no uint8 replay).

    A head supervised from packed rows or pct overlays learns
    ``round(rank/100*255)/255``. For an ideal prediction of a target sitting
    exactly on a bucket boundary, that roundtrip lands one uint8 code BELOW the
    continuous cutpoint, so self-conditioning assigns the lower bucket.
    Quantizing the cutpoints instead would recover this single boundary value
    but would misclassify a ~4x wider band of true ranks on the permissive
    (include) side — and would break exact replay of packed selection and of
    SFT-trained heads, both of which compare against continuous cutoffs.
    """
    encoded = percentile_to_uint8(t.tensor([66.0], dtype=t.float64))
    assert encoded.tolist() == [168]
    decoded = uint8_to_percentile(encoded)
    assert t.equal(decoded, t.tensor([168.0], dtype=t.float32) / 255.0)
    assert float(decoded[0]) == pytest.approx(0.65882, abs=1e-5)

    provider = _provider(
        _model(COMPACT_VOCAB, num_buckets=3),
        _config(
            _threshold(threshold=COMPACT_WIDTH),
            num_buckets=3,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    )
    assert provider._bucket_percentile_cutpoints.tolist() == pytest.approx([0.33, 0.66])
    continuous_boundary = t.tensor([66.0], dtype=t.float32) / 100.0

    assert provider._bucket_percentiles_like_training(decoded).tolist() == [1]
    assert provider._bucket_percentiles_like_training(continuous_boundary).tolist() == [2]

    # The two error bands the decision trades off, in normalized rank units.
    wrongly_excluded = 168.5 / 255.0 - 0.66  # continuous cutpoints, ideal head
    wrongly_included = 0.66 - 167.5 / 255.0  # if the cutpoint were quantized
    assert wrongly_excluded < 0.5 / 255.0
    assert wrongly_included / wrongly_excluded == pytest.approx(4.0)


def test_compact_mask_filters_candidate_values_in_original_id_space() -> None:
    keeper = COMPACT_ORIGINAL_IDS[-2]
    mask = FeatureMask(FeatureMaskMode.whitelist, (keeper,))
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _threshold(threshold=COMPACT_WIDTH),
            feature_mask=mask,
        ),
    )
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(
            state,
            _logits_for_percentiles([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
        )
    assert set(captured[0]) == {keeper}


def test_compact_sampling_never_emits_an_id_without_a_vocab_row() -> None:
    provider = _provider(_model(COMPACT_VOCAB), _config(_sampling(11)))
    allowed = set(COMPACT_ORIGINAL_IDS)
    for seed in range(10):
        logits = t.randn(
            1,
            4,
            COMPACT_WIDTH,
            generator=t.Generator().manual_seed(seed),
        )
        state = build_test_self_cond_state(provider)
        with _capture_segments(provider) as captured:
            provider.step(state, logits)
        assert captured[0]
        assert set(captured[0]) <= allowed
        assert 13 not in captured[0]


def test_compact_top_k_sampling_matches_full_plus_whitelist_and_rng() -> None:
    """Equivalent candidate universes have exact selection and RNG parity."""
    compact_selection = _sampling(73, mode=SamplingMode.top_k_percentile)
    full_selection = _sampling(73, mode=SamplingMode.top_k_percentile)
    initial_rng = compact_selection.rng.get_state().clone()
    assert t.equal(initial_rng, full_selection.rng.get_state())

    whitelist = FeatureMask(FeatureMaskMode.whitelist, COMPACT_ORIGINAL_IDS)
    compact_provider = _provider(
        _model(COMPACT_VOCAB),
        _config(compact_selection, feature_mask=whitelist),
    )
    full_provider = _provider(
        _model(None),
        _config(full_selection, feature_mask=whitelist),
    )
    compact_percentiles = [0.21, 0.82, 0.63, 0.44, 0.95, 0.36]
    full_percentiles = [0.999] * N_SAE
    for feature_id, percentile in zip(
        COMPACT_ORIGINAL_IDS,
        compact_percentiles,
        strict=True,
    ):
        full_percentiles[feature_id] = percentile

    compact_state = build_test_self_cond_state(compact_provider)
    full_state = build_test_self_cond_state(full_provider)
    with _capture_segments(compact_provider) as compact_captured:
        compact_provider.step(
            compact_state,
            _logits_for_percentiles(compact_percentiles),
        )
    with _capture_segments(full_provider) as full_captured:
        full_provider.step(
            full_state,
            _logits_for_percentiles(full_percentiles),
        )

    assert compact_captured == [
        {
            COMPACT_ORIGINAL_IDS[4]: 3,
            COMPACT_ORIGINAL_IDS[1]: 3,
            COMPACT_ORIGINAL_IDS[2]: 2,
            COMPACT_ORIGINAL_IDS[3]: 1,
        }
    ]
    assert full_captured == compact_captured
    assert set(compact_captured[0]) <= set(COMPACT_ORIGINAL_IDS)
    assert 0 not in compact_captured[0], (
        "the full-model distractors must be removed by the equivalent whitelist"
    )
    compact_rng_after = compact_selection.rng.get_state()
    full_rng_after = full_selection.rng.get_state()
    assert t.equal(compact_rng_after, full_rng_after)
    assert not t.equal(compact_rng_after, initial_rng), (
        "top_k_percentile parity is vacuous unless k/tie-breaking consumes RNG"
    )


@pytest.mark.parametrize("feature_vocab", [None, COMPACT_VOCAB], ids=["full", "compact"])
@pytest.mark.parametrize(
    "pool_mode",
    [
        FeaturePredLossMode.LAST_TOKEN,
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.FULL_SEGMENT,
    ],
)
@pytest.mark.parametrize(
    ("selection_kind", "selection_producer", "bucket_producer"),
    [
        (
            "threshold",
            FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            FeatureBucketProducer.UNIFORM_PERCENTILE,
        ),
        (
            "threshold",
            FeatureSelectionProducer.RAW_MAGNITUDE,
            FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
        (
            "sampling",
            FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            FeatureBucketProducer.UNIFORM_PERCENTILE,
        ),
        (
            "sampling",
            FeatureSelectionProducer.FULL_PERCENTILE_RANK,
            FeatureBucketProducer.RAW_STATS_BOUNDARIES,
        ),
    ],
)
def test_no_bias_matches_the_literal_pr1_path_for_every_supported_surface(
    feature_vocab: FeatureVocab | None,
    pool_mode: FeaturePredLossMode,
    selection_kind: str,
    selection_producer: FeatureSelectionProducer,
    bucket_producer: FeatureBucketProducer,
) -> None:
    """The default path is exact, including state and sampler RNG evolution."""
    width = N_SAE if feature_vocab is None else COMPACT_WIDTH
    if selection_kind == "threshold":
        if selection_producer is FeatureSelectionProducer.RAW_MAGNITUDE:
            thresholds = t.full((N_SAE,), 0.5, dtype=t.float64)
            reference_selection = _threshold(
                ThresholdType.feature_percentile,
                0.5,
                thresholds,
            )
            actual_selection = _threshold(
                ThresholdType.feature_percentile,
                0.5,
                thresholds.clone(),
            )
        else:
            reference_selection = _threshold(threshold=min(3, width))
            actual_selection = _threshold(threshold=min(3, width))
    else:
        reference_selection = _sampling(914)
        actual_selection = _sampling(914)

    reference = _provider(
        _model(feature_vocab),
        _config(
            reference_selection,
            pool_mode=pool_mode,
            selection_producer=selection_producer,
            bucket_producer=bucket_producer,
        ),
    )
    actual = _provider(
        _model(feature_vocab),
        _config(
            actual_selection,
            pool_mode=pool_mode,
            selection_producer=selection_producer,
            bucket_producer=bucket_producer,
            fp_logit_bias=None,
        ),
    )
    # Independent oracle: bypass the new helper entirely, reproducing PR1's
    # literal pooled-f32 -> sigmoid edge.
    reference._apply_fp_logit_bias = lambda pooled: pooled  # ty: ignore[invalid-assignment]

    initial_segment = {3: 1} if feature_vocab is None else {COMPACT_ORIGINAL_IDS[0]: 1}
    reference_state = build_test_self_cond_state(reference, initial_segment)
    actual_state = build_test_self_cond_state(actual, initial_segment)
    logits = t.randn(1, 3, width, generator=t.Generator().manual_seed(812))
    logits_before = logits.clone()
    logits_version = logits._version
    global_rng_before = t.random.get_rng_state().clone()

    with _capture_segments(reference) as reference_segments:
        expected = reference.step(reference_state, logits)
    global_rng_after_reference = t.random.get_rng_state().clone()
    t.random.set_rng_state(global_rng_before)
    with _capture_segments(actual) as actual_segments:
        observed = actual.step(actual_state, logits)
    global_rng_after_actual = t.random.get_rng_state().clone()

    assert reference_segments == actual_segments
    _assert_packed_equal(expected.encoder_out, observed.encoder_out)
    assert expected.target_slot_idx == observed.target_slot_idx
    assert t.equal(global_rng_after_reference, global_rng_after_actual)
    assert t.equal(global_rng_after_actual, global_rng_before)
    assert t.equal(logits, logits_before)
    assert logits._version == logits_version
    if selection_kind == "sampling":
        assert isinstance(reference_selection, SamplingBundle)
        assert isinstance(actual_selection, SamplingBundle)
        assert t.equal(
            reference_selection.rng.get_state(),
            actual_selection.rng.get_state(),
        )
        assert not t.equal(
            reference_selection.rng.get_state(),
            t.Generator().manual_seed(914).get_state(),
        )


def test_none_bias_helper_returns_the_exact_input_object_without_side_effects() -> None:
    provider = _provider(_model(COMPACT_VOCAB), _config(_threshold()))
    pooled = t.randn(1, COMPACT_WIDTH)
    pooled_before = pooled.clone()
    pooled_version = pooled._version
    global_rng_before = t.random.get_rng_state().clone()

    result = provider._apply_fp_logit_bias(pooled)

    assert result is pooled
    assert result.data_ptr() == pooled.data_ptr()
    assert result._version == pooled_version
    assert t.equal(result, pooled_before)
    assert t.equal(t.random.get_rng_state(), global_rng_before)

    zero_provider = _provider(
        _model(COMPACT_VOCAB),
        _config(_threshold(), fp_logit_bias=t.zeros(COMPACT_WIDTH)),
    )
    zero_result = zero_provider._apply_fp_logit_bias(pooled)
    assert zero_result is not pooled
    assert zero_result.data_ptr() != pooled.data_ptr()
    assert t.equal(zero_result, pooled)


@pytest.mark.parametrize(
    "pool_mode",
    [
        FeaturePredLossMode.LAST_TOKEN,
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.FULL_SEGMENT,
    ],
)
def test_bias_order_is_pool_then_fp32_add_then_sigmoid(
    pool_mode: FeaturePredLossMode,
) -> None:
    """A sub-bfloat16-ULP bias distinguishes every plausible wrong ordering."""
    logits = (
        1.0
        + t.arange(3 * COMPACT_WIDTH, dtype=t.float32).reshape(
            1, 3, COMPACT_WIDTH
        )
        / 32.0
    ).to(t.bfloat16)
    logits_before = logits.clone()
    logits_version = logits._version
    source_bias = t.linspace(
        0.0005,
        0.0015,
        COMPACT_WIDTH,
        dtype=t.float64,
    )
    source_bias_before = source_bias.clone()
    source_bias_version = source_bias._version
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _threshold(threshold=COMPACT_WIDTH),
            pool_mode=pool_mode,
            fp_logit_bias=source_bias,
        ),
    )
    state = build_test_self_cond_state(provider)

    with _capture_selection_inputs(provider) as captured:
        provider.step(state, logits)

    assert len(captured) == 1
    _, scores = captured[0]
    pooled_f32 = pool_fp_logits(logits, pool_mode).detach().to(t.float32)
    canonical_bias = source_bias.to(t.float32)
    expected = t.sigmoid(pooled_f32 + canonical_bias)[0]
    assert scores.dtype == t.float32
    assert t.equal(scores, expected)

    added_before_fp32 = t.sigmoid(
        pool_fp_logits(
            logits + source_bias.to(t.bfloat16).reshape(1, 1, -1),
            pool_mode,
        ).to(t.float32)
    )[0]
    added_after_sigmoid = t.sigmoid(pooled_f32)[0] + canonical_bias
    assert not t.equal(scores, added_before_fp32)
    assert not t.equal(scores, added_after_sigmoid)
    assert t.equal(logits, logits_before)
    assert logits._version == logits_version
    assert t.equal(source_bias, source_bias_before)
    assert source_bias._version == source_bias_version


@pytest.mark.parametrize("dtype", [t.float16, t.bfloat16, t.float32, t.float64])
def test_bias_is_snapshotted_as_detached_contiguous_float32(dtype: t.dtype) -> None:
    backing = t.linspace(-1.0, 1.0, 2 * COMPACT_WIDTH, dtype=dtype)
    source = backing[::2]
    assert not source.is_contiguous()
    if dtype in (t.float32, t.float64):
        source.requires_grad_(True)
    source_before = source.detach().clone()

    cfg = _config(_threshold(), fp_logit_bias=source)
    canonical = cfg.fp_logit_bias
    assert canonical is not None
    assert canonical.dtype == t.float32
    assert canonical.is_contiguous()
    assert not canonical.requires_grad
    assert canonical.grad_fn is None
    assert canonical.data_ptr() != source.data_ptr()
    assert t.equal(canonical, source_before.to(t.float32))

    with t.no_grad():
        source.add_(10)
    assert t.equal(canonical, source_before.to(t.float32))
    provider = _provider(_model(COMPACT_VOCAB), cfg)
    private_bias = provider._fp_logit_bias
    assert private_bias is not None
    assert private_bias.data_ptr() != canonical.data_ptr()
    assert private_bias.dtype == t.float32
    assert private_bias.is_contiguous()
    assert not private_bias.requires_grad


def test_parameter_bias_never_rebuilds_an_autograd_graph() -> None:
    source = t.nn.Parameter(t.linspace(-0.5, 0.5, COMPACT_WIDTH))
    cfg = _config(_threshold(), fp_logit_bias=source)
    provider = _provider(_model(COMPACT_VOCAB), cfg)
    pooled = t.randn(1, COMPACT_WIDTH, requires_grad=True).detach()

    result = provider._apply_fp_logit_bias(pooled)

    assert not result.requires_grad
    assert result.grad_fn is None
    assert source.grad is None


def test_provider_snapshot_survives_config_mutation_and_revalidates_new_provider() -> None:
    cfg = _config(
        _threshold(threshold=COMPACT_WIDTH),
        fp_logit_bias=t.linspace(-0.5, 0.5, COMPACT_WIDTH),
    )
    provider = _provider(_model(COMPACT_VOCAB), cfg)
    pooled = t.zeros(1, COMPACT_WIDTH)
    expected = provider._apply_fp_logit_bias(pooled).clone()
    assert cfg.fp_logit_bias is not None
    cfg.fp_logit_bias.fill_(float("nan"))

    assert t.equal(provider._apply_fp_logit_bias(pooled), expected)
    with pytest.raises(ValueError, match="remain finite"):
        _provider(_model(COMPACT_VOCAB), cfg)


def test_provider_bias_snapshot_uses_constructor_device_and_device_drift_fails() -> None:
    source = t.linspace(-0.5, 0.5, COMPACT_WIDTH, device="cpu")
    provider = SelfConditionedFeatureProvider(
        model=_model(COMPACT_VOCAB),
        cfg=_config(_threshold(), fp_logit_bias=source),
        device=t.device("cpu"),
    )
    private_bias = provider._fp_logit_bias
    assert private_bias is not None
    assert private_bias.device == t.device("cpu")
    assert private_bias.data_ptr() != source.data_ptr()

    pooled = t.zeros(1, COMPACT_WIDTH, device="cpu")
    assert provider._apply_fp_logit_bias(pooled).device == pooled.device
    provider._fp_logit_bias = t.empty(COMPACT_WIDTH, device="meta")
    with pytest.raises(RuntimeError, match="different devices.*meta.*cpu"):
        provider._apply_fp_logit_bias(pooled)


@pytest.mark.parametrize(
    ("bias", "error_type", "match"),
    [
        (t.tensor(0.0), BeartypeCallHintParamViolation, "bias"),
        (t.zeros(2, COMPACT_WIDTH), BeartypeCallHintParamViolation, "bias"),
        (
            t.zeros(COMPACT_WIDTH, dtype=t.int64),
            BeartypeCallHintParamViolation,
            "bias",
        ),
        (
            t.zeros(COMPACT_WIDTH, dtype=t.bool),
            BeartypeCallHintParamViolation,
            "bias",
        ),
        (
            t.zeros(COMPACT_WIDTH, dtype=t.complex64),
            BeartypeCallHintParamViolation,
            "bias",
        ),
        (
            t.sparse_coo_tensor(
                t.tensor([[0]]),
                t.tensor([1.0]),
                (COMPACT_WIDTH,),
            ),
            BeartypeCallHintParamViolation,
            "bias",
        ),
        (
            t.empty(COMPACT_WIDTH, device="meta"),
            BeartypeCallHintParamViolation,
            "bias",
        ),
        (t.full((COMPACT_WIDTH,), float("nan")), ValueError, "remain finite"),
        (t.full((COMPACT_WIDTH,), float("inf")), ValueError, "remain finite"),
        (t.full((COMPACT_WIDTH,), float("-inf")), ValueError, "remain finite"),
        (
            t.full(
                (COMPACT_WIDTH,),
                t.finfo(t.float64).max,
                dtype=t.float64,
            ),
            ValueError,
            "represented as float32",
        ),
        (
            t.full(
                (COMPACT_WIDTH,),
                t.finfo(t.float64).min,
                dtype=t.float64,
            ),
            ValueError,
            "represented as float32",
        ),
    ],
)
def test_bias_validation_fails_loudly(
    bias: t.Tensor,
    error_type: type[Exception],
    match: str,
) -> None:
    with pytest.raises(error_type, match=match):
        _config(_threshold(), fp_logit_bias=bias)


@pytest.mark.parametrize("width", [0, COMPACT_WIDTH - 1, COMPACT_WIDTH + 1, N_SAE])
def test_compact_bias_width_mismatch_names_head_row_space(width: int) -> None:
    with pytest.raises(ValueError, match=rf"(?s)width {width}.*{COMPACT_WIDTH}.*head"):
        _provider(
            _model(COMPACT_VOCAB),
            _config(_threshold(), fp_logit_bias=t.zeros(width)),
        )


def test_compact_bias_row_selects_its_original_id_and_full_bias_selects_same_id() -> None:
    compact_row = 2
    original_id = COMPACT_ORIGINAL_IDS[compact_row]
    compact_bias = t.zeros(COMPACT_WIDTH)
    compact_bias[compact_row] = 20.0
    compact_provider = _provider(
        _model(COMPACT_VOCAB),
        _config(_threshold(threshold=1), fp_logit_bias=compact_bias),
    )
    full_bias = t.zeros(N_SAE)
    full_bias[original_id] = 20.0
    full_provider = _provider(
        _model(None),
        _config(_threshold(threshold=1), fp_logit_bias=full_bias),
    )
    compact_state = build_test_self_cond_state(compact_provider)
    full_state = build_test_self_cond_state(full_provider)

    with _capture_segments(compact_provider) as compact_segments:
        compact_provider.step(
            compact_state,
            t.zeros(1, 1, COMPACT_WIDTH),
        )
    with _capture_segments(full_provider) as full_segments:
        full_provider.step(
            full_state,
            t.zeros(1, 1, N_SAE),
        )

    assert compact_segments == [{original_id: 3}]
    assert full_segments == compact_segments
    assert compact_row not in compact_segments[0]


def test_bias_precedes_raw_selection_and_bucket_producers() -> None:
    """Bias is orthogonal to both producer axes used downstream."""
    target_row = 2
    target_id = COMPACT_ORIGINAL_IDS[target_row]
    source_percentile = 0.20
    target_percentile = 0.80
    source_logit = float(t.logit(t.tensor(source_percentile)))
    bias = t.zeros(COMPACT_WIDTH)
    bias[target_row] = float(t.logit(t.tensor(target_percentile))) - source_logit
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _threshold(
                ThresholdType.feature_percentile,
                0.5,
                t.full((N_SAE,), 0.5, dtype=t.float64),
            ),
            selection_producer=FeatureSelectionProducer.RAW_MAGNITUDE,
            bucket_producer=FeatureBucketProducer.RAW_STATS_BOUNDARIES,
            fp_logit_bias=bias,
        ),
    )
    state = build_test_self_cond_state(provider)
    logits = t.full((1, 1, COMPACT_WIDTH), source_logit)

    with _capture_selection_inputs(provider) as selection_inputs:
        with _capture_segments(provider) as captured:
            provider.step(state, logits)

    selected_scores = selection_inputs[0][1]
    assert selected_scores[target_row] > 0.79
    assert captured == [{target_id: 1}]


def test_negative_bias_removes_only_its_compact_original_id() -> None:
    removed_row = 2
    removed_id = COMPACT_ORIGINAL_IDS[removed_row]
    bias = t.zeros(COMPACT_WIDTH)
    bias[removed_row] = -20.0
    precomputed = t.zeros(N_SAE, dtype=t.float32)
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _threshold(
                ThresholdType.feature_percentile,
                threshold=0.5,
                precomputed=precomputed,
            ),
            fp_logit_bias=bias,
        ),
    )
    state = build_test_self_cond_state(provider)
    with _capture_segments(provider) as captured:
        provider.step(
            state,
            t.full((1, 1, COMPACT_WIDTH), 1.0),
        )

    assert set(captured[0]) == set(COMPACT_ORIGINAL_IDS) - {removed_id}


def test_feature_mask_defeats_an_arbitrarily_positive_bias() -> None:
    forbidden_row = 2
    forbidden_id = COMPACT_ORIGINAL_IDS[forbidden_row]
    keeper_id = COMPACT_ORIGINAL_IDS[0]
    bias = t.zeros(COMPACT_WIDTH)
    bias[forbidden_row] = 1_000_000.0
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(
            _threshold(threshold=COMPACT_WIDTH),
            feature_mask=FeatureMask(FeatureMaskMode.whitelist, (keeper_id,)),
            fp_logit_bias=bias,
        ),
    )
    state = build_test_self_cond_state(provider)

    with _capture_segments(provider) as captured:
        provider.step(
            state,
            t.zeros(1, 1, COMPACT_WIDTH),
        )

    assert set(captured[0]) == {keeper_id}
    assert forbidden_id not in captured[0]


def test_documented_sparse_and_dense_compact_bias_translations_are_exact() -> None:
    ids = t.tensor([COMPACT_ORIGINAL_IDS[1], COMPACT_ORIGINAL_IDS[4]])
    rows = COMPACT_VOCAB.to_compact(ids)
    assert rows.tolist() == [1, 4]
    sparse_bias = t.zeros(COMPACT_WIDTH)
    sparse_bias[rows] = t.tensor([2.0, -3.0])
    assert sparse_bias.tolist() == [0.0, 2.0, 0.0, 0.0, -3.0, 0.0]

    full_bias = t.zeros(N_SAE, dtype=t.float32)
    full_bias[COMPACT_ORIGINAL_IDS[1]] = 4.0
    full_bias[COMPACT_ORIGINAL_IDS[4]] = -5.0
    controlled_ids = full_bias.nonzero().flatten()
    assert COMPACT_VOCAB.to_compact(controlled_ids).tolist() == [1, 4]
    assert t.equal(
        COMPACT_VOCAB.select_rows(full_bias),
        full_bias[t.tensor(COMPACT_ORIGINAL_IDS)],
    )
    with pytest.raises(FeatureVocabError):
        COMPACT_VOCAB.to_compact(t.tensor([0]))

    excluded_control = full_bias.clone()
    excluded_control[3] = 20.0
    with pytest.raises(FeatureVocabError):
        COMPACT_VOCAB.to_compact(excluded_control.nonzero().flatten())


@pytest.mark.parametrize(
    "mode",
    [SamplingMode.weighted, SamplingMode.top_k_percentile],
)
def test_sampling_receives_biased_percentiles_and_remains_reproducible(
    mode: SamplingMode,
) -> None:
    bias = t.linspace(-1.0, 1.0, COMPACT_WIDTH)
    first_selection = _sampling(770, mode=mode)
    second_selection = _sampling(770, mode=mode)
    first = _provider(
        _model(COMPACT_VOCAB),
        _config(first_selection, fp_logit_bias=bias),
    )
    second = _provider(
        _model(COMPACT_VOCAB),
        _config(second_selection, fp_logit_bias=bias.clone()),
    )
    logits = t.linspace(-0.8, 0.8, COMPACT_WIDTH).reshape(1, 1, -1)
    expected_scores = t.sigmoid(logits.to(t.float32) + bias)[0, 0]
    first_state = build_test_self_cond_state(first)
    second_state = build_test_self_cond_state(second)

    with (
        _capture_selection_inputs(first) as first_inputs,
        _capture_segments(first) as first_segments,
    ):
        first.step(first_state, logits)
    with (
        _capture_selection_inputs(second) as second_inputs,
        _capture_segments(second) as second_segments,
    ):
        second.step(second_state, logits)

    assert t.equal(first_inputs[0][1], expected_scores)
    assert t.equal(second_inputs[0][1], expected_scores)
    assert first_segments == second_segments
    assert t.equal(first_selection.rng.get_state(), second_selection.rng.get_state())
    assert not t.equal(
        first_selection.rng.get_state(),
        t.Generator().manual_seed(770).get_state(),
    )


def test_compact_head_width_must_equal_vocab_size() -> None:
    model = _model(COMPACT_VOCAB)
    wrong_width = COMPACT_WIDTH + 1
    model.feature_pred_head = Linear(model.dec_config.n_embd, wrong_width, bias=True)
    with pytest.raises(RuntimeError, match=rf"(?s){wrong_width}.*{COMPACT_WIDTH}"):
        _provider(model, _config(_threshold()))


def test_config_original_sae_width_must_equal_the_model() -> None:
    with pytest.raises(ValueError, match=rf"(?s){N_SAE - 1}.*{N_SAE}"):
        _provider(
            _model(COMPACT_VOCAB),
            _config(_threshold(), n_sae_features=N_SAE - 1),
        )


@pytest.mark.parametrize("num_buckets", [1, ENCODER_BUCKETS + 1])
def test_config_bucket_count_must_match_full_or_one_reserved(
    num_buckets: int,
) -> None:
    with pytest.raises(ValueError, match="full bucket count"):
        _provider(
            _model(COMPACT_VOCAB),
            _config(_threshold(), num_buckets=num_buckets),
        )


@pytest.mark.parametrize("num_buckets", [ENCODER_BUCKETS - 1, ENCODER_BUCKETS])
def test_config_accepts_both_structurally_valid_bucket_spaces(
    num_buckets: int,
) -> None:
    """Reservation presence lives in metadata, not in the model object.

    The provider can reject every shape except full and full-minus-one, but the
    caller must use ``active_bucket_count`` to choose the correct member.
    """
    provider = _provider(
        _model(COMPACT_VOCAB),
        _config(_threshold(), num_buckets=num_buckets),
    )
    assert provider.cfg.num_buckets == num_buckets


def test_checkpoint_vocab_buffer_must_match_the_python_vocab() -> None:
    model = _model(COMPACT_VOCAB)
    model.encoder.feature_vocab_ids[0] = COMPACT_ORIGINAL_IDS[1]
    with pytest.raises(FeatureVocabError, match="feature_vocab_ids disagrees"):
        _provider(model, _config(_threshold()))


@pytest.mark.parametrize("width", [COMPACT_WIDTH - 1, COMPACT_WIDTH + 1])
def test_step_rejects_logits_from_a_different_feature_space(width: int) -> None:
    provider = _provider(_model(COMPACT_VOCAB), _config(_threshold()))
    state = build_test_self_cond_state(provider)
    with pytest.raises(ValueError, match=rf"(?s)width {width}.*{COMPACT_WIDTH}"):
        provider.step(state, t.zeros(1, 4, width))


def test_seed_from_prompt_inherits_the_feature_width_guard() -> None:
    provider = _provider(_model(COMPACT_VOCAB), _config(_threshold()))
    state = build_test_self_cond_state(provider)
    with pytest.raises(ValueError, match="width"):
        provider.seed_from_prompt(state, t.zeros(1, 16, COMPACT_WIDTH + 1))


@pytest.mark.parametrize(
    "bad_logits",
    [
        t.tensor(0.0),
        t.zeros(16, COMPACT_WIDTH),
        t.zeros(1, 0, COMPACT_WIDTH),
        t.zeros(2, 16, COMPACT_WIDTH),
    ],
)
def test_seed_from_prompt_rejects_invalid_rank_batch_or_empty_prompt(
    bad_logits: t.Tensor,
) -> None:
    provider = _provider(_model(COMPACT_VOCAB), _config(_threshold()))
    state = build_test_self_cond_state(provider)
    with pytest.raises(ValueError, match="shape|batch size|at least one"):
        provider.seed_from_prompt(state, bad_logits)


class _CompactContextProvider:
    def __init__(
        self,
        tokenizer,
        feature_id: int = COMPACT_ORIGINAL_IDS[0],
        bucket: int = 1,
    ) -> None:
        self._tokenizer = tokenizer
        self._feature_id = feature_id
        self._bucket = bucket

    def get(self, token_ids):
        ids = [int(token_id) for token_id in token_ids]
        if not ids:
            return None
        char_end = token_to_char_offsets(self._tokenizer, ids)[-1]
        return (
            ContextFeatureSegment(
                feature_ids=(self._feature_id,),
                buckets=(self._bucket,),
                char_start=0,
                char_end=int(char_end),
            ),
        )


def test_compact_eval_wrapper_crosses_two_feedback_boundaries() -> None:
    """Drive prefill and decode through the real target-aware wrapper."""
    model = _model(COMPACT_VOCAB)
    model.eval()
    tokenizer = MockTokenizer()
    segment_size = 4
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SelfConditionedTarget(),
        context_provider=_CompactContextProvider(tokenizer),
        self_cond_config=_config(
            _threshold(threshold=2),
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
        ),
    )
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.dec_config.n_kv_head,
        seq_len=model.dec_config.sequence_len,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer,
        device=_CPU,
        dtype=t.float32,
    )
    prompt = list(range(2, 10))
    with wrapper.conditioning_context(context_token_ids=[prompt]):
        wrapper(t.tensor([prompt], dtype=t.long), kv_cache=kv_cache)

    runtime = wrapper._self_cond
    assert runtime is not None
    assert runtime.state is not None
    slots_after_prefill = runtime.state.encoder_out.layout.n_segments
    with _capture_segments(runtime.provider) as captured:
        for token_id in range(2 * segment_size):
            wrapper(
                t.tensor([[token_id + 2]], dtype=t.long),
                kv_cache=kv_cache,
            )

    assert runtime.state is not None
    assert runtime.state.encoder_out.layout.n_segments == slots_after_prefill + 2
    assert (
        runtime.state.target_slot_idx
        == runtime.state.encoder_out.layout.n_segments - 1
    )
    assert len(captured) == 2
    allowed = set(COMPACT_ORIGINAL_IDS)
    assert all(set(segment) <= allowed for segment in captured)


def test_wrapper_applies_logit_bias_once_per_prompt_or_decode_boundary() -> None:
    """The hook runs per provider step, never once per decoded token."""
    model = _model(COMPACT_VOCAB)
    model.eval()
    tokenizer = MockTokenizer()
    segment_size = 4
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=SelfConditionedTarget(),
        context_provider=_CompactContextProvider(tokenizer),
        self_cond_config=_config(
            _threshold(threshold=2),
            pool_mode=FeaturePredLossMode.LAST_TOKEN,
            fp_logit_bias=t.linspace(-0.5, 0.5, COMPACT_WIDTH),
        ),
    )
    runtime = wrapper._self_cond
    assert runtime is not None
    bias_calls: list[t.Tensor] = []
    original_apply = runtime.provider._apply_fp_logit_bias

    def count_apply(pooled_f32: t.Tensor) -> t.Tensor:
        bias_calls.append(pooled_f32.detach().clone())
        return original_apply(pooled_f32)

    runtime.provider._apply_fp_logit_bias = count_apply  # ty: ignore[invalid-assignment]
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.dec_config.n_kv_head,
        seq_len=model.dec_config.sequence_len,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer,
        device=_CPU,
        dtype=t.float32,
    )
    prompt = list(range(2, 10))
    with wrapper.conditioning_context(context_token_ids=[prompt]):
        wrapper(t.tensor([prompt], dtype=t.long), kv_cache=kv_cache)
    assert len(bias_calls) == 1, "prompt-tail seeding applies the bias once"

    for token_idx in range(2 * segment_size):
        wrapper(
            t.tensor([[token_idx + 2]], dtype=t.long),
            kv_cache=kv_cache,
        )
        expected_calls = 1 + (token_idx + 1) // segment_size
        assert len(bias_calls) == expected_calls

    assert len(bias_calls) == 3
    assert all(call.dtype == t.float32 for call in bias_calls)


@pytest.mark.parametrize(
    "bad_logits",
    [
        t.tensor(0.0),
        t.zeros(4, COMPACT_WIDTH),
        t.zeros(1, 0, COMPACT_WIDTH),
        t.zeros(1, 5, COMPACT_WIDTH),
        t.zeros(2, 4, COMPACT_WIDTH),
    ],
)
def test_step_rejects_invalid_rank_batch_or_time_before_pooling(
    bad_logits: t.Tensor,
) -> None:
    provider = _provider(_model(COMPACT_VOCAB), _config(_threshold()))
    state = build_test_self_cond_state(provider)
    with pytest.raises(ValueError, match="shape|batch size|time dimension"):
        provider.step(state, bad_logits)


def test_feature_percentile_reference_uses_the_same_original_id_currency() -> None:
    """Independent oracle for the threshold-table indexing assertion above."""
    ids = t.tensor(COMPACT_ORIGINAL_IDS, dtype=t.int32)
    scores = t.tensor([0.10, 0.30, 0.85, 0.50, 0.70, 0.95])
    thresholds = t.full((N_SAE,), 2.0, dtype=t.float64)
    thresholds[COMPACT_ORIGINAL_IDS[2]] = 0.80
    kept_ids, _ = apply_threshold(
        ThresholdConfig(ThresholdType.feature_percentile, 0.0),
        ids,
        scores,
        thresholds,
    )
    assert kept_ids.tolist() == [COMPACT_ORIGINAL_IDS[2]]


_L22_CHECKPOINT_RELATIVE = (
    Path("conditional_encdec_checkpoints/l22_main_h100_8rank_db16_v1/s128")
    / "dist_geometric/mean_12.0/std_12.0/wl_9172c540310ba842"
    / "fc_5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039"
)
_L22_STEP = 54600


@pytest.mark.very_slow
def test_real_l22_compact_checkpoint_drives_multiple_feedback_steps() -> None:
    """Machine-bound real-model gate; absence is a failed provision, not a skip."""
    base_dir = Path(get_base_dir())
    checkpoint_dir = base_dir / _L22_CHECKPOINT_RELATIVE
    meta_path = checkpoint_dir / f"meta_{_L22_STEP:06d}.json"
    if not meta_path.exists():
        pytest.fail(
            f"real compact checkpoint is not provisioned at {meta_path}; "
            "set NANOCHAT_BASE_DIR to the cache that owns the l22 artifact"
        )
    if not t.cuda.is_available():
        pytest.fail("CUDA is required for the real l22 compact self-conditioning gate")

    device = t.device("cuda")
    model, tokenizer, meta = build_encdec_model(
        str(checkpoint_dir),
        _L22_STEP,
        device,
        ModelPhase.EVAL,
        eval_conditioning=EvalConditioning.conditions_within_vocab(
            "compact_self_conditioned_generation",
        ),
    )
    model.eval()
    vocab = model.feature_vocab
    assert vocab is not None
    assert model.feature_pred_head is not None
    assert model.feature_pred_head.out_features == vocab.size

    user_config = meta.get("user_config")
    assert isinstance(user_config, dict)
    active_buckets = active_bucket_count(
        int(model.encoder.config.num_buckets),
        reserves_negative_bucket=user_config.get("negative_selection") is not None,
    )
    segment_size = 8
    fp_logit_bias = t.zeros(vocab.size, dtype=t.float32)
    fp_logit_bias[0] = 0.25
    self_cond_config = SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(segment_size),
        pool_mode=FeaturePredLossMode.MAX_POOL,
        selection=_threshold(threshold=12),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
        num_buckets=PositiveInt.parse(active_buckets),
        n_sae_features=PositiveInt.parse(vocab.n_sae_features),
        fp_logit_bias=fp_logit_bias,
    )
    seed_id = int(vocab.original_ids[0])
    context_bucket = 0
    assert 0 <= context_bucket < active_buckets
    engine = build_engine_self_conditioned(
        model,
        tokenizer,
        device,
        self_cond_config,
        context_provider=_CompactContextProvider(
            tokenizer,
            seed_id,
            bucket=context_bucket,
        ),
    )
    wrapper = engine._model
    assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
    runtime = wrapper._self_cond
    assert runtime is not None
    assert runtime.provider._fp_logit_bias is not None
    assert runtime.provider._fp_logit_bias.device == next(model.parameters()).device
    bias_calls = 0
    original_apply = runtime.provider._apply_fp_logit_bias

    def count_apply(pooled_f32: t.Tensor) -> t.Tensor:
        nonlocal bias_calls
        bias_calls += 1
        return original_apply(pooled_f32)

    runtime.provider._apply_fp_logit_bias = count_apply
    max_new_tokens = 3 * segment_size + 1
    with _capture_segments(runtime.provider) as captured:
        output = engine.generate(
            "The Eiffel Tower stands in Paris, and every spring the gardens "
            "along the Seine fill with visitors. Guides explain that its iron "
            "lattice was assembled in under two years.",
            max_new_tokens=max_new_tokens,
            temperature=0.0,
        )

    assert isinstance(output, str)
    assert max_new_tokens > 2 * segment_size
    assert runtime.state is not None
    assert runtime.state.encoder_out.layout.n_segments >= 5
    assert (
        runtime.state.target_slot_idx
        == runtime.state.encoder_out.layout.n_segments - 1
    )
    allowed = set(vocab.original_ids)
    # One prompt-seed selection plus at least three true decode-boundary steps.
    assert len(captured) >= 4
    assert bias_calls == len(captured)
    assert all(segment for segment in captured)
    assert all(set(segment) <= allowed for segment in captured)
    assert all(
        0 <= bucket < active_buckets
        for segment in captured
        for bucket in segment.values()
    )
