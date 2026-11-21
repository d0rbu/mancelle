"""Compact-vocabulary self-conditioning rollout and STE contracts (v2)."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from loguru import logger

import nanochat.flash_attention as _fa_mod
from errors import FeatureClustersError, SelfCondTrainingError
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import Linear
from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.encoder import SteForm, SteInputs
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    MutableConditioning,
    RolloutResult,
    SelectedActions,
    SelfCondDataSource,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    _step_metrics,
    build_selfcond_state,
    lm_conditioned_slot_mask,
    pool_fp_logits_per_segment,
    rollout,
    sample_action_counts,
    select_topk_bottoml,
    selfcond_loss,
    validate_selfcond_prerequisites,
    write_predicted_slot,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    FeatureClusterDefinition,
    FeatureClusterDescriptor,
    ResolvedFeatureClusters,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch

N_SAE = 64
MASK_IDS = tuple(range(0, N_SAE, 4))
FEATURE_MASK = FeatureMask(FeatureMaskMode.whitelist, MASK_IDS)
CLUSTER_DEFINITION = FeatureClusterDefinition(clusters=((0, 4), (8, 12)))
RESOLVED_CLUSTERS = ResolvedFeatureClusters(
    source_path=Path("<compact-selfcond-clusters>"),
    definition=CLUSTER_DEFINITION,
    descriptor=FeatureClusterDescriptor.from_definition(
        CLUSTER_DEFINITION,
        n_sae_features=N_SAE,
    ),
)
CONSOLIDATION = FeatureClusterConsolidation.build(RESOLVED_CLUSTERS, FEATURE_MASK)
_COMPACT_VOCAB = effective_feature_vocab(CONSOLIDATION, N_SAE)
if _COMPACT_VOCAB is None:
    raise AssertionError("clustered compact test fixture unexpectedly resolved to identity")
COMPACT_VOCAB: FeatureVocab = _COMPACT_VOCAB
RAW_MASK_VOCAB = FeatureVocab(n_sae_features=N_SAE, original_ids=MASK_IDS)


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _config(
    *,
    ste_form: SteForm = SteForm.DENSE,
    positive_max: int = 3,
    negative_max: int = 2,
) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=ste_form,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=2,
            max_k=positive_max,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=negative_max,
        ),
        seed=1901,
    )


def _compact_model(
    *,
    feature_vocab: FeatureVocab = COMPACT_VOCAB,
    seed: int = 1902,
) -> EncoderDecoderGPT:
    t.manual_seed(seed)
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
    )
    model = build_tiny_model(
        n_sae_features=N_SAE,
        num_buckets=2,
        feature_pred_config=fp_config,
        n_layer=2,
        xattn_frequency=1,
        feature_vocab=feature_vocab,
    )
    assert model.feature_pred_head is not None
    with t.no_grad():
        model.feature_pred_head.weight.zero_()
        model.feature_pred_head.bias.copy_(
            t.linspace(-3.0, 3.0, feature_vocab.size, dtype=t.float32)
        )
    return model.eval()


def _compact_batch(
    feature_vocab: FeatureVocab = COMPACT_VOCAB,
    *,
    seed: int = 1903,
) -> SelfcondBatch:
    batch = make_batch([[2], [2]], T=12, F=4, n_sae=N_SAE, seed=seed)
    candidates = feature_vocab.new_to_old_tensor()
    batch.enc_fids.copy_(candidates[batch.enc_fids.remainder(feature_vocab.size)])
    return batch


def _meta(*, compact: bool, clustered: bool, feature_delta: dict | None = None) -> dict:
    return {
        "user_config": {
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "max_pool",
                "loss_fn": "bce",
            },
            "feature_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "min_k": 2,
                "max_k": 3,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 1.0,
                "std": 1.0,
                "min_k": 1,
                "max_k": 2,
            },
            "num_buckets": 2,
            "feature_mask": FEATURE_MASK.canonical_dict() if compact else None,
            "feature_clusters": {"path": "clusters.json"} if clustered else None,
            "feature_delta": feature_delta,
            "label_dropout": 0.0,
            "next_segment_conditioning_offset": False,
            "use_precomputed_batches": False,
            "feature_dir": "/tmp/plain-features",
        }
    }


def _predicted_width(batch: SelfcondBatch, cfg: SelfCondTrainConfig) -> int:
    assert cfg.positive_counts.max_k is not None
    assert cfg.negative_counts.max_k is not None
    return max(
        batch.enc_fids.shape[-1],
        (cfg.positive_counts.max_k - 1) + (cfg.negative_counts.max_k - 1),
    )


def _translate_rows_to_original(
    row_ids: t.Tensor,
    candidates: t.Tensor,
) -> t.Tensor:
    """Inference-contract mirror: translate only real rows and retain -1 padding."""
    original_ids = row_ids.clone()
    real = row_ids >= 0
    original_ids[real] = candidates[row_ids[real]]
    assert t.equal(original_ids[~real], row_ids[~real])
    return original_ids


@t.no_grad()
def _inference_style_rollout_reference(
    model: EncoderDecoderGPT,
    batch: SelfcondBatch,
    cfg: SelfCondTrainConfig,
    rng: t.Generator,
) -> RolloutResult:
    """Independent loop using inference's compact-row to original-ID gather."""
    feature_vocab = model.feature_vocab
    assert feature_vocab is not None
    candidates = feature_vocab.new_to_old_tensor(device=batch.enc_fids.device)
    state = build_selfcond_state(batch, _predicted_width(batch, cfg))
    predicted = lm_conditioned_slot_mask(batch.seg_map, batch.seg_doc_id)
    batch_size, n_seg = batch.seg_doc_id.shape
    counts_k = t.zeros(batch_size, n_seg, dtype=t.int64)
    counts_l = t.zeros_like(counts_k)

    for slot in range(1, n_seg):
        live = predicted[:, slot]
        if not bool(live.any()):
            continue
        fp_logits = model(
            batch.inputs,
            None,
            state.fids,
            state.mags,
            state.mask,
            batch.seg_map,
            fp_logits_only=True,
        )
        pooled = pool_fp_logits_per_segment(
            fp_logits,
            batch.seg_map,
            PositiveInt.parse(n_seg),
            cfg.pool_mode,
        )
        scores = t.sigmoid(pooled[:, slot - 1].float())
        sampled_k, sampled_l = sample_action_counts(cfg, rng, batch_size)
        actions = select_topk_bottoml(scores, sampled_k, sampled_l)
        translated = SelectedActions(
            pos_ids=_translate_rows_to_original(actions.pos_ids, candidates),
            neg_ids=_translate_rows_to_original(actions.neg_ids, candidates),
            k=actions.k,
            l=actions.l,
        )
        write_predicted_slot(state, slot, translated, live)
        counts_k[:, slot] = t.where(live, sampled_k, 0)
        counts_l[:, slot] = t.where(live, sampled_l, 0)

    return RolloutResult(
        conditioning=state,
        predicted=predicted,
        k=counts_k,
        l=counts_l,
    )


def _assert_conditioning_equal(actual: MutableConditioning, expected: MutableConditioning) -> None:
    assert t.equal(actual.fids, expected.fids)
    assert t.equal(actual.mags, expected.mags)
    assert t.equal(actual.mask, expected.mask)
    assert t.equal(actual.sign, expected.sign)


def test_compact_rollout_writes_only_original_representative_ids() -> None:
    model = _compact_model()
    batch = _compact_batch()
    result = rollout(
        model,
        batch,
        _config(),
        t.Generator().manual_seed(1904),
        None,
        consolidation=CONSOLIDATION,
    )

    selected = result.conditioning.fids[result.conditioning.sign != 0]
    assert selected.numel() > 0
    assert set(selected.tolist()) <= set(COMPACT_VOCAB.original_ids)
    assert bool((selected >= COMPACT_VOCAB.size).any())
    CONSOLIDATION.assert_representatives_only(
        selected,
        surface="compact rollout test output",
    )


def test_compact_rollout_matches_inference_style_translation_reference() -> None:
    """Pins the TRANSLATION mechanics shared with inference feedback: stable
    row selection over compact-width scores, ``new_to_old_tensor`` gather,
    original-id write semantics. It deliberately does NOT pin the score
    TRANSFORM: training ranks ``sigmoid(x)`` while inference's
    ``_step_score_interpolation`` ranks ``lerp(2*sigmoid(x)-1, target, w)``
    (see the divergence test below — review finding)."""
    model = _compact_model(seed=1905)
    batch = _compact_batch(seed=1906)
    cfg = _config()

    actual = rollout(model, batch, cfg, t.Generator().manual_seed(1907), None)
    expected = _inference_style_rollout_reference(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(1907),
    )

    assert t.equal(actual.predicted, expected.predicted)
    assert t.equal(actual.k, expected.k)
    assert t.equal(actual.l, expected.l)
    _assert_conditioning_equal(actual.conditioning, expected.conditioning)


def test_training_and_inference_score_transforms_rank_identically_unsaturated() -> None:
    """On non-saturated logits, training's ``sigmoid(x)`` and inference's
    weight-0 ``2*sigmoid(x)-1`` are monotone-equivalent, so stable top-k /
    bottom-l selections agree exactly."""
    logits = t.linspace(-8.0, 8.0, 32).flip(0).unsqueeze(0)
    k = t.tensor([5], dtype=t.int64)
    l = t.tensor([4], dtype=t.int64)  # noqa: E741 - spec notation

    training = select_topk_bottoml(t.sigmoid(logits), k, l)
    inference = select_topk_bottoml(2.0 * t.sigmoid(logits) - 1.0, k, l)

    assert t.equal(training.pos_ids, inference.pos_ids)
    assert t.equal(training.neg_ids, inference.neg_ids)


def test_training_and_inference_score_transforms_diverge_under_saturation() -> None:
    """KNOWN INHERITED DIVERGENCE (predates compact v2; tracked as a
    follow-up issue): float32 ``2*sigmoid(x)-1`` collapses strongly negative
    logits to exactly -1.0, so inference resolves them as ties by index order,
    while ``sigmoid(x)`` still distinguishes them. If a future change unifies
    the transforms, this test flips and must be replaced by an equality pin."""
    logits = t.tensor([[-20.0, -21.0, 20.0, 21.0]])
    k = t.tensor([1], dtype=t.int64)
    l = t.tensor([1], dtype=t.int64)  # noqa: E741 - spec notation

    training = select_topk_bottoml(t.sigmoid(logits), k, l)
    inference = select_topk_bottoml(2.0 * t.sigmoid(logits) - 1.0, k, l)

    saturated = 2.0 * t.sigmoid(logits[0, :2]) - 1.0
    assert bool((saturated == -1.0).all())  # the collapse that creates the tie
    assert t.equal(training.pos_ids, inference.pos_ids)
    assert int(training.neg_ids[0, 0]) == 1  # sigmoid still ranks -21 lowest
    assert int(inference.neg_ids[0, 0]) == 0  # tie resolves by index order


def test_compact_rollout_representative_defense_rejects_mismatched_vocab() -> None:
    model = _compact_model(feature_vocab=RAW_MASK_VOCAB, seed=1908)
    assert model.feature_pred_head is not None
    with t.no_grad():
        model.feature_pred_head.bias.zero_()
        model.feature_pred_head.bias[1] = 100.0  # compact row 1 -> original id 4, consolidated away

    with pytest.raises(FeatureClustersError, match="consolidated-away"):
        rollout(
            model,
            _compact_batch(RAW_MASK_VOCAB, seed=1909),
            _config(),
            t.Generator().manual_seed(1910),
            None,
            consolidation=CONSOLIDATION,
        )


def test_compact_step_metrics_gather_original_ids_in_compact_score_space() -> None:
    batch = _compact_batch(seed=1911)
    state = build_selfcond_state(batch, 4)
    state.sign.zero_()
    positive_ids = (COMPACT_VOCAB.original_ids[3], COMPACT_VOCAB.original_ids[9])
    state.fids[0, 1, 0] = positive_ids[0]
    state.sign[0, 1, 0] = 1.0
    state.fids[1, 1, 0] = positive_ids[1]
    state.sign[1, 1, 0] = 1.0
    result = RolloutResult(
        conditioning=state,
        predicted=t.tensor([[False, True], [False, True]]),
        k=t.tensor([[0, 1], [0, 1]]),
        l=t.zeros(2, 2, dtype=t.int64),
    )
    shifted_scores = t.zeros(2, 2, COMPACT_VOCAB.size)
    shifted_scores[0, 1, 3] = 0.25
    shifted_scores[1, 1, 9] = 0.75

    metrics = _step_metrics(
        t.tensor(1.5),
        batch,
        result,
        shifted_scores,
        feature_vocab=COMPACT_VOCAB,
    )

    assert metrics.mean_selected_pos_score == pytest.approx(0.5)
    assert metrics.frac_predicted_slots == pytest.approx(0.5)


def test_compact_step_metrics_reject_positive_id_outside_vocab() -> None:
    batch = _compact_batch(seed=1912)
    state = build_selfcond_state(batch, 4)
    state.sign.zero_()
    state.fids[0, 1, 0] = 4  # allowed by the mask but consolidated away
    state.sign[0, 1, 0] = 1.0
    result = RolloutResult(
        conditioning=state,
        predicted=t.tensor([[False, True], [False, False]]),
        k=t.tensor([[0, 1], [0, 0]]),
        l=t.zeros(2, 2, dtype=t.int64),
    )

    with pytest.raises(SelfCondTrainingError, match="positive.*compact feature vocabulary"):
        _step_metrics(
            t.tensor(1.0),
            batch,
            result,
            t.zeros(2, 2, COMPACT_VOCAB.size),
            feature_vocab=COMPACT_VOCAB,
        )


def test_compact_sft_c_matrix_accepts_mask_clusters_and_splice_policy() -> None:
    model = _compact_model(seed=1913)
    meta = _meta(
        compact=True,
        clustered=True,
        feature_delta={"mode": "splice_counterfactual"},
    )
    meta["user_config"]["feature_dir"] = "/tmp/base/splice/cut_frac0.5/seed7"
    warnings: list[str] = []
    sink = logger.add(warnings.append, level="WARNING", format="{message}")
    try:
        validate_selfcond_prerequisites(
            model,
            meta,
            _config(),
            N_SAE,
            data_source=SelfCondDataSource.SFT,
        )
    finally:
        logger.remove(sink)

    assert any("Conditioned SFT" in warning and "Proceeding" in warning for warning in warnings)


@pytest.mark.parametrize("mode", ["prev_segment", "cum_max"])
def test_compact_sft_c8_still_rejects_nonapproximable_delta_modes(mode: str) -> None:
    with pytest.raises(SelfCondTrainingError, match=rf"C8.*{mode}"):
        validate_selfcond_prerequisites(
            _compact_model(seed=1914),
            _meta(compact=True, clustered=True, feature_delta={"mode": mode}),
            _config(),
            N_SAE,
            data_source=SelfCondDataSource.SFT,
        )


def test_compact_sft_c1_requires_head_width_equal_action_space() -> None:
    model = _compact_model(seed=1915)
    model.feature_pred_head = Linear(model.dec_config.n_embd, COMPACT_VOCAB.size + 1, bias=True)

    with pytest.raises(SelfCondTrainingError, match="C1.*compact action-space width"):
        validate_selfcond_prerequisites(
            model,
            _meta(compact=True, clustered=True),
            _config(),
            N_SAE,
            data_source=SelfCondDataSource.SFT,
        )


def test_compact_pretraining_keeps_c1_and_c2_rejections_source_scoped() -> None:
    with pytest.raises(NotImplementedError) as compact_guard:
        validate_selfcond_prerequisites(
            _compact_model(seed=1916),
            _meta(compact=True, clustered=True),
            _config(),
            N_SAE,
            data_source=SelfCondDataSource.PRETRAINING,
        )
    assert str(compact_guard.value) == (
        "C1: pretraining-source self-conditioning post-training does not support a compact "
        "feature-prediction head. The checkpoint predicts compact rows whose "
        "meaning depends on its FeatureVocab, while the packed-row rollout/STE "
        "path has no compact target contract. Source-aware SFT post-training has "
        "a separate compact contract; inference-time feedback is separate too. "
        "Use an uncompacted FP checkpoint for pretraining data, or an exact "
        "compact SFT checkpoint with data_source=sft."
    )

    full_model = build_tiny_model(
        n_sae_features=N_SAE,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
        n_layer=2,
        xattn_frequency=1,
    )
    c2_meta = _meta(compact=True, clustered=False)
    c2_meta["user_config"]["use_precomputed_batches"] = True
    with pytest.raises(NotImplementedError) as mask_guard:
        validate_selfcond_prerequisites(
            full_model,
            c2_meta,
            _config(),
            N_SAE,
            data_source=SelfCondDataSource.PRETRAINING,
        )
    assert str(mask_guard.value) == (
        "C2: pretraining-source self-conditioning post-training requires an "
        "UNCOMPACTED encoder, but a "
        "feature_mask or encoder.feature_vocab_ids resolved.\n"
        "Use an unmasked checkpoint for pretraining data. Source-aware SFT "
        "post-training has a separate exact compact target contract, and "
        "inference-time compact feedback is another supported path."
    )


def test_compact_sft_action_bound_uses_vocab_size() -> None:
    cfg = _config(
        positive_max=COMPACT_VOCAB.size + 2,
        negative_max=2,
    )
    with pytest.raises(SelfCondTrainingError, match="action count.*compact action-space width"):
        validate_selfcond_prerequisites(
            _compact_model(seed=1917),
            _meta(compact=True, clustered=True),
            cfg,
            N_SAE,
            data_source=SelfCondDataSource.SFT,
        )


@pytest.mark.parametrize("ste_form", list(SteForm))
def test_compact_selfcond_loss_is_finite_and_reaches_prediction_head(
    ste_form: SteForm,
) -> None:
    model = _compact_model(seed=1918).train()
    batch = _compact_batch(seed=1919)
    cfg = _config(ste_form=ste_form)
    result = rollout(model, batch, cfg, t.Generator().manual_seed(1920), None)
    assert bool(result.predicted.any())

    model.zero_grad(set_to_none=True)
    loss, metrics = selfcond_loss(
        model,
        batch,
        result,
        cfg,
        static_max_seg_size=None,
    )
    assert bool(t.isfinite(loss))
    assert metrics.lm_loss == pytest.approx(float(loss.detach()))
    loss.backward()

    assert model.feature_pred_head is not None
    head_grads = [
        parameter.grad
        for parameter in model.feature_pred_head.parameters()
        if parameter.grad is not None
    ]
    assert len(head_grads) == len(list(model.feature_pred_head.parameters()))
    assert all(bool(t.isfinite(grad).all()) for grad in head_grads)
    assert sum(float(grad.abs().sum()) for grad in head_grads) > 0.0


@pytest.mark.parametrize("ste_form", list(SteForm))
def test_compact_encoder_accepts_compact_width_scores_in_both_forms(
    ste_form: SteForm,
) -> None:
    model = _compact_model(seed=1921)
    batch = _compact_batch(seed=1922)
    scores = t.randn(
        *batch.seg_doc_id.shape,
        COMPACT_VOCAB.size,
        generator=t.Generator().manual_seed(1923),
        requires_grad=True,
    )
    sign = t.where(batch.enc_mask, 1.0, 0.0)

    # The encoder returns a PackedEncoderOutput (values + layout), not a
    # (tensor, mask) tuple: encoder state stays packed end to end.
    baseline = model.encoder(
        batch.enc_fids,
        batch.enc_mags,
        batch.enc_mask,
    )
    actual = model.encoder(
        batch.enc_fids,
        batch.enc_mags,
        batch.enc_mask,
        ste=SteInputs(scores=scores, sign=sign, form=ste_form),
    )

    assert t.equal(actual.values, baseline.values)
    assert t.equal(actual.layout.valid_mask, baseline.layout.valid_mask)
    actual.values.float().square().mean().backward()
    assert scores.grad is not None
    assert bool(t.isfinite(scores.grad).all())
    assert float(scores.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("wrong_width", [COMPACT_VOCAB.size - 1, COMPACT_VOCAB.size + 1])
def test_compact_encoder_rejects_wrong_score_width(wrong_width: int) -> None:
    model = _compact_model(seed=1924)
    batch = _compact_batch(seed=1925)
    scores = t.zeros(*batch.seg_doc_id.shape, wrong_width, requires_grad=True)
    expected = (*batch.seg_doc_id.shape, COMPACT_VOCAB.size)

    with pytest.raises(SelfCondTrainingError) as guard:
        model.encoder(
            batch.enc_fids,
            batch.enc_mags,
            batch.enc_mask,
            ste=SteInputs(scores=scores, sign=t.zeros_like(batch.enc_mags), form=SteForm.DENSE),
        )
    assert str(guard.value) == (
        f"ste.scores shape for a compacted encoder must be {expected}, "
        f"got {tuple(scores.shape)}"
    )


def test_uncompacted_encoder_ste_width_contract_is_unchanged() -> None:
    model = build_tiny_model(
        n_sae_features=N_SAE,
        num_buckets=2,
        n_layer=2,
        xattn_frequency=1,
    )
    batch = _compact_batch(seed=1926)
    sign = t.where(batch.enc_mask, 1.0, 0.0)
    full_scores = t.zeros(*batch.seg_doc_id.shape, N_SAE, requires_grad=True)

    model.encoder(
        batch.enc_fids,
        batch.enc_mags,
        batch.enc_mask,
        ste=SteInputs(scores=full_scores, sign=sign, form=SteForm.SELECTED),
    )

    compact_scores = t.zeros(
        *batch.seg_doc_id.shape,
        COMPACT_VOCAB.size,
        requires_grad=True,
    )
    expected = (*batch.seg_doc_id.shape, N_SAE)
    with pytest.raises(SelfCondTrainingError) as guard:
        model.encoder(
            batch.enc_fids,
            batch.enc_mags,
            batch.enc_mask,
            ste=SteInputs(scores=compact_scores, sign=sign, form=SteForm.SELECTED),
        )
    assert str(guard.value) == (
        f"ste.scores shape must be {expected}, got {tuple(compact_scores.shape)}"
    )
