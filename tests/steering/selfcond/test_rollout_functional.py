"""T-F1..T-F3 plus the self-conditioning prerequisite matrix."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from errors import SelfCondTrainingError
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.feature_prediction import pool_fp_logits
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    MutableConditioning,
    RolloutResult,
    SelectedActions,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    _meta_value,
    _meta_value_null_falls_through,
    build_selfcond_state,
    continuation_mask,
    rollout,
    selfcond_loss,
    validate_selfcond_prerequisites,
    write_predicted_slot,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
    sample_k_batched,
    select_top_k_by_percentile,
)
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch, tiny_head_model


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _config(
    pool_mode: FeaturePredLossMode = FeaturePredLossMode.MAX_POOL,
    *,
    positive_max: int = 4,
    negative_max: int = 3,
) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=pool_mode,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=2,
            max_k=positive_max,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=negative_max,
        ),
        seed=29,
    )


def _predicted_width(batch: SelfcondBatch, cfg: SelfCondTrainConfig) -> int:
    assert cfg.positive_counts.max_k is not None
    assert cfg.negative_counts.max_k is not None
    return max(
        batch.enc_fids.shape[-1],
        (cfg.positive_counts.max_k - 1) + (cfg.negative_counts.max_k - 1),
    )


def _independent_rollout_reference(
    model: EncoderDecoderGPT,
    batch: SelfcondBatch,
    cfg: SelfCondTrainConfig,
    rng: t.Generator,
) -> RolloutResult:
    batch_size, n_seg = batch.seg_doc_id.shape
    state = build_selfcond_state(batch, _predicted_width(batch, cfg))
    predicted = continuation_mask(batch.seg_map, batch.seg_doc_id)
    result_k = t.zeros(batch_size, n_seg, dtype=t.int64)
    result_l = t.zeros(batch_size, n_seg, dtype=t.int64)  # noqa: E741 - spec notation
    n_sae = model.encoder.config.n_sae_features
    feature_ids = t.arange(n_sae, dtype=t.int64)
    feature_magnitudes = t.zeros(n_sae, dtype=t.float32)
    percentile_table = t.zeros(n_sae, 1, dtype=t.float64)

    for slot in range(1, n_seg):
        live = predicted[:, slot]
        if not bool(live.any()):
            continue
        assert cfg.positive_counts.max_k is not None
        assert cfg.negative_counts.max_k is not None
        positive_counts = sample_k_batched(
            cfg.positive_counts.distribution,
            cfg.positive_counts.mean,
            cfg.positive_counts.std,
            rng,
            n=batch_size,
            min_k=cfg.positive_counts.min_k,
            max_k=cfg.positive_counts.max_k,
        )
        negative_counts = sample_k_batched(
            cfg.negative_counts.distribution,
            cfg.negative_counts.mean,
            cfg.negative_counts.std,
            rng,
            n=batch_size,
            min_k=cfg.negative_counts.min_k,
            max_k=cfg.negative_counts.max_k,
        )
        positive_ids = t.full(
            (batch_size, int(positive_counts.max().item())),
            -1,
            dtype=t.int64,
        )
        negative_ids = t.full(
            (batch_size, int(negative_counts.max().item())),
            -1,
            dtype=t.int64,
        )
        for row in range(batch_size):
            if not bool(live[row]):
                continue
            _, fp_logits = model(
                batch.inputs[row : row + 1],
                None,
                state.fids[row : row + 1],
                state.mags[row : row + 1],
                state.mask[row : row + 1],
                batch.seg_map[row : row + 1],
                return_fp_logits=True,
            )
            source_tokens = batch.seg_map[row] == slot - 1
            pooled = pool_fp_logits(fp_logits[:, source_tokens], cfg.pool_mode)
            scores = pooled.float()[0]
            pos, _ = select_top_k_by_percentile(
                feature_ids,
                feature_magnitudes,
                percentile_table,
                int(positive_counts[row]),
                override_weights=scores,
                rng=None,
            )
            # Independent negative oracle (reviewer finding: the +inf-mask
            # trick mirrored the implementation's own mechanism): sort every
            # feature ascending by (score, id) in python, drop the positive
            # set, keep the first l survivors.
            positive_set = set(pos.tolist())
            ascending = sorted(
                range(n_sae),
                key=lambda fid: (float(scores[fid]), fid),
            )
            survivors = [fid for fid in ascending if fid not in positive_set]
            neg = t.tensor(survivors[: int(negative_counts[row])], dtype=t.int64)
            positive_ids[row, : pos.numel()] = pos
            negative_ids[row, : neg.numel()] = neg

        actions = SelectedActions(
            pos_ids=positive_ids,
            neg_ids=negative_ids,
            k=positive_counts,
            l=negative_counts,  # noqa: E741 - spec notation
        )
        write_predicted_slot(state, slot, actions, live)
        result_k[:, slot] = t.where(live, positive_counts, 0)
        result_l[:, slot] = t.where(live, negative_counts, 0)

    return RolloutResult(
        conditioning=state,
        predicted=predicted,
        k=result_k,
        l=result_l,  # noqa: E741 - spec notation
    )


def _assert_conditioning_equal(
    actual: MutableConditioning,
    expected: MutableConditioning,
) -> None:
    assert t.equal(actual.fids, expected.fids)
    assert t.equal(actual.mags, expected.mags)
    assert t.equal(actual.mask, expected.mask)
    assert t.equal(actual.sign, expected.sign)


def test_rollout_provenance_preserves_doc_starts_padding_and_skips_dead_steps() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=41).eval()
    batch = make_batch([[3, 1], [1, 2, 1], [2]], T=12, F=3, n_sae=16, seed=42)
    cfg = _config()
    expected_predicted = t.tensor(
        [
            [False, True, True, False],
            [False, False, True, False],
            [False, True, False, False],
        ],
        dtype=t.bool,
    )
    widened = build_selfcond_state(batch, _predicted_width(batch, cfg))
    rng = t.Generator().manual_seed(123)
    head_calls = 0

    def count_head_calls(_module, _inputs, _output) -> None:
        nonlocal head_calls
        head_calls += 1

    assert model.feature_pred_head is not None
    hook = model.feature_pred_head.register_forward_hook(count_head_calls)
    try:
        result = rollout(model, batch, cfg, rng, static_max_seg_size=None)
    finally:
        hook.remove()

    assert t.equal(result.predicted, expected_predicted)
    assert t.equal(result.k == 0, ~expected_predicted)
    assert t.equal(result.l == 0, ~expected_predicted)
    assert head_calls == 2
    for row in range(expected_predicted.shape[0]):
        for slot in range(expected_predicted.shape[1]):
            if not bool(expected_predicted[row, slot]):
                assert t.equal(result.conditioning.fids[row, slot], widened.fids[row, slot])
                assert t.equal(result.conditioning.mags[row, slot], widened.mags[row, slot])
                assert t.equal(result.conditioning.mask[row, slot], widened.mask[row, slot])
                assert t.equal(result.conditioning.sign[row, slot], widened.sign[row, slot])
    replay = t.Generator().manual_seed(123)
    assert cfg.positive_counts.max_k is not None
    assert cfg.negative_counts.max_k is not None
    for _ in range(2):
        sample_k_batched(
            cfg.positive_counts.distribution,
            cfg.positive_counts.mean,
            cfg.positive_counts.std,
            replay,
            n=3,
            min_k=cfg.positive_counts.min_k,
            max_k=cfg.positive_counts.max_k,
        )
        sample_k_batched(
            cfg.negative_counts.distribution,
            cfg.negative_counts.mean,
            cfg.negative_counts.std,
            replay,
            n=3,
            min_k=cfg.negative_counts.min_k,
            max_k=cfg.negative_counts.max_k,
        )
    assert t.equal(rng.get_state(), replay.get_state())
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize(
    "pool_mode",
    [
        FeaturePredLossMode.MAX_POOL,
        FeaturePredLossMode.FULL_SEGMENT,
        FeaturePredLossMode.LAST_TOKEN,
    ],
)
def test_rollout_actions_match_independent_per_row_reference_exactly(
    pool_mode: FeaturePredLossMode,
) -> None:
    model = tiny_head_model(pool_mode, n_sae=16, seed=43).eval()
    batch = make_batch([[3], [2, 1]], T=12, F=3, n_sae=16, seed=44)
    cfg = _config(pool_mode)

    actual = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(567),
        static_max_seg_size=None,
    )
    expected = _independent_rollout_reference(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(567),
    )

    assert t.equal(actual.predicted, expected.predicted)
    assert t.equal(actual.k, expected.k)
    assert t.equal(actual.l, expected.l)
    _assert_conditioning_equal(actual.conditioning, expected.conditioning)


def test_rollout_ranks_raw_logits_across_both_sigmoid_saturation_sides() -> None:
    batch = make_batch([[2]], T=4, F=2, n_sae=4, seed=1449)
    cfg = SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        seed=1449,
    )
    fixed_logits = t.tensor([-20.0, -21.0, 20.0, 21.0]).view(1, 1, 4)

    class FixedLogitModel:
        # Rollout reads ``model.feature_vocab`` to decide whether action ids
        # need compact->original translation; an uncompacted stand-in is None.
        feature_vocab = None

        def __call__(self, inputs: t.Tensor, *_args, **_kwargs) -> t.Tensor:
            return fixed_logits.expand(inputs.shape[0], inputs.shape[1], -1)

    result = rollout(
        FixedLogitModel(),
        batch,
        cfg,
        t.Generator().manual_seed(1449),
        static_max_seg_size=None,
    )

    assert result.conditioning.fids[0, 1, :2].tolist() == [3, 1]
    assert result.conditioning.sign[0, 1, :2].tolist() == [1, -1]


def test_step_scores_are_bitwise_invariant_to_future_slot_garbage() -> None:
    """R1, swept over EVERY source slot (reviewer finding: anchoring only at
    slot 0 leaves later steps' isolation unpinned)."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=45).eval()
    batch = make_batch([[4], [4]], T=12, F=4, n_sae=16, seed=46)
    n_seg = batch.seg_doc_id.shape[1]
    clean = build_selfcond_state(batch, 5)

    def scores_at(state: MutableConditioning, source_slot: int) -> t.Tensor:
        _, fp_logits = model(
            batch.inputs,
            None,
            state.fids,
            state.mags,
            state.mask,
            batch.seg_map,
            return_fp_logits=True,
        )
        pooled = pool_fp_logits_per_segment_reference(
            fp_logits,
            batch.seg_map,
            n_seg,
        )
        return pooled[:, source_slot].float()

    generator = t.Generator().manual_seed(47)
    for step in range(1, n_seg):
        source_slot = step - 1
        garbage = copy.deepcopy(clean)
        garbage.fids[:, step:] = t.randint(0, 16, garbage.fids[:, step:].shape, generator=generator)
        garbage.mags[:, step:] = t.randint(0, 2, garbage.mags[:, step:].shape, generator=generator)
        garbage.mask[:, step:] = t.rand(garbage.mask[:, step:].shape, generator=generator) > 0.3
        assert t.equal(scores_at(clean, source_slot), scores_at(garbage, source_slot)), (
            f"slot-{source_slot} scores changed when slots >= {step} were garbaged"
        )


def pool_fp_logits_per_segment_reference(
    fp_logits: t.Tensor,
    seg_map: t.Tensor,
    n_seg: int,
) -> t.Tensor:
    rows: list[t.Tensor] = []
    for row in range(fp_logits.shape[0]):
        segments: list[t.Tensor] = []
        for slot in range(n_seg):
            segments.append(
                pool_fp_logits(
                    fp_logits[row : row + 1, seg_map[row] == slot],
                    FeaturePredLossMode.MAX_POOL,
                )[0]
            )
        rows.append(t.stack(segments))
    return t.stack(rows)


def _valid_meta() -> dict:
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
                "max_k": 4,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "min_k": 1,
                "max_k": 3,
            },
            "num_buckets": 2,
            "feature_mask": None,
            "feature_clusters": None,
            "feature_delta": None,
            "label_dropout": 0.0,
            "next_segment_conditioning_offset": False,
            "use_precomputed_batches": True,
            "feature_dir": "/tmp/plain-features",
        }
    }


def test_valid_prerequisite_matrix_passes() -> None:
    validate_selfcond_prerequisites(
        tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=50),
        _valid_meta(),
        _config(),
        16,
    )


def test_c8_uses_top_level_feature_delta_when_user_config_omits_it() -> None:
    meta = _valid_meta()
    del meta["user_config"]["feature_delta"]
    meta["feature_delta"] = {"mode": "splice_counterfactual"}

    with pytest.raises(SelfCondTrainingError, match="C8"):
        validate_selfcond_prerequisites(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=51),
            meta,
            _config(),
            16,
        )


def test_meta_value_variants_split_strict_guards_from_null_fallthrough() -> None:
    """The guard reader must NOT resolve an explicit null to a permissive
    default (review finding: null -> use_precomputed_batches' True default
    would silently disarm C6); only the structural-fact reader falls through."""
    meta = {"user_config": {"num_buckets": None}, "num_buckets": 2}
    assert _meta_value(meta, "num_buckets") is None
    assert _meta_value_null_falls_through(meta, "num_buckets") == 2

    explicit = {"user_config": {"num_buckets": 4}, "num_buckets": 2}
    assert _meta_value(explicit, "num_buckets") == 4
    assert _meta_value_null_falls_through(explicit, "num_buckets") == 4
    assert _meta_value({"user_config": {}}, "absent", "default") == "default"

    # C6's fail-closed direction: an explicitly recorded null must still raise.
    guard_meta = {"user_config": {"use_precomputed_batches": None}}
    assert not bool(_meta_value(guard_meta, "use_precomputed_batches", True))


def test_c1_head_and_recoverable_config_are_required() -> None:
    headless = build_tiny_model(n_sae_features=16, num_buckets=2, feature_pred_config=None)
    with pytest.raises(SelfCondTrainingError, match="C1.*feature-prediction head"):
        validate_selfcond_prerequisites(headless, _valid_meta(), _config(), 16)

    meta = _valid_meta()
    del meta["user_config"]["feature_prediction"]
    with pytest.raises(SelfCondTrainingError, match="C1.*metadata"):
        validate_selfcond_prerequisites(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=51),
            meta,
            _config(),
            16,
        )


def test_c2_compacted_encoder_uses_existing_not_implemented_contract() -> None:
    template = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=52)
    compacted = EncoderDecoderGPT(
        template.dec_config,
        template.enc_config,
        template.xattn_config,
        n_sae_features=16,
        num_buckets=2,
        feature_pred_config=template.feature_pred_config,
        feature_vocab=FeatureVocab(n_sae_features=16, original_ids=(0, 1, 2, 3)),
    )
    compacted.init_weights()
    with pytest.raises(NotImplementedError, match="compact feature-prediction"):
        validate_selfcond_prerequisites(compacted, _valid_meta(), _config(), 16)

    meta = _valid_meta()
    meta["user_config"]["feature_mask"] = {"mode": "blacklist", "ids": [15]}
    with pytest.raises(NotImplementedError, match="C2.*feature_mask"):
        validate_selfcond_prerequisites(template, meta, _config(), 16)


@pytest.mark.parametrize(
    ("constraint", "mutation", "match"),
    [
        ("C3", lambda meta: meta["user_config"].update(num_buckets=3), "C3.*num_buckets"),
        (
            "C4",
            lambda meta: meta["user_config"].update(negative_selection=None),
            "C4.*negative_selection",
        ),
        (
            "C5-clusters",
            lambda meta: meta["user_config"].update(feature_clusters={"path": "x.yaml"}),
            "C5.*feature_clusters",
        ),
        (
            "C5-gradient",
            lambda meta: meta["user_config"]["feature_selection"].update(
                gradient={"mode": "pct_grad_abs_max"}
            ),
            "C5.*gradient",
        ),
        (
            "C5-dropout",
            lambda meta: meta["user_config"].update(label_dropout=0.2),
            "C5.*label_dropout",
        ),
        (
            "C5-offset",
            lambda meta: meta["user_config"].update(next_segment_conditioning_offset=True),
            "C5.*next_segment_conditioning_offset",
        ),
        (
            "C6",
            lambda meta: meta["user_config"].update(use_precomputed_batches=False),
            "C6.*precomputed",
        ),
        (
            "C8-delta",
            lambda meta: meta["user_config"].update(
                feature_delta={"mode": "splice_counterfactual", "neg_delta_ratio": 1.0}
            ),
            "C8.*splice_counterfactual",
        ),
        (
            "C8-path",
            lambda meta: meta["user_config"].update(
                feature_dir="/tmp/base/splice/cut_frac0.5/seed7"
            ),
            "C8.*spliced",
        ),
    ],
)
def test_prerequisite_matrix_rejects_each_meta_constraint(
    constraint: str,
    mutation: Callable[[dict], None],
    match: str,
) -> None:
    meta = _valid_meta()
    mutation(meta)
    with pytest.raises(SelfCondTrainingError, match=match) as guard:
        validate_selfcond_prerequisites(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=53),
            meta,
            _config(),
            16,
        )
    assert constraint.split("-", maxsplit=1)[0] in str(guard.value)


def test_c7_best_token_policy_is_rejected() -> None:
    cfg = _config()
    object.__setattr__(cfg, "pool_mode", FeaturePredLossMode.BEST_TOKEN)
    with pytest.raises(SelfCondTrainingError, match="C7.*best_token"):
        validate_selfcond_prerequisites(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=54),
            _valid_meta(),
            cfg,
            16,
        )


def test_count_sum_must_fit_uncompacted_feature_universe() -> None:
    cfg = _config(positive_max=11, negative_max=9)
    with pytest.raises(SelfCondTrainingError, match="count.*n_sae_features"):
        validate_selfcond_prerequisites(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=55),
            _valid_meta(),
            cfg,
            16,
        )


def test_two_pass_scores_and_ce_are_bitwise_equal_to_existing_paths() -> None:
    """T-F4: pass-1 and encoder-out pass-2 preserve the existing path values."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=60).eval()
    batch = make_batch([[3], [2, 1]], T=12, F=4, n_sae=16, seed=61)
    cfg = _config()
    rollout_logits: list[t.Tensor] = []

    def capture_rollout(_module, _inputs, output: t.Tensor) -> None:
        rollout_logits.append(output.detach().clone())

    assert model.feature_pred_head is not None
    handle = model.feature_pred_head.register_forward_hook(capture_rollout)
    try:
        result = rollout(
            model,
            batch,
            cfg,
            t.Generator().manual_seed(62),
            static_max_seg_size=None,
        )
    finally:
        handle.remove()

    pass_one_logits: list[t.Tensor] = []

    def capture_pass_one(_module, _inputs, output: t.Tensor) -> None:
        pass_one_logits.append(output.detach().clone())

    handle = model.feature_pred_head.register_forward_hook(capture_pass_one)
    try:
        loss, _ = selfcond_loss(model, batch, result, cfg, static_max_seg_size=None)
    finally:
        handle.remove()

    assert len(rollout_logits) == 2
    assert len(pass_one_logits) == 1
    final_pooled = pool_fp_logits_per_segment_reference(
        pass_one_logits[0],
        batch.seg_map,
        batch.seg_doc_id.shape[1],
    )
    for source_slot, step_logits in enumerate(rollout_logits):
        step_pooled = pool_fp_logits_per_segment_reference(
            step_logits,
            batch.seg_map,
            batch.seg_doc_id.shape[1],
        )
        live = result.predicted[:, source_slot + 1]
        assert t.equal(
            t.sigmoid(final_pooled[live, source_slot].float()),
            t.sigmoid(step_pooled[live, source_slot].float()),
        )

    plain_loss = model(
        batch.inputs,
        batch.targets,
        result.conditioning.fids,
        result.conditioning.mags,
        result.conditioning.mask,
        batch.seg_map,
        static_max_seg_size=None,
    )
    assert t.equal(loss, plain_loss)


def test_zero_predicted_slots_equal_plain_pretraining_ce_bitwise() -> None:
    """T-F5: an all-doc-start batch has no value or STE effect."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=63).eval()
    batch = make_batch([[1], [1]], T=8, F=4, n_sae=16, seed=64)
    cfg = _config()
    result = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(65),
        static_max_seg_size=None,
    )
    assert not bool(result.predicted.any())
    assert not bool(result.conditioning.sign.any())

    loss, _ = selfcond_loss(model, batch, result, cfg, static_max_seg_size=None)
    plain_loss = model(
        batch.inputs,
        batch.targets,
        result.conditioning.fids,
        result.conditioning.mags,
        result.conditioning.mask,
        batch.seg_map,
        static_max_seg_size=None,
    )
    assert t.equal(loss, plain_loss)


def test_identical_runs_produce_bitwise_actions_loss_and_gradients() -> None:
    """T-F6: the complete rollout and two-pass estimator are deterministic."""
    first = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=66).eval()
    second = copy.deepcopy(first)
    batch = make_batch([[3], [2, 1]], T=12, F=4, n_sae=16, seed=67)
    cfg = _config()

    first_result = rollout(
        first,
        batch,
        cfg,
        t.Generator().manual_seed(68),
        static_max_seg_size=None,
    )
    second_result = rollout(
        second,
        batch,
        cfg,
        t.Generator().manual_seed(68),
        static_max_seg_size=None,
    )
    first_loss, _ = selfcond_loss(
        first,
        batch,
        first_result,
        cfg,
        static_max_seg_size=None,
    )
    second_loss, _ = selfcond_loss(
        second,
        batch,
        second_result,
        cfg,
        static_max_seg_size=None,
    )
    first_loss.backward()
    second_loss.backward()

    assert t.equal(first_result.predicted, second_result.predicted)
    assert t.equal(first_result.k, second_result.k)
    assert t.equal(first_result.l, second_result.l)
    _assert_conditioning_equal(first_result.conditioning, second_result.conditioning)
    assert t.equal(first_loss, second_loss)
    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters(), strict=True
    ):
        if first_parameter.grad is None or second_parameter.grad is None:
            assert first_parameter.grad is None and second_parameter.grad is None
        else:
            assert t.equal(first_parameter.grad, second_parameter.grad)


def test_truncated_tail_document_rolls_out_normally() -> None:
    """T-F1 (strengthened): a document whose FINAL segment is cut to a single
    token (the packed-row row-tail truncation shape) still classifies its
    continuation slots identically and trains without special-casing."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=48).eval()
    batch = make_batch([[3]], T=12, F=4, n_sae=16, seed=49)
    # Surgically truncate: final segment covers ONLY the last token.
    batch.seg_map[0] = t.tensor([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2], dtype=t.int64)
    live = continuation_mask(batch.seg_map, batch.seg_doc_id)
    assert live[0].tolist() == [False, True, True]

    cfg = _config()
    result = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(50),
        static_max_seg_size=None,
    )
    assert result.predicted[0].tolist() == [False, True, True]
    loss, _ = selfcond_loss(model, batch, result, cfg, static_max_seg_size=None)
    assert bool(t.isfinite(loss))
