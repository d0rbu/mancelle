"""SFT target-only segment semantics for self-conditioning rollout."""

from __future__ import annotations

import copy
import math
import random
from pathlib import Path

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from errors import DataLoadError
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import OptimizerKind
from scripts.train.conditional_encdec_selfcond import (
    SelfCondSftData,
    SelfCondSftSources,
    build_selfcond_sft_data,
    run_selfcond_train_step,
)
from scripts.train.conditional_encdec_sft import (
    SFTState,
    make_sft_data_generator,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.feature_prediction import segment_lengths
from steering.conditional.encoder_decoder.selfcond_training import (
    RolloutResult,
    SelfCondBatch,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    _step_metrics,
    build_selfcond_state,
    continuation_mask,
    freeze_for_head_only,
    lm_conditioned_slot_mask,
    rollout,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
)
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft_schema import SFTSplit
from steering.stats_io import save_tokenizer_checksum
from tasks.common import Task
from tasks.mixtures import SFTMixture
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.steering.selfcond.helpers import make_batch, tiny_head_model
from tests.steering.test_encdec_sft_conditioned_integration import (
    MultiTurnTask,
    SFTMockTokenizer,
    SingleTurnTask,
    _make_boundaries,
    build_synthetic_features,
    patch_load_sft_features,
)

N_SAE_FEATURES = 64
NUM_BUCKETS = 2
TOKENIZER_CHECKSUM = "selfcond-sft-test-tokenizer"


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_fa_mod, "_override_impl", FlashAttnImpl.SDPA)


def _config() -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
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
            min_k=0,
            max_k=1,
        ),
        seed=1438,
    )


def _target_only_batch() -> SelfCondBatch:
    inputs = t.tensor([[2, 3, 5, 7, 11, 13, 17, 19]], dtype=t.int32)
    targets = t.tensor([[-1, -1, 7, 11, 13, 17, 19, 23]], dtype=t.int64)
    enc_fids = t.tensor(
        [
            [
                [0, 0, 0],
                [1, 2, 3],
                [4, 5, 6],
                [7, 6, 5],
                [2, 1, 0],
            ]
        ],
        dtype=t.int64,
    )
    enc_mags = t.tensor(
        [
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
                [0, 1, 0],
            ]
        ],
        dtype=t.int64,
    )
    enc_mask = t.tensor(
        [
            [
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ]
        ],
        dtype=t.bool,
    )
    seg_map = t.tensor([[0, 0, 1, 1, 2, 2, 4, 4]], dtype=t.int64)
    seg_doc_id = t.tensor([[-1, 0, 0, 0, 1]], dtype=t.int64)
    sft_batch = (
        inputs,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        None,
        seg_doc_id,
    )
    return SelfCondBatch.from_sft(sft_batch)


def _assert_rollouts_equal(first, second) -> None:
    assert t.equal(first.predicted, second.predicted)
    assert t.equal(first.k, second.k)
    assert t.equal(first.l, second.l)
    assert t.equal(first.conditioning.fids, second.conditioning.fids)
    assert t.equal(first.conditioning.mags, second.conditioning.mags)
    assert t.equal(first.conditioning.mask, second.conditioning.mask)
    assert t.equal(first.conditioning.sign, second.conditioning.sign)


def _sampling_identity() -> dict[str, object]:
    return {
        "feature_selection": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 1,
            "max_k": 4,
            "mode": "top_k_percentile",
            "delta_weights": None,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 0,
            "max_k": 3,
        },
    }


def _sft_sources(feature_dir: Path) -> SelfCondSftSources:
    return SelfCondSftSources(
        dataset=SFTMixture.SAFETY,
        train_parquet_dir="/synthetic/safety",
        train_feature_dir=str(feature_dir),
        val_parquet_dir=None,
        val_feature_dir=None,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        feature_source_bos_char_len=0,
        alignment_probe_samples=0,
        segment_mode=SegmentMode.fixed_tokens,
    )


def _build_test_sft_data(tmp_path: Path, tokenizer: SFTMockTokenizer) -> SelfCondSftData:
    feature_dir = tmp_path / "features"
    stats_path = write_steering_stats(
        feature_dir / "feature_statistics.safetensors",
        n_sae_features=N_SAE_FEATURES,
    )
    save_tokenizer_checksum(stats_path, TOKENIZER_CHECKSUM)
    return build_selfcond_sft_data(
        sources=_sft_sources(feature_dir),
        identity=_sampling_identity(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        ddp_rank=0,
        tokenizer=tokenizer,
    )


def _real_sft_generator(
    *,
    task: Task,
    tokenizer: SFTMockTokenizer,
    data: SelfCondSftData,
    state: SFTState,
    batch_size: int,
    sequence_len: int,
    buffer_size: int = 100,
):
    assert data.pipeline.negative_bundle is not None
    return make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=batch_size,
        T=sequence_len,
        device="cpu",
        sft_state=state,
        ddp_rank=0,
        ddp_world_size=1,
        buffer_size=buffer_size,
        feature_dir="/synthetic/features",
        parquet_dir="/synthetic/parquet",
        threshold_cfg=data.threshold_cfg,
        boundaries=data.boundaries,
        precomputed_thresholds=data.precomputed_thresholds,
        sampling_bundle=data.sampling_bundle,
        min_loss_tokens=1,
        fp_percentile_table=None,
        negative_bundle=data.pipeline.negative_bundle,
        return_seg_doc_id=True,
        feature_source_bos_char_len=0,
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        alignment_probe_samples=0,
        mixture_name=SFTMixture.SAFETY.value,
        single_pass=False,
        expected_split=SFTSplit.TRAIN,
    )


def _optimizer_for_scope(
    model: t.nn.Module,
    scope: SelfCondTrainScope,
) -> t.optim.AdamW:
    if scope is SelfCondTrainScope.HEAD_ONLY:
        freeze_for_head_only(model)
    optimizer = t.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.002,
        weight_decay=0.0,
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["kind"] = OptimizerKind.ADAMW
    return optimizer


def test_gate_is_identity_when_every_real_segment_has_tokens() -> None:
    layouts = [
        [[1]],
        [[3]],
        [[1, 1, 1]],
        [[3, 1], [1]],
        [[1], [2, 1], [1, 1]],
    ]
    rng = random.Random(1438)
    for _ in range(50):
        batch_layout: list[list[int]] = []
        time = rng.randint(8, 24)
        for _row in range(rng.randint(1, 4)):
            remaining = time
            documents: list[int] = []
            for _document in range(rng.randint(1, 4)):
                if remaining == 0:
                    break
                segment_count = rng.randint(1, remaining)
                documents.append(segment_count)
                remaining -= segment_count
            batch_layout.append(documents)
        layouts.append(batch_layout)

    for index, layout in enumerate(layouts):
        time = max(24, max(sum(row) for row in layout))
        batch = make_batch(layout, T=time, F=4, n_sae=8, seed=2000 + index)
        lengths = segment_lengths(batch.seg_map, batch.seg_doc_id.shape[1])
        assert bool((lengths[batch.seg_doc_id >= 0] > 0).all())
        assert t.equal(
            lm_conditioned_slot_mask(batch.seg_map, batch.seg_doc_id),
            continuation_mask(batch.seg_map, batch.seg_doc_id),
        )

    empty_seg_map = t.zeros(2, 4, dtype=t.int64)
    empty_seg_doc_id = t.zeros(2, 0, dtype=t.int64)
    assert t.equal(
        lm_conditioned_slot_mask(empty_seg_map, empty_seg_doc_id),
        continuation_mask(empty_seg_map, empty_seg_doc_id),
    )


def test_token_free_target_only_slot_is_never_predicted() -> None:
    seg_map = t.tensor([[0, 0, 1, 1, 2, 2, 4, 4]], dtype=t.int64)
    seg_doc_id = t.tensor([[-1, 0, 0, 0, 1]], dtype=t.int64)

    live = lm_conditioned_slot_mask(seg_map, seg_doc_id)

    assert live.tolist() == [[False, False, True, False, False]]
    assert bool(continuation_mask(seg_map, seg_doc_id)[0, 3].item())
    assert segment_lengths(seg_map, 5).tolist() == [[2.0, 2.0, 2.0, 0.0, 2.0]]


def test_rollout_never_writes_a_token_free_slot_and_never_crosses_documents() -> None:
    batch = _target_only_batch()
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=401)
    model.eval()
    cfg = _config()
    original = build_selfcond_state(batch, batch.enc_fids.shape[-1])

    result = rollout(model, batch, cfg, t.Generator().manual_seed(402), None)

    assert result.predicted.tolist() == [[False, False, True, False, False]]
    for slot in (0, 1, 3, 4):
        assert t.equal(result.conditioning.fids[:, slot], original.fids[:, slot])
        assert t.equal(result.conditioning.mags[:, slot], original.mags[:, slot])
        assert t.equal(result.conditioning.mask[:, slot], original.mask[:, slot])
        assert t.equal(result.conditioning.sign[:, slot], original.sign[:, slot])
        assert result.k[0, slot].item() == 0
        assert result.l[0, slot].item() == 0
    assert bool((result.conditioning.sign[:, 2] != 0).any())
    assert not bool((result.conditioning.sign[:, (0, 1, 3, 4)] != 0).any())
    assert result.k[0, 2].item() == 1
    assert result.l[0, 2].item() == 0


def test_lm_loss_is_bitwise_invariant_to_target_only_slot_conditioning() -> None:
    batch = _target_only_batch()
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=411)
    model.eval()
    changed_fids = batch.enc_fids.clone()
    changed_mags = batch.enc_mags.clone()
    changed_mask = batch.enc_mask.clone()
    changed_fids[:, 3].zero_()
    changed_mags[:, 3].zero_()
    changed_mask[:, 3].zero_()

    with t.no_grad():
        original_loss = model(
            batch.inputs,
            batch.targets,
            batch.enc_fids,
            batch.enc_mags,
            batch.enc_mask,
            batch.seg_map,
            loss_reduction="none",
        )
        changed_loss = model(
            batch.inputs,
            batch.targets,
            changed_fids,
            changed_mags,
            changed_mask,
            batch.seg_map,
            loss_reduction="none",
        )

    assert t.equal(original_loss, changed_loss)


def test_step_metrics_denominator_counts_only_token_bearing_real_slots() -> None:
    batch = _target_only_batch()
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=421)
    model.eval()
    result = rollout(model, batch, _config(), t.Generator().manual_seed(422), None)

    metrics = _step_metrics(t.tensor(1.25), batch, result, None)

    token_bearing_real_slots = (batch.seg_doc_id >= 0) & (
        segment_lengths(batch.seg_map, batch.seg_doc_id.shape[1]) > 0
    )
    assert token_bearing_real_slots.tolist() == [[False, True, True, False, True]]
    assert int(result.predicted.sum().item()) == 1
    assert metrics.frac_predicted_slots == pytest.approx(1 / 3)
    assert metrics.mean_k == pytest.approx(1.0)
    assert metrics.mean_l == pytest.approx(0.0)


def test_step_metrics_zero_segment_batch_has_zero_predicted_fraction() -> None:
    batch_size, sequence_len, feature_width = 2, 4, 3
    batch = SelfCondBatch(
        inputs=t.zeros(batch_size, sequence_len, dtype=t.int32),
        targets=t.full((batch_size, sequence_len), -1, dtype=t.int64),
        enc_fids=t.zeros(batch_size, 0, feature_width, dtype=t.int64),
        enc_mags=t.zeros(batch_size, 0, feature_width, dtype=t.int64),
        enc_mask=t.zeros(batch_size, 0, feature_width, dtype=t.bool),
        seg_map=t.zeros(batch_size, sequence_len, dtype=t.int64),
        seg_doc_id=t.zeros(batch_size, 0, dtype=t.int64),
    )
    empty_shape = (batch_size, 0)
    result = RolloutResult(
        conditioning=build_selfcond_state(batch, feature_width),
        predicted=t.zeros(empty_shape, dtype=t.bool),
        k=t.zeros(empty_shape, dtype=t.int64),
        l=t.zeros(empty_shape, dtype=t.int64),
    )

    metrics = _step_metrics(t.tensor(1.25), batch, result, None)

    assert metrics.frac_predicted_slots == 0.0
    assert metrics.mean_k == 0.0
    assert metrics.mean_l == 0.0


def test_rollout_determinism_on_sft_shaped_batches() -> None:
    batch = _target_only_batch()
    first_model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=431)
    second_model = copy.deepcopy(first_model)
    first_model.eval()
    second_model.eval()
    cfg = _config()

    first = rollout(first_model, batch, cfg, t.Generator().manual_seed(432), None)
    second = rollout(second_model, batch, cfg, t.Generator().manual_seed(432), None)

    _assert_rollouts_equal(first, second)


@pytest.mark.parametrize("scope", list(SelfCondTrainScope))
@pytest.mark.parametrize("ste_form", list(SteForm))
def test_generator_batches_flow_through_adapter_and_train_step(
    tmp_path: Path,
    scope: SelfCondTrainScope,
    ste_form: SteForm,
) -> None:
    """The real SFT generator trains every supported scope/form combination."""
    tokenizer = SFTMockTokenizer()
    task = MultiTurnTask(n=8)
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        n_sae_features=N_SAE_FEATURES,
    )
    sft_data = _build_test_sft_data(tmp_path, tokenizer)

    assert isinstance(sft_data.sampling_bundle, SamplingBundle)
    assert sft_data.threshold_cfg is None
    assert sft_data.precomputed_thresholds is None
    assert sft_data.pipeline.negative_bundle is not None
    assert sft_data.sampling_bundle.rng is sft_data.pipeline.negative_bundle.rng
    assert t.equal(
        sft_data.boundaries,
        _make_boundaries(n_sae_features=N_SAE_FEATURES, num_buckets=1),
    )
    assert sft_data.tokenizer_checksum == TOKENIZER_CHECKSUM

    state = SFTState()
    with patch_load_sft_features(features):
        raw = next(
            _real_sft_generator(
                task=task,
                tokenizer=tokenizer,
                data=sft_data,
                state=state,
                batch_size=2,
                sequence_len=64,
            )
        )
    batch = SelfCondBatch.from_sft(raw)
    assert batch.inputs.dtype is t.int32
    assert batch.targets.dtype is t.int64
    assert batch.seg_doc_id.dtype is t.int64
    assert bool(((batch.enc_mags == 1) & batch.enc_mask).any())
    assert state.consumed > 0

    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=610,
    )
    optimizer = _optimizer_for_scope(model, scope)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        SelfCondTrainConfig(
            pool_mode=FeaturePredLossMode.MAX_POOL,
            train_scope=scope,
            ste_form=ste_form,
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
                min_k=0,
                max_k=1,
            ),
            seed=611,
        ),
        global_step=0,
        ddp_rank=0,
        num_iterations=PositiveInt.parse(1),
        grad_clip_max_norm=None,
        static_max_seg_size=None,
    )

    assert math.isfinite(result.loss)
    assert result.metrics.frac_predicted_slots > 0.0
    changed = {
        name for name, parameter in model.named_parameters() if not t.equal(parameter, before[name])
    }
    assert changed
    if scope is SelfCondTrainScope.HEAD_ONLY:
        assert all(name.startswith("feature_pred_head.") for name in changed)


def test_target_only_segments_flow_end_to_end_under_forced_truncation(
    tmp_path: Path,
) -> None:
    tokenizer = SFTMockTokenizer()
    task = MultiTurnTask(n=4, user_chars=3, asst_chars=6)
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=4,
        n_sae_features=N_SAE_FEATURES,
    )
    sft_data = _build_test_sft_data(tmp_path, tokenizer)
    state = SFTState()
    with patch_load_sft_features(features):
        raw = next(
            _real_sft_generator(
                task=task,
                tokenizer=tokenizer,
                data=sft_data,
                state=state,
                batch_size=1,
                sequence_len=12,
                buffer_size=1,
            )
        )
    batch = SelfCondBatch.from_sft(raw)
    lengths = segment_lengths(batch.seg_map, batch.seg_doc_id.shape[1])
    token_free_real = (batch.seg_doc_id >= 0) & (lengths == 0)
    assert bool(token_free_real.any()), "forced truncation must publish a target-only slot"

    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=620,
    )
    optimizer = _optimizer_for_scope(model, SelfCondTrainScope.HEAD_ONLY)
    cfg = _config()
    rollout_result = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(621),
        None,
    )
    assert not bool(rollout_result.predicted[token_free_real].any())
    assert bool(rollout_result.predicted.any())

    train_result = run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        cfg,
        global_step=0,
        ddp_rank=0,
        num_iterations=PositiveInt.parse(1),
        grad_clip_max_norm=None,
        static_max_seg_size=None,
    )
    assert math.isfinite(train_result.loss)
    assert math.isfinite(train_result.metrics.mean_k)
    assert train_result.metrics.frac_predicted_slots > 0.0


class _RecordingTask(Task):
    def __init__(self, inner: Task) -> None:
        super().__init__()
        self.inner = inner
        self.accessed: list[int] = []

    def num_examples(self) -> int:
        return len(self.inner)

    def get_example(self, index: int) -> dict:
        self.accessed.append(index)
        return self.inner[index]


def test_sft_consumed_advances_only_on_pop_and_seeds_cursor(tmp_path: Path) -> None:
    tokenizer = SFTMockTokenizer()
    inner = SingleTurnTask(n=8, user_chars=4, asst_chars=8)
    task = _RecordingTask(inner)
    features = build_synthetic_features(
        inner,
        tokenizer,
        n_segments_per_conv=2,
        n_sae_features=N_SAE_FEATURES,
    )
    sft_data = _build_test_sft_data(tmp_path, tokenizer)
    state = SFTState()
    generator = _real_sft_generator(
        task=task,
        tokenizer=tokenizer,
        data=sft_data,
        state=state,
        batch_size=1,
        sequence_len=14,
        buffer_size=1,
    )

    # The cursor is seeded lazily: nothing is read until a batch is pulled, and
    # ``consumed`` advances only when a conversation is actually popped.
    assert task.accessed == []
    assert state.consumed == 0
    with patch_load_sft_features(features):
        raw = next(generator)

    batch = SelfCondBatch.from_sft(raw)
    assert batch.inputs.shape == (1, 14)
    assert task.accessed == [0]
    assert state.consumed == 1


def test_sft_resume_rejects_consumed_without_buffered_dataloader_state(
    tmp_path: Path,
) -> None:
    """A non-zero cursor alone is not resumable under ``buffered_v2``.

    The committed cursor says how many conversations were consumed but not
    which ones were mid-buffer, so resuming from it would silently repeat or
    skip rows. The generator refuses rather than guess.
    """
    tokenizer = SFTMockTokenizer()
    inner = SingleTurnTask(n=8, user_chars=4, asst_chars=8)
    task = _RecordingTask(inner)
    features = build_synthetic_features(
        inner,
        tokenizer,
        n_segments_per_conv=2,
        n_sae_features=N_SAE_FEATURES,
    )
    sft_data = _build_test_sft_data(tmp_path, tokenizer)
    generator = _real_sft_generator(
        task=task,
        tokenizer=tokenizer,
        data=sft_data,
        state=SFTState(consumed=3),
        batch_size=1,
        sequence_len=14,
        buffer_size=1,
    )

    with patch_load_sft_features(features):
        with pytest.raises(DataLoadError, match="buffered_v2 dataloader state"):
            next(generator)
