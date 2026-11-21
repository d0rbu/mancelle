"""Deterministic SFT validation contracts for self-conditioning."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_selfcond import (
    SelfCondSftData,
    SelfCondSftEvalConfig,
    SelfCondSftSources,
    build_selfcond_sft_data,
    clone_val_bundles,
    evaluate_selfcond_bpb_sft,
    run_sft_validation_cycle,
)
from scripts.train.conditional_encdec_sft import (
    _build_sft_val_loader,
    _SFTValidationLoader,
    evaluate_encdec_bpb,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.loss_eval import (
    accumulate_bpb_nats_bytes,
    reduce_bpb,
)
from steering.conditional.encoder_decoder.selfcond_training import (
    VAL_DATA_SEED_SENTINEL,
    VAL_ROLLOUT_STEP_SENTINEL,
    SelfCondBatch,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    rollout,
    selfcond_step_seed,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
)
from steering.segmentation import SegmentMode
from steering.sft.loss_mask import LossMaskMode
from steering.sft_schema import SFTSplit
from steering.stats_io import save_tokenizer_checksum
from tasks.common import Task
from tasks.mixtures import SFTMixture
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.encdec_test_helpers import VOCAB_SIZE
from tests.steering.selfcond.helpers import tiny_head_model
from tests.steering.test_encdec_sft_conditioned_integration import (
    MultiTurnTask,
    SFTMockTokenizer,
    build_synthetic_features,
    patch_load_sft_features,
)

N_SAE_FEATURES = 64
NUM_BUCKETS = 2
TOKENIZER_CHECKSUM = "selfcond-sft-validation-tokenizer"


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_fa_mod, "_override_impl", FlashAttnImpl.SDPA)


def _selfcond_config(*, seed: int = 730) -> SelfCondTrainConfig:
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
        seed=seed,
    )


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


def _sources(feature_dir: Path) -> SelfCondSftSources:
    return SelfCondSftSources(
        dataset=SFTMixture.SAFETY,
        train_parquet_dir="/synthetic/safety",
        train_feature_dir=str(feature_dir),
        val_parquet_dir="/synthetic/safety_val",
        val_feature_dir="/synthetic/features/safety_sft_val",
        loss_mask_mode=LossMaskMode.FULL_SEQUENCE,
        min_loss_tokens=1,
        feature_source_bos_char_len=0,
        alignment_probe_samples=0,
        segment_mode=SegmentMode.fixed_tokens,
    )


@dataclass(frozen=True)
class _ValidationCase:
    tokenizer: SFTMockTokenizer
    task: Task
    features: dict
    sources: SelfCondSftSources
    data: SelfCondSftData
    eval_cfg: SelfCondSftEvalConfig
    selfcond_cfg: SelfCondTrainConfig
    token_bytes: t.Tensor


@pytest.fixture
def validation_case(tmp_path: Path) -> _ValidationCase:
    tokenizer = SFTMockTokenizer()
    task = MultiTurnTask(n=6)
    features = build_synthetic_features(
        task,
        tokenizer,
        n_segments_per_conv=2,
        n_sae_features=N_SAE_FEATURES,
    )
    feature_dir = tmp_path / "train_features"
    stats_path = write_steering_stats(
        feature_dir / "feature_statistics.safetensors",
        n_sae_features=N_SAE_FEATURES,
    )
    save_tokenizer_checksum(stats_path, TOKENIZER_CHECKSUM)
    sources = _sources(feature_dir)
    data = build_selfcond_sft_data(
        sources=sources,
        identity=_sampling_identity(),
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        ddp_rank=0,
        tokenizer=tokenizer,
    )
    assert data.pipeline.negative_bundle is not None
    return _ValidationCase(
        tokenizer=tokenizer,
        task=task,
        features=features,
        sources=sources,
        data=data,
        eval_cfg=SelfCondSftEvalConfig(B=2, T=64, eval_steps=10, seed=731),
        selfcond_cfg=_selfcond_config(),
        token_bytes=t.ones(VOCAB_SIZE, dtype=t.int64),
    )


def _build_loader(
    case: _ValidationCase,
    *,
    sampling_bundle: SamplingBundle | None,
    negative_bundle: NegativeBundle,
    task: Task | None = None,
    batch_size: int | None = None,
    sequence_len: int | None = None,
) -> _SFTValidationLoader:
    assert case.sources.val_feature_dir is not None
    assert case.sources.val_parquet_dir is not None
    return _build_sft_val_loader(
        val_task=case.task if task is None else task,
        tokenizer=case.tokenizer,
        B=case.eval_cfg.B if batch_size is None else batch_size,
        T=case.eval_cfg.T if sequence_len is None else sequence_len,
        device="cpu",
        ddp_rank=0,
        ddp_world_size=1,
        feature_dir=case.sources.val_feature_dir,
        expected_feature_dir=case.sources.val_feature_dir,
        parquet_dir=case.sources.val_parquet_dir,
        threshold_cfg=case.data.threshold_cfg,
        boundaries=case.data.boundaries,
        precomputed_thresholds=case.data.precomputed_thresholds,
        sampling_bundle=sampling_bundle,
        loss_mask_mode=case.sources.loss_mask_mode,
        min_loss_tokens=case.sources.min_loss_tokens,
        n_sae_features=N_SAE_FEATURES,
        fp_percentile_table=None,
        negative_bundle=negative_bundle,
        feature_source_bos_char_len=case.sources.feature_source_bos_char_len,
        alignment_probe_samples=case.sources.alignment_probe_samples,
        mixture_name=case.sources.dataset.value,
        return_seg_doc_id=True,
    )


def _loader_pair(
    case: _ValidationCase,
    *,
    task: Task | None = None,
    batch_size: int | None = None,
    sequence_len: int | None = None,
) -> tuple[_SFTValidationLoader, _SFTValidationLoader]:
    negative = case.data.pipeline.negative_bundle
    assert negative is not None
    seed = selfcond_step_seed(case.eval_cfg.seed, 0, VAL_DATA_SEED_SENTINEL)
    sampling1, negative1 = clone_val_bundles(
        case.data.sampling_bundle,
        negative,
        seed=seed,
    )
    sampling2, negative2 = clone_val_bundles(
        case.data.sampling_bundle,
        negative,
        seed=seed,
    )
    if sampling1 is not None:
        assert sampling1.rng is negative1.rng
    if sampling2 is not None:
        assert sampling2.rng is negative2.rng
    assert negative1.rng is not negative2.rng
    return (
        _build_loader(
            case,
            sampling_bundle=sampling1,
            negative_bundle=negative1,
            task=task,
            batch_size=batch_size,
            sequence_len=sequence_len,
        ),
        _build_loader(
            case,
            sampling_bundle=sampling2,
            negative_bundle=negative2,
            task=task,
            batch_size=batch_size,
            sequence_len=sequence_len,
        ),
    )


def _drain(loader: _SFTValidationLoader, *, max_steps: int = 100) -> list[tuple]:
    batches: list[tuple] = []
    for _ in range(max_steps):
        if loader.state.exhausted:
            break
        batch = next(loader)
        assert isinstance(batch, tuple)
        batches.append(batch)
    assert loader.state.exhausted
    return batches


def _assert_batch_equal(first: tuple, second: tuple) -> None:
    assert len(first) == len(second) == 8
    for first_item, second_item in zip(first, second, strict=True):
        if first_item is None or second_item is None:
            assert first_item is second_item is None
        else:
            assert isinstance(first_item, t.Tensor)
            assert isinstance(second_item, t.Tensor)
            assert t.equal(first_item, second_item)


def _manual_plain_bpb(
    model: t.nn.Module,
    loader: _SFTValidationLoader,
    steps: int,
    token_bytes: t.Tensor,
) -> float:
    total_nats = t.tensor(0.0, dtype=t.float32)
    total_bytes = t.tensor(0, dtype=t.int64)
    with t.no_grad():
        for _ in range(steps):
            if loader.state.exhausted:
                break
            raw = next(loader)
            batch = SelfCondBatch.from_sft(raw)
            loss = model(
                batch.inputs,
                batch.targets,
                batch.enc_fids,
                batch.enc_mags,
                batch.enc_mask,
                batch.seg_map,
                loss_reduction="none",
            ).reshape(-1)
            batch_nats, batch_bytes = accumulate_bpb_nats_bytes(
                loss,
                batch.targets.reshape(-1),
                token_bytes,
            )
            total_nats += batch_nats
            total_bytes += batch_bytes
    return reduce_bpb(total_nats, total_bytes)


def test_val_bpb_selfcond_is_bit_deterministic_across_repeated_evals(
    validation_case: _ValidationCase,
) -> None:
    case = validation_case
    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=740,
    )
    model.eval()
    optimizer = t.optim.AdamW(model.parameters(), lr=0.001)
    parameters_before = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }
    optimizer_before = copy.deepcopy(optimizer.state_dict())

    with patch_load_sft_features(
        case.features,
        split=SFTSplit.VAL,
        mixture=SFTMixture.SAFETY.value,
    ):
        first = run_sft_validation_cycle(
            model=model,
            sources=case.sources,
            sft_data=case.data,
            val_task=case.task,
            tokenizer=case.tokenizer,
            cfg_eval=case.eval_cfg,
            selfcond_cfg=case.selfcond_cfg,
            ddp_rank=0,
            ddp_world_size=1,
            device="cpu",
            token_bytes=case.token_bytes,
        )
        second = run_sft_validation_cycle(
            model=model,
            sources=case.sources,
            sft_data=case.data,
            val_task=case.task,
            tokenizer=case.tokenizer,
            cfg_eval=case.eval_cfg,
            selfcond_cfg=case.selfcond_cfg,
            ddp_rank=0,
            ddp_world_size=1,
            device="cpu",
            token_bytes=case.token_bytes,
        )

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert all(math.isfinite(metric) for metric in first)
    for name, parameter in model.named_parameters():
        assert t.equal(parameter, parameters_before[name])
        assert parameter.grad is None
    assert optimizer.state_dict() == optimizer_before


def test_both_val_loaders_serve_identical_example_streams(
    validation_case: _ValidationCase,
) -> None:
    case = validation_case
    first_loader, second_loader = _loader_pair(case)
    with patch_load_sft_features(
        case.features,
        split=SFTSplit.VAL,
        mixture=SFTMixture.SAFETY.value,
    ):
        first_batches = _drain(first_loader)
        second_batches = _drain(second_loader)

    assert len(first_batches) == len(second_batches) > 0
    for first, second in zip(first_batches, second_batches, strict=True):
        _assert_batch_equal(first, second)
    assert first_loader.state.consumed == second_loader.state.consumed == len(case.task)


def test_real_val_bpb_matches_sft_evaluator_on_identical_loader_state(
    validation_case: _ValidationCase,
) -> None:
    case = validation_case
    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=750,
    )
    model.eval()
    production_loader, manual_loader = _loader_pair(case)
    with patch_load_sft_features(
        case.features,
        split=SFTSplit.VAL,
        mixture=SFTMixture.SAFETY.value,
    ):
        production = evaluate_encdec_bpb(
            model,
            production_loader,
            case.eval_cfg.eval_steps,
            case.token_bytes,
        )
        manual = _manual_plain_bpb(
            model,
            manual_loader,
            case.eval_cfg.eval_steps,
            case.token_bytes,
        )

    assert production == manual
    assert production_loader.state.consumed == manual_loader.state.consumed
    assert production_loader.state.exhausted
    assert manual_loader.state.exhausted


def test_selfcond_bpb_uses_final_rollout_conditioning(
    validation_case: _ValidationCase,
) -> None:
    case = validation_case
    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=760,
    )
    model.eval()
    evaluator_loader, manual_loader = _loader_pair(case)
    with patch_load_sft_features(
        case.features,
        split=SFTSplit.VAL,
        mixture=SFTMixture.SAFETY.value,
    ):
        raw = next(manual_loader)
        batch = SelfCondBatch.from_sft(raw)
        rng = t.Generator().manual_seed(
            selfcond_step_seed(
                case.selfcond_cfg.seed,
                0,
                VAL_ROLLOUT_STEP_SENTINEL,
            )
        )
        rollout_result = rollout(model, batch, case.selfcond_cfg, rng, None)
        assert bool(rollout_result.predicted.any())
        assert not t.equal(rollout_result.conditioning.mask, batch.enc_mask)
        with t.no_grad():
            final_loss = model(
                batch.inputs,
                batch.targets,
                rollout_result.conditioning.fids,
                rollout_result.conditioning.mags,
                rollout_result.conditioning.mask,
                batch.seg_map,
                loss_reduction="none",
            ).reshape(-1)
        total_nats, total_bytes = accumulate_bpb_nats_bytes(
            final_loss,
            batch.targets.reshape(-1),
            case.token_bytes,
        )
        expected = reduce_bpb(total_nats, total_bytes)
        actual = evaluate_selfcond_bpb_sft(
            model,
            evaluator_loader,
            1,
            case.token_bytes,
            case.selfcond_cfg,
            ddp_rank=0,
            device="cpu",
        )

    assert actual == expected


def test_masked_targets_and_padded_rows_contribute_nothing(
    validation_case: _ValidationCase,
) -> None:
    case = validation_case
    task = MultiTurnTask(n=1)
    features = build_synthetic_features(
        task,
        case.tokenizer,
        n_segments_per_conv=2,
        n_sae_features=N_SAE_FEATURES,
    )
    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=770,
    )
    model.eval()
    evaluator_loader, manual_loader = _loader_pair(case, task=task, batch_size=2)
    with patch_load_sft_features(
        features,
        split=SFTSplit.VAL,
        mixture=SFTMixture.SAFETY.value,
    ):
        raw = next(manual_loader)
        batch = SelfCondBatch.from_sft(raw)
        assert bool((batch.targets[0] == -1).any())
        assert bool((batch.targets[1] == -1).all())
        rng = t.Generator().manual_seed(
            selfcond_step_seed(
                case.selfcond_cfg.seed,
                0,
                VAL_ROLLOUT_STEP_SENTINEL,
            )
        )
        rollout_result = rollout(model, batch, case.selfcond_cfg, rng, None)
        with t.no_grad():
            loss2d = model(
                batch.inputs,
                batch.targets,
                rollout_result.conditioning.fids,
                rollout_result.conditioning.mags,
                rollout_result.conditioning.mask,
                batch.seg_map,
                loss_reduction="none",
            ).reshape_as(batch.targets)
        full_nats, full_bytes = accumulate_bpb_nats_bytes(
            loss2d.reshape(-1),
            batch.targets.reshape(-1),
            case.token_bytes,
        )
        row_nats, row_bytes = accumulate_bpb_nats_bytes(
            loss2d[0].reshape(-1),
            batch.targets[0].reshape(-1),
            case.token_bytes,
        )
        assert t.equal(full_nats, row_nats)
        assert t.equal(full_bytes, row_bytes)
        assert int(full_bytes.item()) == int((batch.targets[0] >= 0).sum().item())
        expected = reduce_bpb(row_nats, row_bytes)
        actual = evaluate_selfcond_bpb_sft(
            model,
            evaluator_loader,
            10,
            case.token_bytes,
            case.selfcond_cfg,
            ddp_rank=0,
            device="cpu",
        )

    assert math.isfinite(actual)
    assert actual == expected
    assert evaluator_loader.state.exhausted


def test_eval_steps_bound_and_exhaustion_break(
    validation_case: _ValidationCase,
) -> None:
    case = validation_case
    task = MultiTurnTask(n=3, user_chars=3, asst_chars=6)
    features = build_synthetic_features(
        task,
        case.tokenizer,
        n_segments_per_conv=4,
        n_sae_features=N_SAE_FEATURES,
    )
    model = tiny_head_model(
        FeaturePredLossMode.MAX_POOL,
        n_sae=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
        seed=780,
    )
    model.eval()
    bounded_loader, exhausted_loader = _loader_pair(
        case,
        task=task,
        batch_size=1,
        sequence_len=12,
    )
    with patch_load_sft_features(
        features,
        split=SFTSplit.VAL,
        mixture=SFTMixture.SAFETY.value,
    ):
        bounded_metric = evaluate_selfcond_bpb_sft(
            model,
            bounded_loader,
            2,
            case.token_bytes,
            case.selfcond_cfg,
            ddp_rank=0,
            device="cpu",
        )
        exhausted_metric = evaluate_selfcond_bpb_sft(
            model,
            exhausted_loader,
            20,
            case.token_bytes,
            case.selfcond_cfg,
            ddp_rank=0,
            device="cpu",
        )

    assert math.isfinite(bounded_metric)
    assert math.isfinite(exhausted_metric)
    assert bounded_loader.state.consumed == 2
    assert not bounded_loader.state.exhausted
    assert exhausted_loader.state.consumed == len(task)
    assert exhausted_loader.state.exhausted
