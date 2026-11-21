"""4090 encoder-decoder config and memory-budget coverage."""

import gc
import math
from pathlib import Path

import pytest
import torch as t

from scripts.train.conditional_encdec_train import (
    build_encdec_meta_model_from_training_cfg,
    resolve_live_pad_n_seg_from_training_cfg,
    resolve_static_pad_f_from_training_cfg,
)
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.config import (
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import ForwardResult
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES, SAEWidth
from tests.feature_prediction_helpers import dense_targets_from_sparse

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAINING_CONFIG_DIR = str(REPO_ROOT / "configs" / "training")
NANOCHAT_VOCAB_SIZE = 32768
GIB = 1024**3
MEMORY_BUDGET_BYTES = 22 * GIB
MIN_4090_CLASS_TOTAL_GIB = 23.0
MIN_MEANINGFUL_FREE_GIB = 22.0
RESIDENT_LEAK_TOLERANCE_BYTES = 256 * 1024**2


# The feature-prediction head is disabled by default (see conditional_encdec_train.yaml).
# The VRAM budget must still hold for a run that opts in, since that is the heaviest
# shape a 4090 can be asked to train, so the budget test composes it explicitly.
FP_HEAD_OVERRIDES = [
    "+feature_prediction.weight=1.0",
    "+feature_prediction.layer=-1",
    "+feature_prediction.loss_mode=best_token",
    "+feature_prediction.loss_fn=bce",
]


def _compose_4090_cfg(hydra_compose, overrides: list[str] | None = None):
    return hydra_compose(TRAINING_CONFIG_DIR, "conditional_encdec_4090", overrides)


def _n_sae_features_from_cfg(cfg) -> int:
    return SAE_WIDTH_TO_NUM_FEATURES[SAEWidth(cfg.sae.width)]


def _build_4090_meta_model(cfg):
    return build_encdec_meta_model_from_training_cfg(
        cfg,
        base_vocab=NANOCHAT_VOCAB_SIZE,
        n_sae_features=_n_sae_features_from_cfg(cfg),
    )


def _peak_delta_bytes(
    *,
    peak_allocated_bytes: int,
    baseline_allocated_bytes: int,
) -> int:
    return peak_allocated_bytes - baseline_allocated_bytes


def _fits_memory_budget(
    *,
    peak_allocated_bytes: int,
    baseline_allocated_bytes: int,
    budget_bytes: int = MEMORY_BUDGET_BYTES,
) -> bool:
    return (
        _peak_delta_bytes(
            peak_allocated_bytes=peak_allocated_bytes,
            baseline_allocated_bytes=baseline_allocated_bytes,
        )
        < budget_bytes
    )


def _cuda_cleanup() -> None:
    gc.collect()
    t.cuda.empty_cache()
    t.cuda.synchronize()


def _cuda_mem_info_gib() -> tuple[float, float]:
    free, total = t.cuda.mem_get_info()
    return free / GIB, total / GIB


def test_4090_training_config_contract_matches_production_helper(hydra_compose) -> None:
    cfg = _compose_4090_cfg(hydra_compose)
    model = _build_4090_meta_model(cfg)

    dec = model.dec_config
    assert dec.sequence_len == 1024
    assert dec.vocab_size == NANOCHAT_VOCAB_SIZE
    assert dec.n_layer == 12
    assert dec.n_embd == 768
    assert dec.n_head == 6
    assert dec.n_kv_head == 6
    assert dec.window_pattern == "SSSL"

    enc = model.encoder.config
    assert enc.n_sae_features == _n_sae_features_from_cfg(cfg)
    assert enc.num_buckets == 2
    assert enc.n_embd == 384
    assert enc.n_layer == 6
    assert enc.n_head == 6
    assert enc.n_kv_head == 6
    assert enc.head_dim == 64

    assert set(model.cross_attn_layers.keys()) == {"3", "7", "11"}
    for layer in model.cross_attn_layers.values():
        assert layer.n_head == dec.n_head
        assert layer.n_kv_head == dec.n_head
        assert layer.head_dim == dec.n_embd // dec.n_head
        assert layer.dec_n_embd == dec.n_embd
        assert layer.enc_n_embd == enc.n_embd

    # The feature-prediction head is opt-in, so the production 4090 config carries none.
    #
    # It is off by default because PR #697's d12_fp / d12_fp10 ablations did not
    # improve steering, and best_token cannot run at inference.
    assert model.feature_pred_config is None
    assert model.feature_pred_head is None

    # Opting in still resolves the head, and that is the shape the memory budget
    # below is measured against.
    #
    # This helper builds without a feature_vocab, so it measures the frozen
    # full-width reference. Compact-head memory is measured separately against an
    # explicit effective vocabulary.
    fp_model = _build_4090_meta_model(_compose_4090_cfg(hydra_compose, FP_HEAD_OVERRIDES))
    assert fp_model.feature_pred_config is not None
    assert fp_model.feature_pred_head is not None
    assert fp_model.feature_pred_config.loss_mode == FeaturePredLossMode.BEST_TOKEN
    assert fp_model.feature_pred_config.loss_fn == FeaturePredLossFn.BCE


def test_training_config_depth_override_helper_uses_requested_depth(hydra_compose) -> None:
    cfg = _compose_4090_cfg(hydra_compose, ["depth=8"])
    assert cfg.depth == 8

    model = build_encdec_meta_model_from_training_cfg(
        cfg,
        base_vocab=NANOCHAT_VOCAB_SIZE,
        n_sae_features=_n_sae_features_from_cfg(cfg),
        depth=12,
    )

    assert model.dec_config.n_layer == 12
    assert model.dec_config.n_embd == 768
    assert model.encoder.config.n_layer == 6
    assert set(model.cross_attn_layers.keys()) == {"3", "7", "11"}


def test_static_pad_f_resolves_4090_sampling_bounds(hydra_compose) -> None:
    cfg = _compose_4090_cfg(hydra_compose)
    assert resolve_static_pad_f_from_training_cfg(cfg) == 63


def test_live_static_pad_n_seg_resolves_4090_padding(hydra_compose) -> None:
    cfg = _compose_4090_cfg(hydra_compose)
    assert resolve_live_pad_n_seg_from_training_cfg(cfg, pad_f=40) == 14
    assert resolve_live_pad_n_seg_from_training_cfg(cfg, pad_f=None) is None


def test_static_pad_f_aligns_fp8_to_encoder_alignment(hydra_compose) -> None:
    cfg = _compose_4090_cfg(hydra_compose, ["fp8=true"])
    assert resolve_static_pad_f_from_training_cfg(cfg) == 64


def test_static_pad_f_no_negative_selection_uses_positive_max(hydra_compose) -> None:
    cfg = _compose_4090_cfg(hydra_compose, ["negative_selection=null"])
    assert resolve_static_pad_f_from_training_cfg(cfg) == 48


@pytest.mark.parametrize(
    "overrides",
    [
        ["feature_selection.max_k=null"],
        ["negative_selection.max_k=null"],
    ],
    ids=["positive-unbounded", "negative-unbounded"],
)
def test_static_pad_f_unbounded_sampling_returns_none(
    hydra_compose,
    overrides: list[str],
) -> None:
    cfg = _compose_4090_cfg(hydra_compose, overrides)
    assert resolve_static_pad_f_from_training_cfg(cfg) is None


def test_memory_budget_accounting_uses_peak_delta_not_process_peak() -> None:
    baseline = 30 * GIB
    peak = baseline + 2 * GIB

    assert (
        _peak_delta_bytes(
            peak_allocated_bytes=peak,
            baseline_allocated_bytes=baseline,
        )
        == 2 * GIB
    )
    assert _fits_memory_budget(
        peak_allocated_bytes=peak,
        baseline_allocated_bytes=baseline,
    )
    assert peak > MEMORY_BUDGET_BYTES


@pytest.mark.slow
def test_cuda_peak_delta_excludes_live_baseline_allocation() -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    sentinel = None
    probe = None
    baseline_allocated = 0
    try:
        _cuda_cleanup()
        free_gib, _total_gib = _cuda_mem_info_gib()
        if free_gib < 1.0:
            pytest.skip(
                f"Need >= 1.0 GiB free VRAM for sentinel allocation, have {free_gib:.1f} GiB"
            )
        sentinel = t.empty(512 * 1024**2, dtype=t.uint8, device="cuda")
        t.cuda.synchronize()
        baseline_allocated = t.cuda.memory_allocated()
        t.cuda.reset_peak_memory_stats()

        probe = t.empty(64 * 1024**2, dtype=t.uint8, device="cuda")
        t.cuda.synchronize()
        delta = _peak_delta_bytes(
            peak_allocated_bytes=t.cuda.max_memory_allocated(),
            baseline_allocated_bytes=baseline_allocated,
        )

        assert delta < baseline_allocated
        assert delta >= 64 * 1024**2
    finally:
        del probe
        del sentinel
        if t.cuda.is_available():
            _cuda_cleanup()


@pytest.mark.slow
def test_4090_config_fits_in_24gb(hydra_compose) -> None:
    """Verify the 4090 production config stays below a 22 GiB fwd/bwd budget."""
    if not t.cuda.is_available():
        pytest.skip("CUDA required")

    _cuda_cleanup()
    free_gib, total_gib = _cuda_mem_info_gib()
    if total_gib < MIN_4090_CLASS_TOTAL_GIB:
        pytest.skip(
            f"Need 4090-class total VRAM >= {MIN_4090_CLASS_TOTAL_GIB:.1f} GiB, "
            f"have {total_gib:.1f} GiB"
        )
    if free_gib < MIN_MEANINGFUL_FREE_GIB:
        pytest.skip(
            f"Need >= {MIN_MEANINGFUL_FREE_GIB:.1f} GiB free VRAM after cleanup, "
            f"have {free_gib:.1f} GiB"
        )

    # Measure the heaviest shape a 4090 can be asked to train: the head is opt-in now,
    # but a run that enables it must still fit in the budget.
    cfg = _compose_4090_cfg(hydra_compose, FP_HEAD_OVERRIDES)
    n_sae_features = _n_sae_features_from_cfg(cfg)
    pad_f = resolve_static_pad_f_from_training_cfg(cfg)
    assert pad_f is not None

    model = None
    result = None
    loss = None
    idx = None
    targets = None
    enc_fids = None
    enc_mags = None
    enc_mask = None
    enc_percentiles = None
    seg_map = None
    seg_doc_id = None
    fp_targets = None
    baseline_allocated = t.cuda.memory_allocated()
    peak_delta = None

    try:
        t.cuda.reset_peak_memory_stats()

        model = _build_4090_meta_model(cfg)
        model.to_empty(device="cuda")
        model.init_weights()
        model.train()

        batch_size = int(cfg.device_batch_size)
        sequence_len = int(cfg.max_seq_len)
        segment_size = int(cfg.segment_size)
        actual_n_seg = math.ceil(sequence_len / segment_size)
        n_seg = resolve_live_pad_n_seg_from_training_cfg(cfg, pad_f=pad_f)
        assert n_seg is not None
        assert n_seg >= actual_n_seg
        static_max_seg_size = ((segment_size + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT

        idx = t.randint(
            0,
            NANOCHAT_VOCAB_SIZE,
            (batch_size, sequence_len),
            dtype=t.long,
            device="cuda",
        )
        targets = t.randint(
            0,
            NANOCHAT_VOCAB_SIZE,
            (batch_size, sequence_len),
            dtype=t.long,
            device="cuda",
        )
        enc_fids = t.randint(
            0,
            n_sae_features,
            (batch_size, n_seg, pad_f),
            dtype=t.long,
            device="cuda",
        )
        enc_mags = t.randint(
            0,
            int(cfg.num_buckets),
            (batch_size, n_seg, pad_f),
            dtype=t.long,
            device="cuda",
        )
        enc_mask = t.zeros(
            batch_size,
            n_seg,
            pad_f,
            dtype=t.bool,
            device="cuda",
        )
        enc_mask[:, :actual_n_seg, :] = True
        enc_percentiles = t.zeros(
            batch_size,
            n_seg,
            pad_f,
            dtype=t.float32,
            device="cuda",
        )
        enc_percentiles[:, :actual_n_seg, :] = t.rand(
            batch_size,
            actual_n_seg,
            pad_f,
            dtype=t.float32,
            device="cuda",
        )
        seg_map = (
            t.arange(sequence_len, device="cuda", dtype=t.long).unsqueeze(0).expand(batch_size, -1)
            // segment_size
        )
        seg_doc_id = t.full((batch_size, n_seg), -1, dtype=t.long, device="cuda")
        seg_doc_id[:, :actual_n_seg] = 0
        fp_targets = t.stack(
            [
                dense_targets_from_sparse(
                    enc_fids[row],
                    enc_percentiles[row],
                    enc_mask[row],
                    n_sae_features,
                )
                for row in range(batch_size)
            ],
            dim=0,
        )

        result = model(
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
            static_max_seg_size=static_max_seg_size,
            fp_targets=fp_targets,
        )
        loss = result.loss if isinstance(result, ForwardResult) else result
        loss.backward()
        t.cuda.synchronize()

        peak_allocated = t.cuda.max_memory_allocated()
        peak_delta = _peak_delta_bytes(
            peak_allocated_bytes=peak_allocated,
            baseline_allocated_bytes=baseline_allocated,
        )
        print(f"Peak memory delta: {peak_delta / 1024 / 1024:.1f} MiB")
    finally:
        del loss
        del result
        del fp_targets
        del seg_doc_id
        del seg_map
        del enc_percentiles
        del enc_mask
        del enc_mags
        del enc_fids
        del targets
        del idx
        del model
        if t.cuda.is_available():
            _cuda_cleanup()
            resident_after = t.cuda.memory_allocated()
            assert resident_after <= (baseline_allocated + RESIDENT_LEAK_TOLERANCE_BYTES), (
                f"Resident CUDA allocation after cleanup "
                f"({resident_after / 1024 / 1024:.1f} MiB) did not return near "
                f"baseline ({baseline_allocated / 1024 / 1024:.1f} MiB)"
            )

    assert peak_delta is not None
    assert peak_delta < MEMORY_BUDGET_BYTES, (
        f"Peak memory delta {peak_delta / 1024 / 1024:.1f} MiB exceeds "
        f"{MEMORY_BUDGET_BYTES / 1024 / 1024:.1f} MiB limit"
    )
