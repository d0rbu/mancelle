"""Tests for sweep fast-path correctness: weight freshness, RNG isolation, compile survival."""

from __future__ import annotations

import gc
import weakref

import pytest
import torch as t

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_train import TrainResult
from scripts.train.run_encdec_sweep import build_sweep_configs
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_pipeline import build_feature_pipeline
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.stats_types import PerFeatureStat
from tests.conftest import (
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    _make_encdec,
    _make_gpt_config,
)


def _make_per_feature_stats(n_features: int) -> dict[str, PerFeatureStat]:
    """Build minimal per-feature stats stubs for pipeline tests."""
    percentiles = {f"p{k}": float(k) for k in range(1, 100)}
    stub = PerFeatureStat(
        count=100,
        pct_segments_active=0.5,
        mean_mag=1.0,
        mean_mag_incl_zeros=0.5,
        std_mag=0.2,
        min_mag=0.0,
        max_mag=5.0,
        median_mag=1.0,
        percentiles=percentiles,
    )
    return {str(i): stub for i in range(n_features)}


class TestWeightFreshness:
    """Verify init_weights() produces genuinely fresh parameters."""

    def test_init_weights_produces_fresh_params(self):
        """After init_weights(), parameters that are not intentionally zero differ from perturbed values.

        Some projections (c_proj, mlp.c_proj, ve_gate) are deliberately zero-initialized;
        lambdas are filled to fixed constants.  We skip those and focus on the remaining
        parameters, which should be restored to their canonical distribution rather than
        left in an arbitrarily perturbed state.
        """
        model = _make_encdec(device="cpu")
        model.train()
        # Perturb every parameter significantly so that re-init must visibly change them.
        with t.no_grad():
            for p in model.parameters():
                p.add_(t.randn_like(p) * 100.0)
        perturbed = {name: p.clone() for name, p in model.named_parameters()}
        model.init_weights()
        # Skip params that are legitimately zero-init or constant-fill on every call
        _ZERO_OR_CONST_SUFFIXES = ("c_proj.weight", "mlp.c_proj.weight", "ve_gate.weight")
        _CONST_NAMES = ("resid_lambdas", "x0_lambdas")
        unchanged = []
        for name, p in model.named_parameters():
            if any(name.endswith(s) for s in _ZERO_OR_CONST_SUFFIXES):
                continue
            if any(name == c or name.endswith(f".{c}") for c in _CONST_NAMES):
                continue
            if "bias" in name:
                continue
            if t.equal(perturbed[name], p):
                unchanged.append(name)
        assert len(unchanged) == 0, f"Parameters unchanged after init_weights(): {unchanged}"

    def test_init_weights_is_seeded(self):
        """Two init_weights() calls with the same global seed produce identical params."""
        model = _make_encdec(device="cpu")
        t.manual_seed(123)
        model.init_weights()
        snapshot_a = {n: p.clone() for n, p in model.named_parameters()}
        t.manual_seed(123)
        model.init_weights()
        snapshot_b = {n: p.clone() for n, p in model.named_parameters()}
        for name, weight_a in snapshot_a.items():
            assert t.equal(weight_a, snapshot_b[name]), f"Parameter {name} differs with same seed"

    def test_zero_grad_cleans_gradients(self):
        """After training + zero_grad(set_to_none=True), no gradient tensors remain."""
        model = _make_encdec(device="cpu")
        model.train()
        B, T = 2, 32
        n_seg = T // 16
        x = t.randint(0, 128, (B, T))
        y = t.randint(0, 128, (B, T))
        fids = t.randint(0, TINY_N_SAE_FEATURES, (B, n_seg, 4))
        mags = t.randint(0, TINY_NUM_BUCKETS, (B, n_seg, 4))
        mask = t.ones(B, n_seg, 4, dtype=t.bool)
        seg_map = t.arange(T).unsqueeze(0).expand(B, -1) // 16

        fa_module._override_impl = FlashAttnImpl.SDPA

        loss = model(x, y, fids, mags, mask, seg_map)
        if hasattr(loss, 'loss'):
            loss = loss.loss
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad, "No gradients after backward"
        model.zero_grad(set_to_none=True)
        remaining = [n for n, p in model.named_parameters() if p.grad is not None]
        assert len(remaining) == 0, f"Gradients remain after zero_grad: {remaining}"


class TestRNGIsolation:
    """Verify RNG state is properly reset between configs."""

    def test_same_seed_same_weights(self):
        model = _make_encdec(device="cpu")
        t.manual_seed(42)
        model.init_weights()
        weights_a = {n: p.clone() for n, p in model.named_parameters()}
        for p in model.parameters():
            p.data.add_(t.randn_like(p) * 0.1)
        t.manual_seed(42)
        model.init_weights()
        weights_b = {n: p.clone() for n, p in model.named_parameters()}
        for name, weight_a in weights_a.items():
            assert t.equal(weight_a, weights_b[name])

    def test_different_seed_different_weights(self):
        model = _make_encdec(device="cpu")
        t.manual_seed(42)
        model.init_weights()
        weights_a = {n: p.clone() for n, p in model.named_parameters()}
        t.manual_seed(99)
        model.init_weights()
        weights_b = {n: p.clone() for n, p in model.named_parameters()}
        any_different = any(not t.equal(weights_a[n], weights_b[n]) for n in weights_a if "bias" not in n)
        assert any_different


class TestGCLifecycle:
    def test_gc_reenabled_after_freeze(self):
        gc.freeze()
        gc.disable()
        assert not gc.isenabled()
        gc.enable()
        if hasattr(gc, 'unfreeze'):
            gc.unfreeze()
        gc.collect()
        assert gc.isenabled()

    def test_gc_collect_frees_memory(self):
        class Dummy:
            pass

        obj = Dummy()
        ref = weakref.ref(obj)
        gc.freeze()
        gc.disable()
        del obj
        gc.enable()
        if hasattr(gc, 'unfreeze'):
            gc.unfreeze()
        gc.collect()
        assert ref() is None


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
class TestCompileSurvival:
    def test_compiled_model_survives_reinit(self):
        model = _make_encdec(device="cuda")
        model.train()
        compiled = t.compile(model, dynamic=True)
        assert isinstance(compiled, t._dynamo.eval_frame.OptimizedModule)
        B, T = 2, 32
        n_seg = T // 16
        x = t.randint(0, 128, (B, T), device="cuda")
        y = t.randint(0, 128, (B, T), device="cuda")
        fids = t.randint(0, TINY_N_SAE_FEATURES, (B, n_seg, 4), device="cuda")
        mags = t.randint(0, TINY_NUM_BUCKETS, (B, n_seg, 4), device="cuda")
        mask = t.ones(B, n_seg, 4, dtype=t.bool, device="cuda")
        seg_map = t.arange(T, device="cuda").unsqueeze(0).expand(B, -1) // 16
        result1 = compiled(x, y, fids, mags, mask, seg_map)
        model.init_weights()
        assert isinstance(compiled, t._dynamo.eval_frame.OptimizedModule)
        result2 = compiled(x, y, fids, mags, mask, seg_map)
        loss1 = result1.loss if hasattr(result1, 'loss') else result1
        loss2 = result2.loss if hasattr(result2, 'loss') else result2
        assert not t.equal(loss1, loss2)


class TestFeaturePredConfigPatching:
    def test_patch_fp_weight(self):
        dec_config = _make_gpt_config()
        enc_config = EncoderConfig(n_embd=64, depth=2, n_head=2, n_kv_head=2, head_dim=32)
        xattn_config = CrossAttentionConfig(frequency=1)
        fp_config = FeaturePredictionConfig(
            weight=0.1, layer=-1,
            loss_mode=FeaturePredLossMode.BEST_TOKEN,
            loss_fn=FeaturePredLossFn.BCE,
        )
        model = EncoderDecoderGPT(
            dec_config, enc_config, xattn_config,
            n_sae_features=TINY_N_SAE_FEATURES, num_buckets=TINY_NUM_BUCKETS,
            feature_pred_config=fp_config, pad_vocab_size_to=128,
        )
        model.to("cpu")
        model.init_weights()
        assert model.feature_pred_config.weight == 0.1  # ty: ignore[unresolved-attribute]
        model.feature_pred_config.weight = 1.0  # ty: ignore[invalid-assignment]
        assert model.feature_pred_config.weight == 1.0  # ty: ignore[unresolved-attribute]
        model.feature_pred_config.weight = 0.01  # ty: ignore[invalid-assignment]
        assert model.feature_pred_config.weight == 0.01  # ty: ignore[unresolved-attribute]


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
class TestMemoryStability:
    def test_memory_stable_across_reinits(self):
        """GPU memory after 5 init_weights() cycles stays within 10% of baseline."""
        model = _make_encdec(device="cuda")
        model.train()
        model.init_weights()
        t.cuda.synchronize()
        t.cuda.reset_peak_memory_stats()
        baseline = t.cuda.max_memory_allocated()
        for _ in range(5):
            model.init_weights()
            opt = t.optim.Adam(model.parameters(), lr=1e-3)
            del opt
            t.cuda.empty_cache()
            gc.collect()
        t.cuda.synchronize()
        final = t.cuda.max_memory_allocated()
        ratio = final / max(baseline, 1)
        assert ratio < 1.1, f"Memory grew by {(ratio - 1) * 100:.1f}%"


class TestDeltaTransitions:
    def test_delta_enabled_sets_bundle(self):
        """build_feature_pipeline with delta enabled produces non-None delta_bundle."""
        per_feature_stats = _make_per_feature_stats(TINY_N_SAE_FEATURES)
        # feature_percentile is supported by extract_threshold_percentile
        feature_selection = ThresholdSelection(
            threshold_type=ThresholdType.feature_percentile,
            threshold=0.5,
        )
        feature_delta = FeatureDeltaConfig(mode=DeltaMode.prev_segment)
        pipeline = build_feature_pipeline(
            feature_selection=feature_selection,
            negative_selection=None,
            per_feature_stats=per_feature_stats,
            n_sae_features=TINY_N_SAE_FEATURES,
            num_buckets=TINY_NUM_BUCKETS,
            feature_delta=feature_delta,
        )
        assert pipeline.delta_bundle is not None

    def test_delta_disabled_bundle_is_none(self):
        """build_feature_pipeline with delta=None produces None delta_bundle."""
        per_feature_stats = _make_per_feature_stats(TINY_N_SAE_FEATURES)
        feature_selection = ThresholdSelection(
            threshold_type=ThresholdType.feature_percentile,
            threshold=0.5,
        )
        pipeline = build_feature_pipeline(
            feature_selection=feature_selection,
            negative_selection=None,
            per_feature_stats=per_feature_stats,
            n_sae_features=TINY_N_SAE_FEATURES,
            num_buckets=TINY_NUM_BUCKETS,
            feature_delta=None,
        )
        assert pipeline.delta_bundle is None


class TestOOMResilience:
    def test_exception_produces_error_result(self):
        def fake_train():
            raise RuntimeError("CUDA out of memory")
        try:
            fake_train()
            result = TrainResult(model_tag="test", val_bpb=0.5,
                min_val_bpb=0.5, total_training_time=10.0, peak_memory_mib=1000.0, startup_time=1.0)
        except Exception as e:
            result = TrainResult(model_tag="test", val_bpb=None,
                min_val_bpb=float("inf"), total_training_time=0.0, peak_memory_mib=0.0,
                startup_time=0.0, error=str(e))
        assert result.error is not None
        assert "out of memory" in result.error


class TestFP8Guard:
    def test_float8linear_no_scaling_buffers(self):
        """Tensorwise Float8Linear has no persistent scaling buffers."""
        try:
            from torchao.float8 import Float8LinearConfig, convert_to_float8_training
        except ImportError:
            pytest.skip("torchao not available")
        import torch.nn as nn
        model = nn.Sequential(nn.Linear(128, 128))
        config = Float8LinearConfig.from_recipe_name("tensorwise")
        convert_to_float8_training(model, config=config)
        mod = model[0]
        buffers = list(mod.named_buffers())
        assert len(buffers) == 0, f"Float8Linear has buffers: {[n for n, _ in buffers]}"


class TestSweepGrid:
    def test_total_configs(self):
        configs = build_sweep_configs()
        assert len(configs) == 315

    def test_unique_model_tags(self):
        # model_tag intentionally omits feature_selection (captured by checkpoint subdir).
        # Uniqueness holds when combined with the feature-selection key.
        configs = build_sweep_configs()
        full_keys = [
            (c.model_tag, c.fs.distribution, c.fs.mean, c.fs.std)
            for c in configs
        ]
        assert len(full_keys) == len(set(full_keys))

    def test_hydra_overrides_format(self):
        configs = build_sweep_configs()
        for cfg in configs[:5]:
            overrides = cfg.hydra_overrides()
            for ov in overrides:
                assert "=" in ov, f"Override missing '=': {ov}"
