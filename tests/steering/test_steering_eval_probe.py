# tests/test_steering_eval_probe.py
"""Tests for apply_sae_activation in steering/eval/feature_probe.py."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import CrossAttentionConfig, EncoderConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_index import SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.feature_probe import (
    apply_sae_activation,
    apply_sae_activation_batched,
    compute_raw_probe,
    compute_raw_probe_batched,
    max_pool_masked,
    max_pool_masked_batched,
)
from steering.eval.inline_steering_eval import (
    RoundtripCompletion,
    _aggregate_roundtrip_metrics,
    _probe_roundtrip_raw,
    evaluate_steering,
)
from steering.eval.multiseg_generator import CompletionSample
from steering.eval.probe_model import ProbeModel
from steering.feature_io import save_features
from steering.stats_types import PerFeatureStat


class _FakeSAEConfig:
    """Minimal stand-in for sae_lens SAE config (JumpReLU / ReLU)."""

    def __init__(self, apply_b_dec_to_input: bool = True):
        self.apply_b_dec_to_input = apply_b_dec_to_input


class _JumpReLUSAE:
    """Minimal stand-in for a JumpReLU SAE (has .threshold attribute)."""

    def __init__(
        self,
        W_enc: t.Tensor,
        b_enc: t.Tensor,
        b_dec: t.Tensor,
        threshold: t.Tensor,
    ):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.threshold = threshold
        self.cfg = _FakeSAEConfig(apply_b_dec_to_input=True)

    def encode(self, x: t.Tensor) -> t.Tensor:
        sae_in = x - self.b_dec if self.cfg.apply_b_dec_to_input else x
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        return t.relu(hidden_pre) * (hidden_pre > self.threshold).float()


class _ReLUSAE:
    """Minimal stand-in for a plain ReLU SAE (no threshold attribute)."""

    def __init__(
        self,
        W_enc: t.Tensor,
        b_enc: t.Tensor,
        b_dec: t.Tensor,
    ):
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.b_dec = b_dec
        self.cfg = _FakeSAEConfig(apply_b_dec_to_input=True)

    def encode(self, x: t.Tensor) -> t.Tensor:
        sae_in = x - self.b_dec if self.cfg.apply_b_dec_to_input else x
        hidden_pre = sae_in @ self.W_enc + self.b_enc
        return t.relu(hidden_pre)


class _TopKSAEConfig:
    """Minimal stand-in for a TopK SAE config."""

    def __init__(self, apply_b_dec_to_input: bool = True, activation_fn: str = "topk"):
        self.apply_b_dec_to_input = apply_b_dec_to_input
        self.activation_fn = activation_fn


class _TopKSAE:
    """Minimal stand-in for a TopK SAE (no threshold, unsupported activation)."""

    def __init__(self):
        self.cfg = _TopKSAEConfig(apply_b_dec_to_input=True, activation_fn="topk")


class TestApplySaeActivationJumpReLU:
    """JumpReLU gating: pre_act * (pre_act > threshold[fid])."""

    def _make_sae(self, n_features: int = 3, threshold_val: float = 0.5) -> _JumpReLUSAE:
        d_model = 4
        W_enc = t.randn(d_model, n_features)
        b_enc = t.zeros(n_features)
        b_dec = t.zeros(d_model)
        threshold = t.full((n_features,), threshold_val)
        return _JumpReLUSAE(W_enc, b_enc, b_dec, threshold)

    def test_below_threshold_zeroed(self):
        sae = self._make_sae(threshold_val=1.0)
        # pre_act below threshold → 0
        pre_act = t.tensor([0.3, 0.5, 0.9], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.all(result == 0.0), f"Expected all zeros, got {result}"

    def test_above_threshold_passes_through(self):
        sae = self._make_sae(threshold_val=0.5)
        pre_act = t.tensor([0.8, 1.2, 2.0], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.allclose(result, pre_act, atol=1e-6), (
            f"Expected pass-through, got {result}"
        )

    def test_at_threshold_is_zero_strict_greater_than(self):
        # JumpReLU uses strict >, so value exactly at threshold → 0
        sae = self._make_sae(threshold_val=1.0)
        pre_act = t.tensor([1.0, 1.0, 1.0], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.all(result == 0.0), (
            f"Value at threshold should be 0 (strict >), got {result}"
        )

    def test_negative_pre_act_zeroed(self):
        sae = self._make_sae(threshold_val=0.0)
        pre_act = t.tensor([-1.0, -0.5, -0.001], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.all(result == 0.0), f"Negative pre_act should be 0, got {result}"

    def test_per_feature_threshold_respected(self):
        d_model = 2
        n_features = 3
        W_enc = t.zeros(d_model, n_features)
        b_enc = t.zeros(n_features)
        b_dec = t.zeros(d_model)
        # Different per-feature thresholds
        threshold = t.tensor([0.1, 0.5, 0.9])
        sae = _JumpReLUSAE(W_enc, b_enc, b_dec, threshold)

        pre_act = t.tensor([0.6, 0.6, 0.6], dtype=t.float32)

        # fid=0: threshold=0.1, 0.6 > 0.1 → passes through
        r0 = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.allclose(r0, pre_act), f"fid=0 should pass through, got {r0}"

        # fid=1: threshold=0.5, 0.6 > 0.5 → passes through
        r1 = apply_sae_activation(pre_act, sae, feature_id=1)
        assert t.allclose(r1, pre_act), f"fid=1 should pass through, got {r1}"

        # fid=2: threshold=0.9, 0.6 < 0.9 → zeroed
        r2 = apply_sae_activation(pre_act, sae, feature_id=2)
        assert t.all(r2 == 0.0), f"fid=2 should be 0, got {r2}"

    def test_result_always_non_negative(self):
        sae = self._make_sae(n_features=5, threshold_val=0.0)
        # Mix of positive, zero, and negative
        pre_act = t.tensor([-2.0, -0.001, 0.0, 0.5, 3.0], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.all(result >= 0.0), f"Result must be >= 0, got {result}"


class TestApplySaeActivationReLU:
    """ReLU fallback when SAE has no threshold attribute."""

    def _make_relu_sae(self, n_features: int = 2) -> _ReLUSAE:
        d_model = 4
        W_enc = t.randn(d_model, n_features)
        b_enc = t.zeros(n_features)
        b_dec = t.zeros(d_model)
        return _ReLUSAE(W_enc, b_enc, b_dec)

    def test_positive_passes_through(self):
        sae = self._make_relu_sae()
        pre_act = t.tensor([0.1, 0.5, 2.0, 10.0], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.allclose(result, pre_act, atol=1e-6), (
            f"Positive values should pass through ReLU, got {result}"
        )

    def test_negative_zeroed(self):
        sae = self._make_relu_sae()
        pre_act = t.tensor([-3.0, -1.0, -0.001], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.all(result == 0.0), f"Negative values should be 0 after ReLU, got {result}"

    def test_zero_maps_to_zero(self):
        sae = self._make_relu_sae()
        pre_act = t.tensor([0.0, 0.0], dtype=t.float32)
        result = apply_sae_activation(pre_act, sae, feature_id=0)
        assert t.all(result == 0.0), f"Zero should remain 0 after ReLU, got {result}"


class TestApplySaeActivationEquivalence:
    """apply_sae_activation must match sae.encode(act)[:, fid] for same pre-act."""

    def _make_jumprelu_sae(self, d_model: int = 4, n_features: int = 3) -> _JumpReLUSAE:
        W_enc = t.randn(d_model, n_features)
        b_enc = t.randn(n_features)
        b_dec = t.randn(d_model)
        threshold = t.rand(n_features)
        return _JumpReLUSAE(W_enc, b_enc, b_dec, threshold)

    def _make_relu_sae(self, d_model: int = 4, n_features: int = 3) -> _ReLUSAE:
        W_enc = t.randn(d_model, n_features)
        b_enc = t.randn(n_features)
        b_dec = t.randn(d_model)
        return _ReLUSAE(W_enc, b_enc, b_dec)

    def test_equivalence_jumprelu(self):
        sae = self._make_jumprelu_sae()
        act = t.randn(5, 4)

        for fid in range(3):
            pre_act = compute_raw_probe(
                act, sae.W_enc, sae.b_enc, sae.b_dec, fid,
                apply_b_dec=sae.cfg.apply_b_dec_to_input,
            )
            result = apply_sae_activation(pre_act, sae, feature_id=fid)
            expected = sae.encode(act)[:, fid]
            assert t.allclose(result, expected, atol=1e-5), (
                f"JumpReLU fid={fid}: apply_sae_activation={result} != encode={expected}"
            )

    def test_equivalence_relu(self):
        sae = self._make_relu_sae()
        act = t.randn(5, 4)

        for fid in range(3):
            pre_act = compute_raw_probe(
                act, sae.W_enc, sae.b_enc, sae.b_dec, fid,
                apply_b_dec=sae.cfg.apply_b_dec_to_input,
            )
            result = apply_sae_activation(pre_act, sae, feature_id=fid)
            expected = sae.encode(act)[:, fid]
            assert t.allclose(result, expected, atol=1e-5), (
                f"ReLU fid={fid}: apply_sae_activation={result} != encode={expected}"
            )


class TestApplySaeActivationUnsupported:
    """TopK SAE must raise ValueError with 'Unsupported SAE activation'."""

    def test_topk_raises_value_error(self):
        sae = _TopKSAE()
        pre_act = t.tensor([1.0, 2.0, 3.0], dtype=t.float32)
        with pytest.raises(ValueError, match="Unsupported SAE activation.*topk"):
            apply_sae_activation(pre_act, sae, feature_id=0)


# ---------------------------------------------------------------------------
# Mock classes for _probe_roundtrip_raw tests
# ---------------------------------------------------------------------------


class _MockSAEConfig:
    apply_b_dec_to_input = True


class _MockSAEWithThreshold:
    """Mock SAE with JumpReLU threshold for probe model tests."""

    def __init__(self, d_model, n_features, threshold_value=0.5):
        self.W_enc = t.randn(d_model, n_features)
        self.b_enc = t.zeros(n_features)
        self.b_dec = t.zeros(d_model)
        self.threshold = t.full((n_features,), threshold_value)
        self.cfg = _MockSAEConfig()

    def parameters(self):
        return iter([self.W_enc, self.b_enc, self.b_dec])

    def to(self, device):
        return self


class _MockHookedTransformer:
    def __init__(self, d_model, hook_name):
        self._d_model = d_model
        self._hook_name = hook_name
        self._param = t.nn.Parameter(t.zeros(1))

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        return self

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None, attention_mask=None):
        B, T = tokens.shape
        act = t.randn(B, T, self._d_model)
        return None, {self._hook_name: act}


class _MockGemmaTokenizer:
    def encode(self, text):
        return list(text.encode("latin-1"))

    def decode(self, tokens):
        return bytes(t for t in tokens if 0 <= t < 256).decode("latin-1")


def _make_probe_model(d_model=32, n_features=50, threshold_value=0.5):
    hook = "blocks.8.hook_resid_post"
    probe = ProbeModel.__new__(ProbeModel)
    probe.model = _MockHookedTransformer(d_model, hook)
    probe.sae = _MockSAEWithThreshold(d_model, n_features, threshold_value)
    probe.tokenizer = _MockGemmaTokenizer()
    probe.hook_name = hook
    probe.layer = 8
    probe.apply_b_dec = True
    return probe


def _make_completions(conditions=("real", "random"), n_features=3):
    """Build a minimal list of RoundtripCompletion objects for testing."""
    samples = [
        CompletionSample(condition=cond, text=" hello world", seed=0)
        for cond in conditions
    ]
    comp = RoundtripCompletion.__new__(RoundtripCompletion)
    comp.doc_idx = 0
    comp.target_segment_idx = 0
    comp.pre_segment_text = "Some context text."
    comp.ground_truth_feature_ids = list(range(n_features))
    comp.samples = samples
    return [comp]


# ---------------------------------------------------------------------------
# Tests for _probe_roundtrip_raw post-act output
# ---------------------------------------------------------------------------


class TestProbeRoundtripRawPostAct:
    """_probe_roundtrip_raw must return (condition, pre_act, post_act, pct) 4-tuples."""

    def test_returns_three_element_tuples(self):
        probe = _make_probe_model()
        completions = _make_completions()
        scores = _probe_roundtrip_raw(completions, probe)
        assert len(scores) > 0, "Expected at least one score"
        for entry in scores:
            assert len(entry) == 4, (
                f"Expected 4-element tuple (condition, pre_act, post_act, pct), got {entry!r}"
            )

    def test_post_act_always_non_negative(self):
        probe = _make_probe_model(threshold_value=0.5)
        completions = _make_completions()
        scores = _probe_roundtrip_raw(completions, probe)
        for condition, _, post_val, _ in scores:
            assert post_val >= 0.0, (
                f"post_act must be >= 0, got {post_val} for condition={condition}"
            )

    def test_post_act_leq_pre_act_when_threshold_high(self):
        """With a very high threshold, JumpReLU gates everything to zero."""
        probe = _make_probe_model(threshold_value=100.0)
        completions = _make_completions()
        scores = _probe_roundtrip_raw(completions, probe)
        for condition, _, post_val, _ in scores:
            assert post_val < 1e-6, (
                f"With threshold=100, post_act should be ~0, got {post_val} "
                f"for condition={condition}"
            )

    def test_pre_act_values_finite(self):
        probe = _make_probe_model()
        completions = _make_completions()
        scores = _probe_roundtrip_raw(completions, probe)
        for condition, pre_val, _, _ in scores:
            assert math.isfinite(pre_val), (
                f"pre_act must be finite, got {pre_val} for condition={condition}"
            )

    def test_empty_completions_returns_empty(self):
        probe = _make_probe_model()
        scores = _probe_roundtrip_raw([], probe)
        assert scores == [], f"Expected empty list, got {scores!r}"

    def test_empty_gt_features_raises(self):
        """Empty gt_fids raises ValueError instead of silent skip.

        Earlier this test asserted silent-skip behavior. The user removed
        the silent skip: an empty ground_truth_feature_ids indicates a
        caller-side invariant violation (upstream selection should never
        produce a completion with zero features).
        """
        comp = RoundtripCompletion.__new__(RoundtripCompletion)
        comp.doc_idx = 0
        comp.target_segment_idx = 0
        comp.pre_segment_text = "context"
        comp.ground_truth_feature_ids = []  # no features
        comp.samples = [CompletionSample(condition="real", text=" text", seed=0)]
        probe = _make_probe_model()
        with pytest.raises(ValueError, match=r"empty\s+ground_truth_feature_ids"):
            _probe_roundtrip_raw([comp], probe)

    def test_whitespace_only_completion_raises(self):
        """Whitespace-only full_text raises ValueError instead of silent skip.

        Earlier this test asserted the silent-skip behavior. The user removed
        the silent skip: empty/whitespace-only full_text indicates a real
        upstream bug (degenerate pre_segment_text or sampler producing
        nothing) and the loud failure mode is the right default.
        """
        comp = RoundtripCompletion.__new__(RoundtripCompletion)
        comp.doc_idx = 0
        comp.target_segment_idx = 0
        comp.pre_segment_text = ""
        comp.ground_truth_feature_ids = [0]
        comp.samples = [CompletionSample(condition="real", text="   ", seed=0)]
        probe = _make_probe_model()
        with pytest.raises(ValueError, match=r"Empty/whitespace-only full_text"):
            _probe_roundtrip_raw([comp], probe)

    def test_expected_conditions_present(self):
        probe = _make_probe_model()
        completions = _make_completions(conditions=("real", "random"))
        scores = _probe_roundtrip_raw(completions, probe)
        conditions_found = {s[0] for s in scores}
        assert "real" in conditions_found, f"Missing 'real' in {conditions_found}"
        assert "random" in conditions_found, f"Missing 'random' in {conditions_found}"


class TestAggregateRoundtripMetricsPostAct:
    def test_both_metric_sets_present(self):
        """Output dict has both pre-act and post-act metric keys."""
        scores = [
            ("real", 5.0, 3.0, float("nan")), ("real", 3.0, 1.0, float("nan")),
            ("random", 2.0, 1.5, float("nan")), ("random", 1.0, 0.5, float("nan")),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=6, n_docs=2)
        # Pre-act keys
        assert "steering/roundtrip_real_mean" in m
        assert "steering/roundtrip_random_mean" in m
        assert "steering/roundtrip_specificity" in m
        # Post-act keys
        assert "steering/roundtrip_real_mean_post_act" in m
        assert "steering/roundtrip_random_mean_post_act" in m
        assert "steering/roundtrip_specificity_post_act" in m

    def test_pre_act_values_correct(self):
        scores = [
            ("real", 5.0, 3.0, float("nan")), ("real", 3.0, 1.0, float("nan")),
            ("random", 2.0, 1.5, float("nan")), ("random", 1.0, 0.5, float("nan")),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=6, n_docs=2)
        assert m["steering/roundtrip_real_mean"] == pytest.approx(4.0)
        assert m["steering/roundtrip_random_mean"] == pytest.approx(1.5)
        assert m["steering/roundtrip_specificity"] == pytest.approx(2.5)

    def test_post_act_values_correct(self):
        scores = [
            ("real", 5.0, 3.0, float("nan")), ("real", 3.0, 1.0, float("nan")),
            ("random", 2.0, 1.5, float("nan")), ("random", 1.0, 0.5, float("nan")),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=6, n_docs=2)
        assert m["steering/roundtrip_real_mean_post_act"] == pytest.approx(2.0)
        assert m["steering/roundtrip_random_mean_post_act"] == pytest.approx(1.0)
        assert m["steering/roundtrip_specificity_post_act"] == pytest.approx(1.0)

    def test_post_act_specificity_differs_from_pre_act(self):
        scores = [("real", 5.0, 0.0, float("nan")), ("random", 1.0, 0.0, float("nan"))]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=2, n_docs=1)
        assert m["steering/roundtrip_specificity"] == pytest.approx(4.0)
        assert m["steering/roundtrip_specificity_post_act"] == pytest.approx(0.0)

    def test_empty_scores_all_zero(self):
        m = _aggregate_roundtrip_metrics([], total_gt_features=0, n_docs=0)
        assert m["steering/roundtrip_real_mean"] == 0.0
        assert m["steering/roundtrip_real_mean_post_act"] == 0.0
        assert m["steering/roundtrip_specificity"] == 0.0
        assert m["steering/roundtrip_specificity_post_act"] == 0.0

    def test_shared_keys_not_duplicated(self):
        scores = [("real", 1.0, 0.5, float("nan"))]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=3, n_docs=1)
        assert m["steering/roundtrip_n_samples"] == 1
        assert m["steering/roundtrip_n_gt_features_avg"] == pytest.approx(3.0)
        assert "steering/roundtrip_n_samples_post_act" not in m

    def test_all_below_threshold_post_act_zero(self):
        scores = [
            ("real", 2.0, 0.0, float("nan")), ("random", 1.5, 0.0, float("nan")),
        ]
        m = _aggregate_roundtrip_metrics(scores, total_gt_features=3, n_docs=1)
        assert m["steering/roundtrip_real_mean_post_act"] == 0.0
        assert m["steering/roundtrip_specificity_post_act"] == 0.0


# ---------------------------------------------------------------------------
# Fixtures copied from tests/test_inline_steering_eval.py for e2e test
# ---------------------------------------------------------------------------


def _make_per_feature_stats(n_features: int) -> dict[str, PerFeatureStat]:
    """Create synthetic PerFeatureStat for features 0..(n_features-1)."""
    stats = {}
    for fid in range(n_features):
        pcts = {f"p{i}": float(i) / 100.0 * 10.0 for i in range(1, 100)}
        stats[str(fid)] = PerFeatureStat(
            count=100,
            pct_segments_active=0.5,
            mean_mag=5.0,
            mean_mag_incl_zeros=2.5,
            std_mag=2.0,
            min_mag=0.1,
            max_mag=10.0,
            median_mag=5.0,
            percentiles=pcts,
            histogram=None,
        )
    return stats


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend for all tests in this module."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


class ByteTokenizer:
    """Byte-level tokenizer where token IDs == Latin-1 byte values (0-255).

    BOS token is 0 (null byte), keeping all IDs within vocab_size=256.
    The Latin-1 mapping keeps synthetic generated tokens one-character-wide,
    so token_to_char_offsets() sees a reversible tokenizer instead of UTF-8
    replacement-character collapse.
    """

    def __init__(self):
        self._bos = 0

    def get_bos_token_id(self):
        return self._bos

    def get_vocab_size(self):
        return 256

    def encode(self, text, prepend=None, append=None, num_threads=None):
        tokens = list(text.encode("latin-1"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [tok for tok in tokens if 0 <= tok < 256]
        return bytes(byte_tokens).decode("latin-1")


@pytest.fixture
def byte_tokenizer():
    """A byte-level tokenizer suitable for vocab_size=256 models."""
    return ByteTokenizer()


@pytest.fixture
def tiny_model():
    """A tiny EncoderDecoderGPT: 2 layers, dim=64, vocab=256, seq_len=128."""
    dec_config = GPTConfig(
        sequence_len=128,
        vocab_size=256,
        n_layer=2,
        n_head=2,
        n_kv_head=2,
        n_embd=64,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16)
    xattn_config = CrossAttentionConfig(frequency=2, n_head=2, n_kv_head=2)
    rng_state = t.random.get_rng_state()
    t.manual_seed(0)
    try:
        model = EncoderDecoderGPT(
            dec_config,
            enc_config,
            xattn_config,
            n_sae_features=50,
            num_buckets=5,
        )
        model.init_weights()
    finally:
        t.random.set_rng_state(rng_state)
    model.eval()
    return model


@pytest.fixture
def feature_shard_dir():
    """Temp dir with one compressed feature shard.

    Shard contains 3 documents x 4 segments = 12 segments, each with 3
    features. Feature IDs 0, 1, 2 appear in every segment.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        n_docs = 3
        n_segs_per_doc = 4
        n_segs = n_docs * n_segs_per_doc  # 12
        features_per_seg = 3
        total_features = n_segs * features_per_seg  # 36

        # offsets: CSR boundaries over the flat feature arrays
        offsets = t.arange(0, total_features + 1, features_per_seg, dtype=t.int64)

        # doc_ids: segment i belongs to doc i // n_segs_per_doc
        doc_ids = t.repeat_interleave(
            t.arange(n_docs, dtype=t.int64),
            n_segs_per_doc,
        )

        # segment_indices: 0,1,2,3 repeating for each doc
        segment_indices = t.tile(
            t.arange(n_segs_per_doc, dtype=t.int64), (n_docs,)
        )

        # feature_ids: features 0,1,2 in every segment
        feature_ids = t.tile(
            t.tensor([0, 1, 2], dtype=t.int32), (n_segs,)
        )

        # feature_mags: vary by segment and feature position
        feature_mags = t.zeros(total_features, dtype=t.float32)
        for seg_i in range(n_segs):
            s = seg_i * features_per_seg
            feature_mags[s : s + features_per_seg] = t.tensor(
                [1.0 + seg_i * 0.1, 2.0 + seg_i * 0.1, 3.0 + seg_i * 0.1]
            )

        data = {
            "offsets": offsets,
            "doc_ids": doc_ids,
            "segment_indices": segment_indices,
            "feature_ids": feature_ids,
            "feature_mags": feature_mags,
            "n_segments": n_segs,
        }

        save_features(data, tmpdir_path / "shard_00000_features")
        yield tmpdir_path


@pytest.fixture
def per_feature_stats():
    """Synthetic PerFeatureStat for features 0..49."""
    return _make_per_feature_stats(50)


@pytest.fixture
def boundaries():
    """(50, 4) float64 discretization boundaries for 50 SAE features, 5 buckets."""
    return t.linspace(0.0, 10.0, steps=4).unsqueeze(0).expand(50, -1).double().clone()


@pytest.fixture
def threshold_config():
    """Segment top-k=3 threshold config."""
    return ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=3.0,
    )


@pytest.fixture
def precomputed_thresholds(per_feature_stats):
    """Precomputed threshold tensor: p50 value for each of 50 features."""
    thresholds = t.zeros(50, dtype=t.float64)
    for fid in range(50):
        stat = per_feature_stats[str(fid)]
        thresholds[fid] = stat.percentiles.get("p50", 5.0)
    return thresholds


@pytest.fixture
def val_parquet_dir():
    """Temp dir with a small validation parquet file (3 short documents)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "An encoder-decoder model learns to condition on SAE features.",
            "Sparse autoencoders extract interpretable features from neural networks.",
        ]
        table = pa.table({"text": texts})
        pq.write_table(table, tmpdir_path / "val_00000.parquet")
        yield tmpdir_path


# ---------------------------------------------------------------------------
# E2E test: evaluate_steering includes post-act metric keys
# ---------------------------------------------------------------------------


class TestEvaluateSteeringPostActKeys:
    """Verify evaluate_steering output includes post-act metric keys."""

    def test_e2e_has_post_act_keys(
        self, tiny_model, feature_shard_dir, byte_tokenizer,
        boundaries, threshold_config, precomputed_thresholds,
        val_parquet_dir,
    ):
        doc_text = "The quick brown fox jumps over the lazy dog."
        bos_prefix_len = 1
        doc_len = len(doc_text)
        quarter = doc_len // 4
        seg_bounds = [
            (0, bos_prefix_len + quarter),
            (bos_prefix_len + quarter, bos_prefix_len + 2 * quarter),
            (bos_prefix_len + 2 * quarter, bos_prefix_len + 3 * quarter),
            (bos_prefix_len + 3 * quarter, bos_prefix_len + doc_len),
        ]

        def _make_segments():
            return [
                SegmentFeatures(
                    segment_idx=i, position=start,
                    feature_ids=t.tensor([0, 1, 2], dtype=t.int64),
                    feature_mags=t.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=t.float32),
                    char_start=start, char_end=end,
                    token_start=start, token_end=end,
                )
                for i, (start, end) in enumerate(seg_bounds)
            ]

        class MockFeatureIndex:
            def lookup(self, shard_stem, doc_idx):
                return _make_segments() if 0 <= doc_idx <= 2 else None
            def doc_ids_for_shard(self, shard_stem):
                return [0, 1, 2]

        probe = _make_probe_model()

        result = evaluate_steering(
            model=tiny_model,
            feature_dir=feature_shard_dir,
            tokenizer=byte_tokenizer,
            device=t.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            data_dir=str(val_parquet_dir),
            feature_index=MockFeatureIndex(),  # ty: ignore[invalid-argument-type]
            probe_model=probe,
        )

        # Pre-act keys
        assert "steering/roundtrip_real_mean" in result
        assert "steering/roundtrip_random_mean" in result
        assert "steering/roundtrip_specificity" in result
        # Post-act keys
        assert "steering/roundtrip_real_mean_post_act" in result
        assert "steering/roundtrip_random_mean_post_act" in result
        assert "steering/roundtrip_specificity_post_act" in result
        # All values finite
        for key, value in result.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} = {value}"


# ---------------------------------------------------------------------------
# P1: compute_raw_probe_batched tests
# ---------------------------------------------------------------------------


def test_compute_raw_probe_batched_shape():
    """compute_raw_probe_batched returns shape (seq_len, n_feats)."""
    seq_len, d_model, n_features = 7, 4, 10
    act = t.randn(seq_len, d_model)
    W_enc = t.randn(d_model, n_features)
    b_enc = t.randn(n_features)
    b_dec = t.randn(d_model)
    feature_ids = t.tensor([0, 3, 7], dtype=t.int64)

    result = compute_raw_probe_batched(act, W_enc, b_enc, b_dec, feature_ids)
    assert result.shape == (seq_len, 3)


def test_compute_raw_probe_batched_equals_scalar_stack():
    """Batched output == stack([scalar(f) for f in feature_ids]) within tolerance.

    Uses float64 so float32 matmul accumulation error doesn't cause false failures
    at any d_model. The tolerance (rtol=1e-6, atol=1e-9) reflects the algebraic
    identity that this test is asserting, not accumulated rounding error.
    """
    t.manual_seed(42)
    seq_len, d_model, n_features = 16, 8, 32
    act = t.randn(seq_len, d_model, dtype=t.float64)
    W_enc = t.randn(d_model, n_features, dtype=t.float64)
    b_enc = t.randn(n_features, dtype=t.float64)
    b_dec = t.randn(d_model, dtype=t.float64)
    feature_ids = t.tensor([2, 5, 11, 17, 29], dtype=t.int64)

    batched = compute_raw_probe_batched(act, W_enc, b_enc, b_dec, feature_ids, apply_b_dec=True)
    expected = t.stack(
        [compute_raw_probe(act, W_enc, b_enc, b_dec, int(fid), apply_b_dec=True) for fid in feature_ids],
        dim=1,
    )
    assert t.allclose(batched, expected, rtol=1e-6, atol=1e-9)


def test_compute_raw_probe_batched_apply_b_dec_false_branch():
    """apply_b_dec=False branch produces numerically correct output matching the scalar version.

    Uses float64 for the same reason as test_compute_raw_probe_batched_equals_scalar_stack.
    """
    t.manual_seed(123)
    act = t.randn(5, 4, dtype=t.float64)
    W_enc = t.randn(4, 8, dtype=t.float64)
    b_enc = t.randn(8, dtype=t.float64)
    b_dec = t.randn(4, dtype=t.float64)  # should be ignored with apply_b_dec=False
    feature_ids = t.tensor([1, 4], dtype=t.int64)

    batched = compute_raw_probe_batched(act, W_enc, b_enc, b_dec, feature_ids, apply_b_dec=False)
    expected = t.stack(
        [compute_raw_probe(act, W_enc, b_enc, b_dec, int(fid), apply_b_dec=False) for fid in feature_ids],
        dim=1,
    )
    assert t.allclose(batched, expected, rtol=1e-6, atol=1e-9)


def test_compute_raw_probe_batched_empty_feature_ids():
    """Empty feature_ids returns shape (seq_len, 0) without crashing."""
    act = t.randn(10, 4)
    W_enc = t.randn(4, 8)
    b_enc = t.randn(8)
    b_dec = t.randn(4)
    feature_ids = t.zeros(0, dtype=t.int64)

    result = compute_raw_probe_batched(act, W_enc, b_enc, b_dec, feature_ids)
    assert result.shape == (10, 0)


# ---------------------------------------------------------------------------
# P1: apply_sae_activation_batched tests
# ---------------------------------------------------------------------------


class _P1MockSAEWithThreshold:
    """Minimal SAE-like object with a per-feature threshold tensor (JumpReLU)."""

    def __init__(self, n_features, threshold_values=None):
        if threshold_values is not None:
            self.threshold = t.as_tensor(threshold_values)
        else:
            self.threshold = t.zeros(n_features)
        self.cfg = type("Cfg", (), {"activation_fn": "jumprelu"})()


class _P1MockSAEReLU:
    """Minimal SAE-like object with ReLU activation (no threshold)."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"activation_fn": "relu"})()


class _P1MockSAEUnsupported:
    """Minimal SAE-like object with an unsupported activation function."""

    def __init__(self):
        self.cfg = type("Cfg", (), {"activation_fn": "topk"})()


def test_apply_sae_activation_batched_jumprelu_parity():
    """Batched JumpReLU matches scalar JumpReLU per feature.

    Uses float64 for the pre-activation tensor so the comparison is an
    algebraic identity rather than a fragile empirical threshold.
    """
    t.manual_seed(7)
    seq_len, n_feats = 10, 5
    n_features = 20
    thresholds = t.linspace(0.1, 0.9, n_features, dtype=t.float64)
    sae = _P1MockSAEWithThreshold(n_features, thresholds)
    feature_ids = t.tensor([1, 4, 7, 13, 19], dtype=t.int64)
    pre_act = t.randn(seq_len, n_feats, dtype=t.float64) * 0.5  # span positive and negative

    batched = apply_sae_activation_batched(pre_act, sae, feature_ids)
    expected_cols = [
        apply_sae_activation(pre_act[:, i], sae, int(feature_ids[i])) for i in range(n_feats)
    ]
    expected = t.stack(expected_cols, dim=1)
    assert t.allclose(batched, expected, rtol=1e-6, atol=1e-9)


def test_apply_sae_activation_batched_relu_parity():
    """Batched ReLU fallback matches scalar ReLU per feature.

    Uses float64 for the same algebraic-identity reasoning.
    """
    t.manual_seed(11)
    seq_len, n_feats = 8, 4
    sae = _P1MockSAEReLU()
    feature_ids = t.tensor([0, 1, 2, 3], dtype=t.int64)
    pre_act = t.randn(seq_len, n_feats, dtype=t.float64)

    batched = apply_sae_activation_batched(pre_act, sae, feature_ids)
    expected_cols = [
        apply_sae_activation(pre_act[:, i], sae, int(feature_ids[i])) for i in range(n_feats)
    ]
    expected = t.stack(expected_cols, dim=1)
    assert t.allclose(batched, expected, rtol=1e-6, atol=1e-9)


def test_apply_sae_activation_batched_unsupported_raises_same_message():
    """Batched path raises ValueError with the same message as scalar path."""
    sae = _P1MockSAEUnsupported()
    pre_act_scalar = t.randn(10)
    pre_act_batched = t.randn(10, 2)
    feature_ids = t.tensor([0, 1], dtype=t.int64)

    # Capture scalar error message
    try:
        apply_sae_activation(pre_act_scalar, sae, feature_id=0)
    except ValueError as e:
        scalar_msg = str(e)
    else:
        pytest.fail("Scalar apply_sae_activation should have raised ValueError")

    with pytest.raises(ValueError) as exc_info:
        apply_sae_activation_batched(pre_act_batched, sae, feature_ids)
    assert str(exc_info.value) == scalar_msg


def test_apply_sae_activation_batched_preserves_shape():
    """Input shape (seq_len, n_feats) produces output shape (seq_len, n_feats)."""
    sae = _P1MockSAEWithThreshold(n_features=32)
    pre_act = t.randn(50, 7)
    feature_ids = t.arange(7, dtype=t.int64)

    result = apply_sae_activation_batched(pre_act, sae, feature_ids)
    assert result.shape == (50, 7)


def test_apply_sae_activation_batched_jumprelu_respects_per_feature_threshold():
    """JumpReLU gate fires iff pre_act > threshold[fid] for each column independently."""
    # Per-feature threshold: feature i has threshold i
    sae = _P1MockSAEWithThreshold(n_features=10, threshold_values=[float(i) for i in range(10)])
    feature_ids = t.tensor([0, 1, 2, 3], dtype=t.int64)
    # pre_act: row 0 is 0.5, row 1 is 1.5, row 2 is 2.5, row 3 is 3.5 (all columns)
    pre_act = t.tensor([
        [0.5, 0.5, 0.5, 0.5],
        [1.5, 1.5, 1.5, 1.5],
        [2.5, 2.5, 2.5, 2.5],
        [3.5, 3.5, 3.5, 3.5],
    ])

    result = apply_sae_activation_batched(pre_act, sae, feature_ids)
    # Column 0: threshold=0, gate fires when pre > 0 → [0.5, 1.5, 2.5, 3.5]
    # Column 1: threshold=1, gate fires when pre > 1 → [0, 1.5, 2.5, 3.5]
    # Column 2: threshold=2, gate fires when pre > 2 → [0, 0, 2.5, 3.5]
    # Column 3: threshold=3, gate fires when pre > 3 → [0, 0, 0, 3.5]
    expected = t.tensor([
        [0.5, 0.0, 0.0, 0.0],
        [1.5, 1.5, 0.0, 0.0],
        [2.5, 2.5, 2.5, 0.0],
        [3.5, 3.5, 3.5, 3.5],
    ])
    assert t.allclose(result, expected, rtol=1e-6, atol=1e-9)


def test_apply_sae_activation_batched_empty_feature_ids():
    """Empty feature_ids returns shape (seq_len, 0) without crashing.

    Mirrors test_compute_raw_probe_batched_empty_feature_ids for the
    apply_sae_activation_batched helper.
    """
    sae = _P1MockSAEWithThreshold(n_features=10)
    pre_act = t.randn(5, 0)
    feature_ids = t.zeros(0, dtype=t.int64)

    result = apply_sae_activation_batched(pre_act, sae, feature_ids)
    assert result.shape == (5, 0)


# ---------------------------------------------------------------------------
# P1: max_pool_masked_batched tests
# ---------------------------------------------------------------------------


def test_max_pool_masked_batched_shape():
    """Input (seq_len, n_feats) + (seq_len,) → output (n_feats,)."""
    probes = t.randn(10, 5)
    mask = t.tensor([True, False, True, True, False, True, True, False, True, True])

    result = max_pool_masked_batched(probes, mask)
    assert result.shape == (5,)


def test_max_pool_masked_batched_equals_scalar_per_column():
    """Batched result equals the stack of scalar max_pool_masked across columns.

    Uses float64 for the probes tensor so the comparison is an exact algebraic
    identity rather than a fragile float32 empirical check.
    """
    t.manual_seed(17)
    seq_len, n_feats = 25, 7
    probes = t.randn(seq_len, n_feats, dtype=t.float64)
    mask = t.rand(seq_len) > 0.3  # ~70% True
    # Guarantee at least one True element to avoid triggering the all-False path
    if not mask.any():
        mask[0] = True

    batched = max_pool_masked_batched(probes, mask)
    expected = t.tensor(
        [max_pool_masked(probes[:, i], mask) for i in range(n_feats)],
        dtype=batched.dtype,
    )
    assert t.allclose(batched, expected, rtol=1e-6, atol=1e-9)


def test_max_pool_masked_batched_empty_mask_returns_nan_tensor():
    """All-False mask returns a tensor of NaN with shape (n_feats,)."""
    probes = t.randn(10, 4)
    mask = t.zeros(10, dtype=t.bool)

    result = max_pool_masked_batched(probes, mask)

    assert result.shape == (4,)
    assert t.isnan(result).all()


def test_max_pool_masked_batched_single_true_position():
    """Mask with exactly one True returns that row as the per-column max."""
    probes = t.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],  # this row is the only True
        [7.0, 8.0, 9.0],
    ])
    mask = t.tensor([False, True, False])

    result = max_pool_masked_batched(probes, mask)
    assert t.allclose(result, t.tensor([4.0, 5.0, 6.0]))


def test_max_pool_masked_batched_preserves_device_and_dtype():
    """Output device and dtype match the input probes tensor."""
    probes = t.randn(8, 3, dtype=t.float64)
    mask = t.tensor([True, True, False, True, False, True, True, True])

    result = max_pool_masked_batched(probes, mask)
    assert result.dtype == t.float64
    assert result.device == probes.device


# ---------------------------------------------------------------------------
# P1: property tests for batched probe helpers
# ---------------------------------------------------------------------------


@given(
    seq_len=st.integers(min_value=1, max_value=32),
    d_model=st.integers(min_value=2, max_value=16),
    n_all_features=st.integers(min_value=4, max_value=32),
    n_picked=st.integers(min_value=1, max_value=4),
    apply_b_dec=st.booleans(),
)
@settings(max_examples=20, deadline=None)
def test_property_compute_raw_probe_batched_equals_scalar_stack(
    seq_len, d_model, n_all_features, n_picked, apply_b_dec,
):
    """Property: batched probe equals stacked scalar probes, for any valid shape.

    Uses float64 so cross-method rounding error is negligible and the tight
    tolerance (rtol=1e-5, atol=1e-7) reflects the algebraic identity.
    """
    n_picked = min(n_picked, n_all_features)
    t.manual_seed(hash((seq_len, d_model, n_all_features, n_picked, apply_b_dec)) & 0x7FFFFFFF)
    act = t.randn(seq_len, d_model, dtype=t.float64)
    W_enc = t.randn(d_model, n_all_features, dtype=t.float64)
    b_enc = t.randn(n_all_features, dtype=t.float64)
    b_dec = t.randn(d_model, dtype=t.float64)
    feature_ids = t.randperm(n_all_features)[:n_picked].to(t.int64)

    batched = compute_raw_probe_batched(
        act, W_enc, b_enc, b_dec, feature_ids, apply_b_dec=apply_b_dec,
    )
    expected = t.stack(
        [
            compute_raw_probe(act, W_enc, b_enc, b_dec, int(fid), apply_b_dec=apply_b_dec)
            for fid in feature_ids
        ],
        dim=1,
    )
    assert t.allclose(batched, expected, rtol=1e-5, atol=1e-7)


@given(
    seq_len=st.integers(min_value=1, max_value=32),
    n_all_features=st.integers(min_value=4, max_value=32),
    n_picked=st.integers(min_value=1, max_value=4),
    use_threshold=st.booleans(),
)
@settings(max_examples=20, deadline=None)
def test_property_apply_sae_activation_batched_equals_scalar_stack(
    seq_len, n_all_features, n_picked, use_threshold,
):
    """Property: batched SAE activation equals stacked scalar activations.

    Covers both JumpReLU (use_threshold=True) and ReLU fallback (False).
    Uses float64 for the pre_act tensor.
    """
    n_picked = min(n_picked, n_all_features)
    t.manual_seed(hash((seq_len, n_all_features, n_picked, use_threshold)) & 0x7FFFFFFF)

    if use_threshold:
        sae = _P1MockSAEWithThreshold(
            n_all_features,
            threshold_values=t.rand(n_all_features, dtype=t.float64) * 2 - 1,
        )
    else:
        sae = _P1MockSAEReLU()

    feature_ids = t.randperm(n_all_features)[:n_picked].to(t.int64)
    pre_act = t.randn(seq_len, n_picked, dtype=t.float64)

    batched = apply_sae_activation_batched(pre_act, sae, feature_ids)
    expected = t.stack(
        [
            apply_sae_activation(pre_act[:, i], sae, int(feature_ids[i]))
            for i in range(n_picked)
        ],
        dim=1,
    )
    assert t.allclose(batched, expected, rtol=1e-5, atol=1e-7)


@given(
    seq_len=st.integers(min_value=2, max_value=32),
    n_feats=st.integers(min_value=1, max_value=8),
    mask_density=st.floats(min_value=0.2, max_value=1.0),
)
@settings(max_examples=20, deadline=None)
def test_property_max_pool_masked_batched_equals_scalar_per_column(
    seq_len, n_feats, mask_density,
):
    """Property: batched masked max equals stacked scalar masked max per column.

    Mask density floor of 0.2 ensures the mask is not all-False (that path
    has its own dedicated unit test). Position 0 is force-enabled as a
    safety net against randomness.
    """
    t.manual_seed(hash((seq_len, n_feats, int(mask_density * 100))) & 0x7FFFFFFF)
    probes = t.randn(seq_len, n_feats, dtype=t.float64)
    # Ensure at least one True by forcing position 0
    mask = t.rand(seq_len) < mask_density
    mask[0] = True

    batched = max_pool_masked_batched(probes, mask)
    expected = t.tensor(
        [max_pool_masked(probes[:, i], mask) for i in range(n_feats)],
        dtype=t.float64,
    )
    assert t.allclose(batched, expected, rtol=1e-6, atol=1e-9)
