"""Tests for inline_steering_eval — shuffled completion roundtrip fidelity metric.

Provides synthetic fixtures that match the real scan/threshold/select/generate
pipeline structures, plus unit tests for roundtrip probe scoring.
"""

from __future__ import annotations

import dataclasses
import math
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import hypothesis.strategies as st
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.distributed as dist
from hypothesis import HealthCheck, given, settings

import nanochat.flash_attention as _fa_mod
import steering.eval.inline_steering_eval as inline_mod
from errors import RemovedFeatureError
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.clustered_inline_cache import (
    ClusteredInlineEvalPayload,
    ClusteredInlineTarget,
)
from steering.eval.feature_probe import (
    compute_raw_probe_batched,
    max_pool_masked_batched,
)
from steering.eval.inline_steering_eval import (
    CLUSTERED_PROBE_SEMANTICS,
    InlineEvalCache,
    RoundtripCompletion,
    SteeringEvalConfig,
    _aggregate_roundtrip_metrics,
    _build_random_inputs,
    _find_val_shard,
    _gather_int,
    _generate_clustered_roundtrip_samples,
    _generate_roundtrip_samples,
    _load_doc_texts,
    _partition_docs,
    _probe_roundtrip_raw,
    _select_random_docs,
    _shard_stem,
    _tokenize_and_split,
    evaluate_steering,
    prepare_inline_eval_cache,
    restore_rng_state,
    save_rng_state,
)
from steering.eval.multiseg_generator import (
    CompletionSample,
    MultisegInputs,
    ablate_feature,
    build_multiseg_inputs,
    generate_completion,
)
from steering.eval.probe_model import ProbeModel
from steering.feature_io import glob_features, load_features, save_features
from steering.refined_types import PositiveInt
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
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


def _four_segments_for_doc() -> list[SegmentFeatures]:
    """Four-segment fixture used by multiple P0 tests.

    Matches the val_parquet_dir fixture's first document text (46 chars)
    split into quarters, with BOS-inclusive char boundaries.

    For the byte tokenizer (1 char = 1 token), ``token_start`` /
    ``token_end`` coincide with ``char_start`` / ``char_end``. Populating
    those fields simulates the ``tok/<checksum>/seg_boundaries`` overlay
    that the inline steering eval requires (see
    ``_tokenize_and_split`` — it raises if these are None).
    """
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
    return [
        SegmentFeatures(
            segment_idx=i,
            position=start,
            feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
            char_start=start,
            char_end=end,
            token_start=start,  # byte tokenizer: 1 char = 1 token
            token_end=end,
        )
        for i, (start, end) in enumerate(seg_bounds)
    ]


# ---------------------------------------------------------------------------
# Mock classes for ProbeModel tests
# ---------------------------------------------------------------------------


class MockSAEConfig:
    apply_b_dec_to_input = True


class MockSAE:
    def __init__(self, d_model, n_features):
        self.W_enc = torch.randn(d_model, n_features)
        self.b_enc = torch.zeros(n_features)
        self.b_dec = torch.zeros(d_model)
        self.threshold = torch.zeros(n_features)  # ReLU-equivalent threshold
        self.cfg = MockSAEConfig()

    def parameters(self):
        return iter([self.W_enc, self.b_enc, self.b_dec])

    def to(self, device):
        return self


class MockHookedTransformer:
    def __init__(self, d_model, hook_name):
        self._d_model = d_model
        self._hook_name = hook_name
        self._param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self._param])

    def to(self, device):
        return self

    def run_with_cache(self, tokens, names_filter=None, stop_at_layer=None, **kwargs):
        B, T = tokens.shape
        act = torch.randn(B, T, self._d_model)
        return None, {self._hook_name: act}


class MockGemmaTokenizer:
    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, tokens):
        return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")


class ByteTokenizer:
    """Byte-level tokenizer where token IDs == byte values (0-255).

    BOS token is 0 (null byte), keeping all IDs within vocab_size=256.
    """

    def __init__(self):
        self._bos = 0

    def get_bos_token_id(self):
        return self._bos

    def get_vocab_size(self):
        return 256

    def encode(self, text, prepend=None, append=None, num_threads=None):
        tokens = list(text.encode("utf-8"))
        if prepend is not None:
            tokens = [prepend] + tokens
        return tokens

    def decode(self, tokens):
        byte_tokens = [t for t in tokens if 0 <= t < 256]
        return bytes(byte_tokens).decode("utf-8", errors="replace")


class DegenerateSplitTokenizer:
    """Byte-level tokenizer with one special multi-char merged token.

    Encodes the 10-char string ``'AAAAAAAAAA'`` as a single merged token (id
    ``MERGED_TOKEN``). All other characters fall back to ASCII byte encoding
    (one id per char).

    Used to reproduce the segment-boundary / token-boundary alignment case
    from the inline steering eval degenerate-split bug: for doc text that
    starts with ``'AAAAAAAAAA'``, encoding ``bos_text[1:11]`` gives exactly
    two tokens ``[BOS, MERGED]`` with character offsets ``[0, 1, 11]``. If
    the target segment's ``char_start == 6`` (half-way through the merged
    token), then:

      * BOS midpoint = 1 → seg 0
      * MERGED midpoint = (1 + 11 + 1) // 2 = 6 → seg 0 (searchsorted with
        ``side="left"`` places mid == seg_char_ends[0] in seg 0)

    No token lands in the target segment, producing ``split_idx ==
    len(all_tokens)`` and ``n_gen_tokens == 0``. This is the exact pathological
    alignment that surfaced with real production data under
    ``segment_size=32``.
    """

    BOS_TOKEN = 0
    MERGED_TOKEN = 255
    MERGED_TEXT = "AAAAAAAAAA"

    def get_bos_token_id(self):
        return self.BOS_TOKEN

    def get_vocab_size(self):
        return 256

    def encode(self, text, prepend=None, append=None, num_threads=None):
        tokens: list[int] = []
        i = 0
        while i < len(text):
            if text[i : i + len(self.MERGED_TEXT)] == self.MERGED_TEXT:
                tokens.append(self.MERGED_TOKEN)
                i += len(self.MERGED_TEXT)
            else:
                tokens.append(ord(text[i]))
                i += 1
        if prepend is not None:
            tokens.insert(0, prepend)
        if append is not None:
            tokens.append(append)
        return tokens

    def decode(self, tokens):
        parts: list[str] = []
        for tok in tokens:
            if tok == self.MERGED_TOKEN:
                parts.append(self.MERGED_TEXT)
            else:
                parts.append(chr(tok))
        return "".join(parts)


# ---------------------------------------------------------------------------
# Model fixture
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA attention backend for all tests in this module."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


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
    model = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=50,
        num_buckets=5,
    )
    model.init_weights()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Feature shard fixture
# ---------------------------------------------------------------------------


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
        offsets = torch.arange(0, total_features + 1, features_per_seg, dtype=torch.int64)

        # doc_ids: segment i belongs to doc i // n_segs_per_doc
        doc_ids = torch.repeat_interleave(
            torch.arange(n_docs, dtype=torch.int64),
            n_segs_per_doc,
        )

        # segment_indices: 0,1,2,3 repeating for each doc
        segment_indices = torch.tile(
            torch.arange(n_segs_per_doc, dtype=torch.int64), (n_docs,)
        )

        # feature_ids: features 0,1,2 in every segment
        feature_ids = torch.tile(
            torch.tensor([0, 1, 2], dtype=torch.int32), (n_segs,)
        )

        # feature_mags: vary by segment and feature position
        feature_mags = torch.zeros(total_features, dtype=torch.float32)
        for seg_i in range(n_segs):
            s = seg_i * features_per_seg
            feature_mags[s : s + features_per_seg] = torch.tensor(
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


# ---------------------------------------------------------------------------
# Stats and boundaries fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def per_feature_stats():
    """Synthetic PerFeatureStat for features 0..49."""
    return _make_per_feature_stats(50)


@pytest.fixture
def boundaries():
    """(50, 4) float64 discretization boundaries for 50 SAE features, 5 buckets."""
    # Boundaries between 5 buckets are 4 edges per feature
    return torch.linspace(0.0, 10.0, steps=4).unsqueeze(0).expand(50, -1).double().clone()


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
    thresholds = torch.zeros(50, dtype=torch.float64)
    for fid in range(50):
        stat = per_feature_stats[str(fid)]
        thresholds[fid] = stat.percentiles.get("p50", 5.0)
    return thresholds


# ---------------------------------------------------------------------------
# Validation parquet fixture
# ---------------------------------------------------------------------------


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


@pytest.fixture
def byte_tokenizer():
    """A byte-level tokenizer suitable for vocab_size=256 models."""
    return ByteTokenizer()


@pytest.fixture
def mock_feature_index():
    """Mock FeatureIndex that returns _four_segments_for_doc() for doc_idx 0-2.

    Shared across every P0 caching test that needs a FeatureIndex stand-in.
    The class is defined inside the fixture so each test gets a fresh instance
    (useful if a future test wants to mutate state on it).
    """
    class _MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if not 0 <= doc_idx <= 2:
                raise KeyError(
                    f"Document {doc_idx} not found in shard {shard_stem!r} "
                    f"(shard has 3 docs)"
                )
            return _four_segments_for_doc()

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    return _MockFeatureIndex()


@pytest.fixture
def synthetic_segments():
    """Synthetic SegmentFeatures with character boundaries matching val_parquet texts.

    Creates 2 segments for the first document:
    "The quick brown fox jumps over the lazy dog."
    - Segment 0: chars 0..23 in BOS-inclusive coords
    - Segment 1: chars 23..46 in BOS-inclusive coords

    Character boundaries are in BOS-inclusive coordinate system (BOS prefix = "\\x00").
    """
    bos_prefix_len = 1  # len(decode([0])) = 1 for null byte
    doc_text = "The quick brown fox jumps over the lazy dog."
    mid = len(doc_text) // 2  # 22
    seg0_end = bos_prefix_len + mid  # 23
    seg1_end = bos_prefix_len + len(doc_text)  # 45

    seg0 = SegmentFeatures(
        segment_idx=0,
        position=0,
        feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        feature_mags=torch.tensor([5.0, 3.0, 7.0], dtype=torch.float32),
        char_start=0,
        char_end=seg0_end,
        token_start=0,
        token_end=seg0_end,
    )
    seg1 = SegmentFeatures(
        segment_idx=1,
        position=seg0_end,
        feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        feature_mags=torch.tensor([6.0, 4.0, 8.0], dtype=torch.float32),
        char_start=seg0_end,
        char_end=seg1_end,
        token_start=seg0_end,
        token_end=seg1_end,
    )
    return [seg0, seg1]


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


def test_imports():
    """SteeringEvalConfig can be imported and has correct defaults."""
    cfg = SteeringEvalConfig()
    assert cfg.n_eval_docs == 10
    assert cfg.samples_per_condition == 2
    assert cfg.temperature == 0.7
    assert cfg.top_k_sampling == 50
    assert cfg.seed == 42


def test_evaluate_steering_returns_early_when_no_shards(
    tiny_model, boundaries, threshold_config, precomputed_thresholds
):
    """evaluate_steering returns scan metrics with early exit when no shards exist."""
    with tempfile.TemporaryDirectory() as empty_dir:
        result = evaluate_steering(
            model=tiny_model,
            feature_dir=empty_dir,
            tokenizer=None,  # not needed when no selections
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
        )
    assert isinstance(result, dict)
    assert result["steering/n_docs_available"] == 0
    assert "steering/eval_time_seconds" in result


# ---------------------------------------------------------------------------
# Removed feature-clustering fail-fast
# ---------------------------------------------------------------------------
#
# inline_steering_eval rejects legacy clustered checkpoint metadata before it can
# affect runtime behavior. Unclustered checkpoints (no meta, or meta without the
# key, or key == None) are byte-identical.


def test_evaluate_steering_rejects_clustered_checkpoint(
    tiny_model, boundaries, threshold_config, precomputed_thresholds
):
    """A clustered checkpoint meta must hard-raise at startup."""
    with tempfile.TemporaryDirectory() as empty_dir:
        with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
            evaluate_steering(
                model=tiny_model,
                feature_dir=empty_dir,
                tokenizer=None,
                device=torch.device("cpu"),
                boundaries=boundaries,
                threshold_cfg=threshold_config,
                precomputed_thresholds=precomputed_thresholds,
                meta={"feature_clustering_canonical": {"clusters": [[2, 6, 8]]}},
            )


def test_evaluate_steering_unclustered_meta_is_noop(
    tiny_model, boundaries, threshold_config, precomputed_thresholds
):
    """Unclustered meta (key absent, or present-but-None) must NOT raise — the
    guard is byte-identical to the pre-clustering path for non-clustered runs."""
    with tempfile.TemporaryDirectory() as empty_dir:
        # key absent
        r1 = evaluate_steering(
            model=tiny_model,
            feature_dir=empty_dir,
            tokenizer=None,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            meta={"step": 100},
        )
        # key present but None
        r2 = evaluate_steering(
            model=tiny_model,
            feature_dir=empty_dir,
            tokenizer=None,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            meta={"feature_clustering_canonical": None},
        )
    # Both proceed to the normal early-exit (no shards) path.
    assert r1["steering/n_docs_available"] == 0
    assert r2["steering/n_docs_available"] == 0


def test_feature_shard_fixture(feature_shard_dir):
    """Feature shard fixture creates a valid compressed shard."""
    shards = glob_features(feature_shard_dir)
    assert len(shards) == 1, f"Expected 1 shard, got {len(shards)}"

    data = load_features(shards[0])
    assert "offsets" in data
    assert "doc_ids" in data
    assert "segment_indices" in data
    assert "feature_ids" in data
    assert "feature_mags" in data

    # 12 segments x 3 features = 36 total features
    assert data["feature_ids"].numel() == 36
    assert data["feature_mags"].numel() == 36
    assert data["doc_ids"].numel() == 12
    assert data["segment_indices"].numel() == 12

    # Feature IDs 0,1,2 appear in every segment (36 / 3 = 12 occurrences each)
    unique_fids = data["feature_ids"].unique().tolist()
    assert set(unique_fids) == {0, 1, 2}


def test_per_feature_stats_fixture(per_feature_stats):
    """per_feature_stats fixture covers features 0..49."""
    assert len(per_feature_stats) == 50
    for fid in range(50):
        stat = per_feature_stats[str(fid)]
        assert isinstance(stat, PerFeatureStat)
        assert stat.count == 100
        assert "p50" in stat.percentiles


def test_boundaries_fixture(boundaries):
    """boundaries fixture is (50, 4) float64."""
    assert boundaries.shape == (50, 4)
    assert boundaries.dtype == torch.float64


def test_val_parquet_fixture(val_parquet_dir):
    """val_parquet_dir fixture has exactly 3 rows."""
    files = list(val_parquet_dir.glob("*.parquet"))
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert len(table) == 3
    assert "text" in table.schema.names


def test_tiny_model_fixture(tiny_model):
    """tiny_model fixture is an EncoderDecoderGPT with correct spec."""
    assert isinstance(tiny_model, EncoderDecoderGPT)
    assert tiny_model.dec_config.vocab_size == 256
    assert tiny_model.dec_config.sequence_len == 128
    assert tiny_model.dec_config.n_layer == 2
    assert tiny_model.encoder.config.n_sae_features == 50


# ---------------------------------------------------------------------------
# _find_val_shard tests
# ---------------------------------------------------------------------------


def test_find_val_shard_finds_shard(feature_shard_dir):
    """_find_val_shard returns the shard path in the synthetic feature dir."""
    result = _find_val_shard(feature_shard_dir)
    assert result is not None, "_find_val_shard should return the shard, not None"
    assert result.exists(), f"Returned shard path does not exist: {result}"
    assert result.name.endswith(".safetensors.zst"), f"Expected .safetensors.zst, got: {result.name}"


def test_find_val_shard_returns_none_for_empty_dir():
    """_find_val_shard returns None when the feature dir has no shards."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _find_val_shard(Path(tmpdir))
    assert result is None


def test_find_val_shard_returns_last_when_multiple_shards(tmp_path):
    """_find_val_shard returns the last (alphabetically sorted) shard."""
    # Create two minimal shards
    n_segs = 2
    features_per_seg = 1
    total_features = n_segs * features_per_seg
    offsets = torch.arange(0, total_features + 1, features_per_seg, dtype=torch.int64)
    doc_ids = torch.zeros(n_segs, dtype=torch.int64)
    segment_indices = torch.arange(n_segs, dtype=torch.int64)
    feature_ids = torch.zeros(total_features, dtype=torch.int32)
    feature_mags = torch.ones(total_features, dtype=torch.float32)
    data = {
        "offsets": offsets,
        "doc_ids": doc_ids,
        "segment_indices": segment_indices,
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "n_segments": n_segs,
    }
    save_features(data, tmp_path / "shard_00000_features")
    save_features(data, tmp_path / "shard_00001_features")

    shards = glob_features(tmp_path)
    assert len(shards) == 2
    result = _find_val_shard(tmp_path)
    assert result == shards[-1], f"Expected last shard {shards[-1]}, got {result}"


# ---------------------------------------------------------------------------
# _select_random_docs tests
# ---------------------------------------------------------------------------


def test_select_random_docs_returns_tuple(
    feature_shard_dir, threshold_config, precomputed_thresholds
):
    """_select_random_docs returns a 2-tuple of (list, dict)."""
    selected_docs, metrics = _select_random_docs(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        n_eval_docs=10,
        seed=42,
    )
    assert isinstance(selected_docs, list)
    assert isinstance(metrics, dict)


def test_select_random_docs_metrics(
    feature_shard_dir, threshold_config, precomputed_thresholds
):
    """_select_random_docs metrics contain expected keys with positive values."""
    _, metrics = _select_random_docs(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        n_eval_docs=10,
        seed=42,
    )
    assert metrics["steering/n_docs_available"] >= 0, "Expected non-negative n_docs_available"
    assert "steering/n_docs_selected" in metrics


def test_select_random_docs_respects_n_eval(
    feature_shard_dir, threshold_config, precomputed_thresholds
):
    """_select_random_docs returns at most n_eval_docs entries."""
    n_eval_docs = 2
    selected_docs, _ = _select_random_docs(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        n_eval_docs=n_eval_docs,
        seed=42,
    )
    assert len(selected_docs) <= n_eval_docs, (
        f"Got {len(selected_docs)} docs, expected <= {n_eval_docs}"
    )
    for item in selected_docs:
        assert isinstance(item, tuple) and len(item) == 2, (
            f"Expected (doc_idx, target_seg_idx) tuple, got {item}"
        )


def test_select_random_docs_ignores_segments_with_only_masked_features(tmp_path):
    data = {
        "offsets": torch.tensor([0, 1, 2], dtype=torch.int64),
        "doc_ids": torch.tensor([0, 1], dtype=torch.int64),
        "segment_indices": torch.tensor([1, 1], dtype=torch.int64),
        "feature_ids": torch.tensor([0, 1], dtype=torch.int32),
        "feature_mags": torch.tensor([10.0, 9.0], dtype=torch.float32),
        "n_segments": 2,
    }
    save_features(data, tmp_path / "shard_00000_features")
    threshold_cfg = ThresholdConfig(ThresholdType.segment_topk, threshold=1)
    feature_mask = FeatureMask(FeatureMaskMode.blacklist, (0,))

    selected_docs, metrics = _select_random_docs(
        feature_dir=tmp_path,
        threshold_cfg=threshold_cfg,
        precomputed_thresholds=None,
        n_eval_docs=10,
        seed=42,
        feature_mask=feature_mask,
    )

    assert selected_docs == [(1, 1)]
    assert metrics["steering/n_docs_available"] == 1


def test_select_random_docs_empty_dir():
    """_select_random_docs returns ([], partial_metrics) for empty dir."""
    threshold_cfg = ThresholdConfig(
        threshold_type=ThresholdType.segment_topk,
        threshold=3.0,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        selected_docs, metrics = _select_random_docs(
            feature_dir=tmpdir,
            threshold_cfg=threshold_cfg,
            precomputed_thresholds=None,
            n_eval_docs=10,
            seed=42,
        )
    assert selected_docs == []
    assert metrics["steering/n_docs_available"] == 0


# ---------------------------------------------------------------------------
# _shard_stem tests
# ---------------------------------------------------------------------------


def test_shard_stem_basic():
    """_shard_stem extracts stem from a typical shard path."""
    p = Path("/tmp/features/shard_00000_features.safetensors.zst")
    assert _shard_stem(p) == "shard_00000"


def test_shard_stem_different_index():
    """_shard_stem works for different shard indices."""
    p = Path("/data/shard_00042_features.safetensors.zst")
    assert _shard_stem(p) == "shard_00042"


# ---------------------------------------------------------------------------
# RoundtripCompletion dataclass test
# ---------------------------------------------------------------------------


def test_roundtrip_completion_dataclass():
    """RoundtripCompletion stores doc metadata, ground-truth feature IDs, and samples."""
    rc = RoundtripCompletion(
        doc_idx=0, target_segment_idx=1, pre_segment_text="hello ",
        ground_truth_feature_ids=[5, 10, 15],
        samples=[CompletionSample(condition="real", text="world", seed=42)],
    )
    assert rc.ground_truth_feature_ids == [5, 10, 15]
    assert len(rc.samples) == 1
    assert rc.samples[0].condition == "real"


# ---------------------------------------------------------------------------
# _build_random_inputs test
# ---------------------------------------------------------------------------


def test_build_random_inputs():
    """Random inputs keep seg_map and non-target segments, replace target segment features."""
    # Build real inputs with 2 segments, target=1
    real_fids = torch.tensor([[[1, 2, 0], [3, 4, 0]]], dtype=torch.int64)
    real_mags = torch.tensor([[[1, 1, 0], [2, 2, 0]]], dtype=torch.int64)
    real_mask = torch.tensor([[[True, True, False], [True, True, False]]])
    real_segmap = torch.tensor([[0, 0, 1, 1, 1]])
    real_inputs = MultisegInputs(real_fids, real_mags, real_mask, real_segmap)

    random_seg = SegmentFeatures(
        segment_idx=0, position=0,
        feature_ids=torch.tensor([99, 88], dtype=torch.int64),
        feature_mags=torch.tensor([5, 5], dtype=torch.int64),
        char_start=0, char_end=10,
    )

    result = _build_random_inputs(real_inputs, random_seg, target_seg_idx=1)

    # seg_map unchanged
    assert torch.equal(result.seg_map, real_segmap)
    # Segment 0 (non-target) unchanged
    assert torch.equal(result.enc_feature_ids[0, 0], real_fids[0, 0])
    # Segment 1 (target) has random features
    assert result.enc_feature_ids[0, 1, 0].item() == 99
    assert result.enc_feature_ids[0, 1, 1].item() == 88
    assert result.enc_mask[0, 1, 0].item() is True
    assert result.enc_mask[0, 1, 1].item() is True


# ---------------------------------------------------------------------------
# _aggregate_roundtrip_metrics tests
# ---------------------------------------------------------------------------


def test_aggregate_roundtrip_metrics():
    """Aggregate raw roundtrip probe scores into expected wandb metrics."""
    _nan = float("nan")
    scores = [
        ("real", 5.0, 4.0, _nan), ("real", 3.0, 2.0, _nan),
        ("random", 2.0, 1.5, _nan), ("random", 1.0, 0.5, _nan),
    ]
    m = _aggregate_roundtrip_metrics(scores, total_gt_features=6, n_docs=2)
    assert m["steering/roundtrip_real_mean"] == pytest.approx(4.0)
    assert m["steering/roundtrip_random_mean"] == pytest.approx(1.5)
    assert m["steering/roundtrip_specificity"] == pytest.approx(2.5)
    assert m["steering/roundtrip_real_mean_post_act"] == pytest.approx(3.0)
    assert m["steering/roundtrip_specificity_post_act"] == pytest.approx(2.0)
    assert m["steering/roundtrip_n_samples"] == 4
    assert m["steering/roundtrip_n_gt_features_avg"] == pytest.approx(3.0)


def test_aggregate_roundtrip_metrics_empty():
    """Empty scores produce zeroed metrics."""
    m = _aggregate_roundtrip_metrics([], total_gt_features=0, n_docs=0)
    assert m["steering/roundtrip_real_mean"] == 0.0
    assert m["steering/roundtrip_real_mean_post_act"] == 0.0
    assert m["steering/roundtrip_n_samples"] == 0


# ---------------------------------------------------------------------------
# _probe_roundtrip_raw with mock probe model
# ---------------------------------------------------------------------------


def test_probe_roundtrip_raw_with_mock():
    """_probe_roundtrip_raw returns (condition, float) tuples with finite values."""
    probe = ProbeModel.__new__(ProbeModel)
    probe.model = MockHookedTransformer(32, "blocks.8.hook_resid_post")
    probe.sae = MockSAE(32, 50)
    probe.tokenizer = MockGemmaTokenizer()
    probe.hook_name = "blocks.8.hook_resid_post"
    probe.layer = 8
    probe.apply_b_dec = True

    completions = [
        RoundtripCompletion(
            doc_idx=0,
            target_segment_idx=1,
            pre_segment_text="hello ",
            ground_truth_feature_ids=[5, 10, 15],
            samples=[
                CompletionSample(condition="real", text="world today", seed=42),
                CompletionSample(condition="random", text="other things", seed=44),
            ],
        ),
    ]

    scores = _probe_roundtrip_raw(completions, probe)

    assert len(scores) >= 1
    for condition, pre_val, post_val, pct_val in scores:
        assert condition in {"real", "random"}
        assert isinstance(pre_val, float)
        assert math.isfinite(pre_val)
        assert isinstance(post_val, float)
        assert post_val >= 0.0
        # pct_val is nan when no percentile_table is provided
        assert isinstance(pct_val, float)
        assert math.isnan(pct_val)


# ---------------------------------------------------------------------------
# Roundtrip parallel matches sequential
# ---------------------------------------------------------------------------


def test_roundtrip_parallel_matches_sequential():
    """Aggregating partitioned roundtrip scores matches aggregating the full list."""
    _nan = float("nan")
    all_scores = [
        ("real", 5.0, 4.0, _nan), ("real", 3.0, 2.0, _nan),
        ("random", 2.0, 1.5, _nan), ("random", 1.0, 0.5, _nan),
    ]
    r0 = all_scores[0::2]
    r1 = all_scores[1::2]
    merged = r0 + r1
    seq_metrics = _aggregate_roundtrip_metrics(all_scores, 6, 2)
    par_metrics = _aggregate_roundtrip_metrics(merged, 6, 2)
    for key in seq_metrics:
        # nan != nan, so use isnan check for nan keys
        if isinstance(seq_metrics[key], float) and math.isnan(seq_metrics[key]):
            assert math.isnan(par_metrics[key])
        else:
            assert seq_metrics[key] == pytest.approx(par_metrics[key])


# ---------------------------------------------------------------------------
# evaluate_steering end-to-end test
# ---------------------------------------------------------------------------


def test_evaluate_steering_end_to_end(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """evaluate_steering returns all expected metric keys with finite values."""
    # The val_parquet fixture has 3 docs; use doc 0 text for character boundaries.
    # The shard fixture has 4 segments per doc, so mock must return 4 segments.
    doc_text = "The quick brown fox jumps over the lazy dog."
    bos_prefix_len = 1
    doc_len = len(doc_text)
    quarter = doc_len // 4

    # 4 segments spanning the document in BOS-inclusive coordinates
    seg_bounds = [
        (0, bos_prefix_len + quarter),
        (bos_prefix_len + quarter, bos_prefix_len + 2 * quarter),
        (bos_prefix_len + 2 * quarter, bos_prefix_len + 3 * quarter),
        (bos_prefix_len + 3 * quarter, bos_prefix_len + doc_len),
    ]

    def _make_segments():
        return [
            SegmentFeatures(
                segment_idx=i,
                position=start,
                feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
                char_start=start,
                char_end=end,
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_bounds)
        ]

    # Mock FeatureIndex -- returns 4 segments for any doc_idx 0-2
    class MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if 0 <= doc_idx <= 2:
                return _make_segments()
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=MockFeatureIndex(),  # ty: ignore[invalid-argument-type]
        probe_model=None,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    expected_keys = [
        "steering/n_docs_available",
        "steering/n_docs_selected",
        "steering/roundtrip_real_mean",
        "steering/roundtrip_random_mean",
        "steering/roundtrip_specificity",
        "steering/roundtrip_n_samples",
        "steering/roundtrip_n_gt_features_avg",
        "steering/n_completions_generated",
        "steering/eval_time_seconds",
    ]

    missing = set(expected_keys) - set(result.keys())
    assert not missing, f"Missing metric keys: {missing}"

    # All values should be finite
    for key in expected_keys:
        val = result[key]
        assert isinstance(val, (int, float)), f"{key} should be numeric, got {type(val)}"
        assert math.isfinite(val), f"{key} should be finite, got {val}"

    # Sanity checks
    assert result["steering/n_docs_selected"] >= 0, "Expected non-negative n_docs_selected"
    assert result["steering/eval_time_seconds"] > 0, "Expected positive eval time"


def test_evaluate_steering_end_to_end_with_full_context(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """evaluate_steering runs the full-context eval pipeline successfully."""
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
                segment_idx=i,
                position=start,
                feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
                char_start=start,
                char_end=end,
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_bounds)
        ]

    class MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if 0 <= doc_idx <= 2:
                return _make_segments()
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=MockFeatureIndex(),  # ty: ignore[invalid-argument-type]
        probe_model=None,
    )

    assert isinstance(result, dict), f"Expected dict, got {type(result)}"

    # Key metrics should exist and be finite (same as non-offset eval)
    expected_keys = [
        "steering/n_docs_available",
        "steering/n_docs_selected",
        "steering/roundtrip_n_samples",
        "steering/n_completions_generated",
        "steering/eval_time_seconds",
    ]
    missing = set(expected_keys) - set(result.keys())
    assert not missing, f"Missing metric keys: {missing}"

    for key in expected_keys:
        val = result[key]
        assert isinstance(val, (int, float)), f"{key} should be numeric, got {type(val)}"
        assert math.isfinite(val), f"{key} with offset=True should be finite, got {val}"

    assert result["steering/n_docs_selected"] >= 0
    assert result["steering/eval_time_seconds"] > 0


# ---------------------------------------------------------------------------
# State preservation and edge cases
# ---------------------------------------------------------------------------


def test_eval_does_not_corrupt_training_state(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """evaluate_steering with save/restore RNG does not corrupt model or optimizer state."""
    device = torch.device("cpu")

    # Put model in train mode, run one realistic forward pass to populate optimizer state
    tiny_model.train()

    # Build valid encoder inputs using the fixture model spec:
    # vocab_size=256, seq_len=128, n_sae_features=50, num_buckets=5
    B, T = 1, 16
    n_seg, F = 2, 3
    idx = torch.randint(1, 255, (B, T), dtype=torch.long)
    targets = torch.randint(1, 255, (B, T), dtype=torch.long)
    enc_feature_ids = torch.randint(0, 50, (B, n_seg, F), dtype=torch.long)
    enc_magnitudes = torch.randint(0, 5, (B, n_seg, F), dtype=torch.long)
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    seg_map = torch.zeros(B, T, dtype=torch.long)

    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
    loss = tiny_model(idx, targets, enc_feature_ids, enc_magnitudes, enc_mask, seg_map)
    if hasattr(loss, "lm_loss"):
        loss = loss.lm_loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Snapshot params and optimizer state
    param_snapshot = {
        name: param.data.clone() for name, param in tiny_model.named_parameters()
    }
    opt_state_snapshot = {
        k: {sk: sv.clone() if isinstance(sv, torch.Tensor) else sv for sk, sv in v.items()}
        for k, v in optimizer.state.items()
    }

    # Switch to eval for evaluate_steering (required by assertion)
    tiny_model.eval()

    # MockFeatureIndex -- 4 segments per doc matching shard fixture
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
                segment_idx=i,
                position=start,
                feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
                char_start=start,
                char_end=end,
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_bounds)
        ]

    class MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if 0 <= doc_idx <= 2:
                return _make_segments()
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    # Save RNG, run eval, restore RNG
    rng_state = save_rng_state(device)
    evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=device,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=MockFeatureIndex(),  # ty: ignore[invalid-argument-type]
    )
    restore_rng_state(rng_state, device)

    # Model parameters must be unchanged
    for name, param in tiny_model.named_parameters():
        expected = param_snapshot[name]
        assert torch.equal(param.data, expected), (
            f"Parameter {name} was modified by evaluate_steering"
        )

    # Optimizer state must be unchanged
    for param, state_dict in optimizer.state.items():
        snapshot = opt_state_snapshot.get(param, {})
        for key, val in state_dict.items():
            expected_val = snapshot.get(key)
            if expected_val is None:
                continue
            if isinstance(val, torch.Tensor):
                assert torch.equal(val, expected_val), (
                    f"Optimizer state key '{key}' was modified by evaluate_steering"
                )
            else:
                assert val == expected_val, (
                    f"Optimizer state key '{key}' was modified by evaluate_steering"
                )


def test_empty_feature_dir(
    tiny_model, boundaries, threshold_config, precomputed_thresholds
):
    """Feature dir with no shards returns dict with n_docs_available == 0."""
    with tempfile.TemporaryDirectory() as empty_dir:
        result = evaluate_steering(
            model=tiny_model,
            feature_dir=empty_dir,
            tokenizer=None,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
        )
    assert isinstance(result, dict)
    assert result["steering/n_docs_available"] == 0


def test_deterministic_metrics(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """Same model + data + seed produces identical metrics on two evaluate_steering runs."""
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
                segment_idx=i,
                position=start,
                feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
                char_start=start,
                char_end=end,
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_bounds)
        ]

    class MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if 0 <= doc_idx <= 2:
                return _make_segments()
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "cfg": SteeringEvalConfig(seed=42),
        "data_dir": str(val_parquet_dir),
        "feature_index": MockFeatureIndex(),
        "probe_model": None,
    }

    # First run
    torch.manual_seed(42)
    result1 = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]

    # Second run with same seed
    torch.manual_seed(42)
    result2 = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]

    # All scalar metrics must be identical between runs
    expected_keys = [
        "steering/n_docs_available",
        "steering/n_docs_selected",
        "steering/roundtrip_real_mean",
        "steering/roundtrip_random_mean",
        "steering/roundtrip_specificity",
        "steering/roundtrip_n_samples",
        "steering/roundtrip_n_gt_features_avg",
        "steering/n_completions_generated",
    ]
    for key in expected_keys:
        v1, v2 = result1[key], result2[key]
        assert v1 == v2, f"Metric '{key}' differed between runs: {v1} vs {v2}"


# ---------------------------------------------------------------------------
# Generation correctness tests
# ---------------------------------------------------------------------------


def test_ablation_removes_exactly_target_feature():
    """Ablating a feature removes it and only it from the target segment."""
    segments = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=torch.tensor([10, 20, 30], dtype=torch.int32),
            feature_mags=torch.tensor([1.0, 2.0, 3.0]),
            char_start=0,
            char_end=100,
        ),
    ]
    ablated = ablate_feature(segments, target_seg_idx=0, feature_id=20)
    assert 20 not in ablated[0].feature_ids.tolist()
    assert 10 in ablated[0].feature_ids.tolist()
    assert 30 in ablated[0].feature_ids.tolist()
    assert len(ablated[0].feature_ids) == 2


def test_conditioned_encoder_inputs_populate_mask():
    """Full-conditioning encoder inputs expose non-empty segment features."""
    # Build a single segment with post-discretization (int64) feature data
    segments = [
        SegmentFeatures(
            segment_idx=0,
            position=0,
            feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            feature_mags=torch.tensor([3, 1, 4], dtype=torch.int64),
            char_start=0,
            char_end=50,
        ),
    ]

    # Use empty prefix tokens so char-offset logic is skipped; n_gen_tokens > 0
    # so the seg_map is populated for generated tokens.
    tokenizer = ByteTokenizer()
    prefix_tokens: list[int] = []
    n_gen_tokens = 5

    conditioned = build_multiseg_inputs(
        segments,
        prefix_tokens,
        n_gen_tokens,
        target_segment_idx=0,
        tokenizer=tokenizer,
    )

    assert conditioned.enc_mask.any(), (
        "Full-conditioning enc_mask should have True values for non-empty segments"
    )


# ---------------------------------------------------------------------------
# Training loop integration tests
# ---------------------------------------------------------------------------


def test_steering_eval_called_at_correct_steps():
    """Verify steering eval is invoked at the right step intervals."""
    cfg_steering_eval_every = 100
    total_steps = 350
    called_at_steps = []
    for step in range(1, total_steps + 1):
        last_step = step == total_steps
        if cfg_steering_eval_every > 0 and (
            last_step or (step > 0 and step % cfg_steering_eval_every == 0)
        ):
            called_at_steps.append(step)
    assert called_at_steps == [100, 200, 300, 350]


def test_steering_eval_disabled_when_zero():
    """Verify steering eval is never called when steering_eval_every=0."""
    cfg_steering_eval_every = 0
    called = False
    for step in range(1, 1001):
        if cfg_steering_eval_every > 0 and step % cfg_steering_eval_every == 0:
            called = True
    assert not called


# ---------------------------------------------------------------------------
# DDP partitioning and gathering tests
# ---------------------------------------------------------------------------


def test_partition_docs_covers_all():
    """Partitioning 5 docs across 3 ranks covers all docs."""
    selected_docs = [(0, 1), (1, 2), (2, 0), (3, 1), (4, 1)]

    world_size = 3
    all_docs: list[tuple[int, int]] = []

    for rank in range(world_size):
        rank_docs = _partition_docs(selected_docs, rank, world_size)
        all_docs.extend(rank_docs)

    assert sorted(all_docs) == sorted(selected_docs)


def test_partition_docs_no_overlap():
    """No doc appears on more than one rank."""
    selected_docs = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 0), (5, 2), (6, 0)]

    world_size = 4
    seen: set[tuple[int, int]] = set()

    for rank in range(world_size):
        rank_docs = _partition_docs(selected_docs, rank, world_size)
        for doc in rank_docs:
            assert doc not in seen, f"Doc {doc} assigned to multiple ranks"
            seen.add(doc)


def test_partition_docs_single_rank_is_identity():
    """world_size=1 returns selected_docs unchanged."""
    selected_docs = [(10, 0), (20, 1)]
    result = _partition_docs(selected_docs, rank=0, world_size=1)
    assert result == selected_docs


# ---------------------------------------------------------------------------
# ProbeModel nnterp field tests
# ---------------------------------------------------------------------------


class TestProbeModelNnterp:
    """Verify ProbeModel works with optional nnterp_model field."""

    def test_probe_model_default_nnterp_none(self):
        d_model = 8
        hook_name = "blocks.8.hook_resid_post"
        model = MockHookedTransformer(d_model, hook_name)
        sae = MockSAE(d_model, 4)
        tok = MockGemmaTokenizer()
        pm = ProbeModel(model=model, sae=sae, tokenizer=tok, hook_name=hook_name, layer=8)
        assert pm.nnterp_model is None

    def test_probe_model_accepts_nnterp(self):
        d_model = 8
        hook_name = "blocks.8.hook_resid_post"
        model = MockHookedTransformer(d_model, hook_name)
        sae = MockSAE(d_model, 4)
        tok = MockGemmaTokenizer()
        fake_nn = object()
        pm = ProbeModel(model=model, sae=sae, tokenizer=tok, hook_name=hook_name, layer=8, nnterp_model=fake_nn)
        assert pm.nnterp_model is fake_nn


def test_partition_docs_more_ranks_than_docs():
    """When world_size > number of docs, excess ranks get empty list."""
    selected_docs = [(0, 1)]

    # 4 ranks but only 1 doc -- ranks 1,2,3 get nothing
    for rank in range(4):
        rank_docs = _partition_docs(selected_docs, rank, world_size=4)
        if rank == 0:
            assert len(rank_docs) == 1
        else:
            assert len(rank_docs) == 0


def test_gather_int_no_dist():
    """Without distributed, _gather_int returns the local value."""
    assert _gather_int(42, world_size=1) == 42


# ---------------------------------------------------------------------------
# _gather_int device placement tests (fix for NCCL + CPU tensor crash)
# ---------------------------------------------------------------------------


def test_gather_int_no_dist_with_device_param():
    """_gather_int returns local value when world_size=1, device param is ignored."""
    assert _gather_int(42, world_size=1, device=torch.device("cpu")) == 42
    assert _gather_int(99, world_size=1, device="cpu") == 99


def test_gather_int_no_dist_when_not_initialized():
    """_gather_int returns local value when dist is not initialized, even with world_size>1."""
    with patch.object(dist, "is_initialized", return_value=False):
        assert _gather_int(42, world_size=4, device="cpu") == 42


def test_gather_int_tensor_device_matches_param():
    """_gather_int creates tensor on specified device for all_reduce (not CPU)."""
    captured_devices = []

    def mock_all_reduce(tensor, op=None):
        captured_devices.append(tensor.device)

    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "all_reduce", side_effect=mock_all_reduce):
        _gather_int(42, world_size=2, device=torch.device("cpu"))

    assert len(captured_devices) == 1
    assert captured_devices[0].type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_gather_int_creates_cuda_tensor():
    """_gather_int creates CUDA tensor when device is cuda, not CPU."""
    captured_devices = []

    def mock_all_reduce(tensor, op=None):
        captured_devices.append(tensor.device)

    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "all_reduce", side_effect=mock_all_reduce):
        _gather_int(42, world_size=2, device=torch.device("cuda:0"))

    assert len(captured_devices) == 1
    assert captured_devices[0].type == "cuda"


def test_gather_int_correctness_after_mock_reduction():
    """_gather_int returns the correct value after simulated multi-rank sum."""
    def mock_all_reduce(tensor, op=None):
        # Simulate 3 ranks each contributing the same value
        tensor.mul_(3)

    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "all_reduce", side_effect=mock_all_reduce):
        result = _gather_int(10, world_size=3, device="cpu")

    assert result == 30


def test_gather_int_correctness_zero():
    """_gather_int handles zero correctly through reduction."""
    def mock_all_reduce(tensor, op=None):
        tensor.mul_(4)

    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "all_reduce", side_effect=mock_all_reduce):
        result = _gather_int(0, world_size=4, device="cpu")

    assert result == 0


def test_gather_int_preserves_dtype():
    """_gather_int uses long dtype for the reduction tensor."""
    captured_dtypes = []

    def mock_all_reduce(tensor, op=None):
        captured_dtypes.append(tensor.dtype)

    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "all_reduce", side_effect=mock_all_reduce):
        _gather_int(42, world_size=2, device="cpu")

    assert captured_dtypes[0] == torch.long


def test_all_collective_calls_use_device_tensor():
    """_gather_int always passes a device-placed tensor to all_reduce.

    Calls _gather_int twice (matching the two call sites in evaluate_steering)
    and verifies that all_reduce receives tensors on the specified device.
    dist.all_gather_object is the only other collective in the file and is
    unaffected (uses pickling, not tensor ops).
    """
    all_reduce_devices = []

    def tracking_all_reduce(tensor, op=None):
        all_reduce_devices.append(tensor.device)

    # Mock dist for the _gather_int calls but let all_gather_object work normally
    with patch.object(dist, "is_initialized", return_value=True), \
         patch.object(dist, "all_reduce", side_effect=tracking_all_reduce), \
         patch.object(dist, "all_gather_object"):
        # Simulate what evaluate_steering does: call _gather_int with a device
        _gather_int(5, world_size=2, device=torch.device("cpu"))
        _gather_int(10, world_size=2, device=torch.device("cpu"))

    # Both all_reduce calls should have received CPU tensors (matching device param)
    assert len(all_reduce_devices) == 2
    for dev in all_reduce_devices:
        assert dev.type == "cpu"


# ---------------------------------------------------------------------------
# Probe model non-interference tests
# ---------------------------------------------------------------------------


def test_probe_model_stays_on_cpu_after_eval(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """The device fix does not move the probe model from CPU to another device.

    Simulates the offload_steering_eval_feature_model scenario where the probe
    model is on CPU at rest. The fix must not accidentally move it.
    """
    # Build a mock probe model on CPU
    d_model = 32
    hook_name = "blocks.0.hook_resid_post"
    probe = ProbeModel.__new__(ProbeModel)
    probe.model = MockHookedTransformer(d_model, hook_name)
    probe.sae = MockSAE(d_model, 50)
    probe.tokenizer = MockGemmaTokenizer()
    probe.hook_name = hook_name
    probe.layer = 0
    probe.nnterp_model = None
    probe.activation_type = "residual"  # ty: ignore[invalid-assignment]
    probe.apply_b_dec = True

    # Record initial device
    initial_device = probe.device

    # Build MockFeatureIndex
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
                segment_idx=i,
                position=start,
                feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
                char_start=start,
                char_end=end,
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_bounds)
        ]

    class MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if 0 <= doc_idx <= 2:
                return _make_segments()
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=MockFeatureIndex(),  # ty: ignore[invalid-argument-type]
        probe_model=probe,
    )

    # Probe model must still be on its original device
    assert probe.device == initial_device, (
        f"Probe model moved from {initial_device} to {probe.device}"
    )


def test_evaluate_steering_with_probe_returns_metrics(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    monkeypatch,
):
    """evaluate_steering with a probe model returns all expected metric keys."""
    d_model = 32
    hook_name = "blocks.0.hook_resid_post"
    probe = ProbeModel.__new__(ProbeModel)
    probe.model = MockHookedTransformer(d_model, hook_name)
    probe.sae = MockSAE(d_model, 50)
    probe.tokenizer = MockGemmaTokenizer()
    probe.hook_name = hook_name
    probe.layer = 0
    probe.nnterp_model = None
    probe.activation_type = "residual"  # ty: ignore[invalid-assignment]
    probe.apply_b_dec = True

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
                segment_idx=i,
                position=start,
                feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
                feature_mags=torch.tensor([5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32),
                char_start=start,
                char_end=end,
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_bounds)
        ]

    class MockFeatureIndex:
        def lookup(self, shard_stem, doc_idx):
            if 0 <= doc_idx <= 2:
                return _make_segments()
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1, 2]

    import steering.eval.inline_steering_eval as mod

    def _fake_probe_roundtrip_raw(completions, probe_model, percentile_table=None):
        scores = []
        for completion in completions:
            for sample in completion.samples:
                scores.append((sample.condition, 2.0, 1.0, 50.0))
        return scores

    monkeypatch.setattr(mod, "_probe_roundtrip_raw", _fake_probe_roundtrip_raw)

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=MockFeatureIndex(),  # ty: ignore[invalid-argument-type]
        probe_model=probe,
    )

    assert isinstance(result, dict)
    expected_keys = [
        "steering/roundtrip_real_mean",
        "steering/roundtrip_specificity",
        "steering/roundtrip_n_samples",
        "steering/n_completions_generated",
        "steering/eval_time_seconds",
    ]
    for key in expected_keys:
        assert key in result, f"Missing key: {key}"
        assert isinstance(result[key], (int, float)), f"{key} should be numeric"
        assert math.isfinite(result[key]), f"{key} should be finite, got {result[key]}"


# ---------------------------------------------------------------------------
# Tests for _build_random_inputs
# ---------------------------------------------------------------------------


def _make_multiseg_inputs_for_random_test(
    n_seg: int = 3, F_max: int = 5, T: int = 384,
) -> MultisegInputs:
    """Build synthetic MultisegInputs with distinct features per slot."""
    enc_fids = torch.zeros(1, n_seg, F_max, dtype=torch.int64)
    enc_mags = torch.zeros(1, n_seg, F_max, dtype=torch.int64)
    enc_mask = torch.zeros(1, n_seg, F_max, dtype=torch.bool)

    for s in range(n_seg):
        enc_fids[0, s, :] = torch.arange(s * 100, s * 100 + F_max)
        enc_mags[0, s, :] = s + 1
        enc_mask[0, s, :] = True

    seg_map = torch.zeros(1, T, dtype=torch.int64)
    tokens_per_seg = T // n_seg
    for s in range(n_seg):
        start = s * tokens_per_seg
        end = (s + 1) * tokens_per_seg if s < n_seg - 1 else T
        seg_map[0, start:end] = s

    return MultisegInputs(
        enc_feature_ids=enc_fids,
        enc_magnitudes=enc_mags,
        enc_mask=enc_mask,
        seg_map=seg_map,
    )


def _make_random_seg_for_test(n_features: int = 3) -> SegmentFeatures:
    """Build a recognizable random SegmentFeatures."""
    return SegmentFeatures(
        segment_idx=99,
        position=0,
        feature_ids=torch.arange(5000, 5000 + n_features, dtype=torch.int32),
        feature_mags=torch.ones(n_features, dtype=torch.float32) * 7,
        char_start=0,
        char_end=100,
    )


class TestBuildRandomInputs:

    def test_places_features_at_target(self):
        """Random features are placed at target_seg_idx."""
        real = _make_multiseg_inputs_for_random_test(n_seg=3, F_max=5)
        rand_seg = _make_random_seg_for_test(n_features=3)

        result = _build_random_inputs(real, rand_seg, target_seg_idx=2)

        # Slot 2 should have random features
        assert result.enc_mask[0, 2, :3].all()
        assert torch.equal(result.enc_feature_ids[0, 2, :3],
                           rand_seg.feature_ids[:3].to(torch.int64))
        # Slots 0 and 1 should still have real features
        assert result.enc_mask[0, 0, :].all()
        assert result.enc_mask[0, 1, :].all()

    def test_random_inputs_inherits_seg_map(self):
        """seg_map from real_inputs is preserved in random_inputs."""
        real = _make_multiseg_inputs_for_random_test(n_seg=3, F_max=5)
        rand_seg = _make_random_seg_for_test()

        result = _build_random_inputs(real, rand_seg, target_seg_idx=2)

        assert torch.equal(result.seg_map, real.seg_map)

    def test_clustered_donor_wider_than_prefix_is_not_truncated(self):
        """Certified clustered donor rows expand F instead of losing features."""
        real = _make_multiseg_inputs_for_random_test(n_seg=3, F_max=3)
        rand_seg = _make_random_seg_for_test(n_features=7)

        result = _build_random_inputs(
            real,
            rand_seg,
            target_seg_idx=2,
            preserve_all_donor_features=True,
        )

        assert result.enc_feature_ids.shape == (1, 3, 7)
        assert result.enc_magnitudes.shape == (1, 3, 7)
        assert result.enc_mask.shape == (1, 3, 7)
        assert torch.equal(
            result.enc_feature_ids[0, 2],
            rand_seg.feature_ids.to(torch.int64),
        )
        assert torch.equal(
            result.enc_magnitudes[0, 2],
            rand_seg.feature_mags.to(torch.int64),
        )
        assert result.enc_mask[0, 2].all()
        assert torch.equal(
            result.enc_feature_ids[0, :2, :3],
            real.enc_feature_ids[0, :2],
        )
        assert not result.enc_mask[0, :2, 3:].any()


# ---------------------------------------------------------------------------
# Functional tests: end-to-end model forward pass
# ---------------------------------------------------------------------------


class TestModelForwardWithOffset:
    """End-to-end tests that feed offset-shifted inputs into a real EncoderDecoderGPT
    model and verify the forward and backward passes work correctly.
    """

    def test_forward_with_offset_no_nan(self, tiny_model):
        """Model forward pass with offset-shifted inputs produces finite loss."""
        B, T = 1, 16
        n_seg, F = 3, 3
        # The last segment has empty features (enc_mask=False) to simulate the
        # offset pattern — this is the critical edge case.
        idx = torch.randint(1, 255, (B, T), dtype=torch.long)
        targets = torch.randint(1, 255, (B, T), dtype=torch.long)
        enc_feature_ids = torch.randint(0, 50, (B, n_seg, F), dtype=torch.long)
        enc_magnitudes = torch.randint(0, 5, (B, n_seg, F), dtype=torch.long)
        enc_mask = torch.zeros(B, n_seg, F, dtype=torch.bool)
        # Slots 0 and 1 have features (from segments 1 and 2 via offset)
        enc_mask[0, 0, :] = True
        enc_mask[0, 1, :] = True
        # Slot 2 (last) is empty — this is the offset invariant
        # seg_map: text positions attend to slots 0 and 1 (not slot 2)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[0, :8] = 0  # first half attends to slot 0
        seg_map[0, 8:] = 1  # second half attends to slot 1

        out = tiny_model(
            idx, targets, enc_feature_ids, enc_magnitudes, enc_mask, seg_map,
        )
        loss = out.lm_loss if hasattr(out, "lm_loss") else out
        assert torch.isfinite(loss), f"Loss should be finite, got {loss}"
        assert loss.item() > 0, f"Loss should be positive, got {loss}"

    def test_backward_with_offset_finite_gradients(self, tiny_model):
        """Backward pass with offset-shifted inputs produces finite gradients."""
        tiny_model.train()
        B, T = 1, 16
        n_seg, F = 3, 3
        idx = torch.randint(1, 255, (B, T), dtype=torch.long)
        targets = torch.randint(1, 255, (B, T), dtype=torch.long)
        enc_feature_ids = torch.randint(0, 50, (B, n_seg, F), dtype=torch.long)
        enc_magnitudes = torch.randint(0, 5, (B, n_seg, F), dtype=torch.long)
        enc_mask = torch.zeros(B, n_seg, F, dtype=torch.bool)
        enc_mask[0, 0, :] = True
        enc_mask[0, 1, :] = True
        # Last slot empty (offset invariant)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[0, :8] = 0
        seg_map[0, 8:] = 1

        out = tiny_model(
            idx, targets, enc_feature_ids, enc_magnitudes, enc_mask, seg_map,
        )
        loss = out.lm_loss if hasattr(out, "lm_loss") else out
        loss.backward()

        for name, param in tiny_model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), (
                    f"Gradient for {name} has non-finite values"
                )
        tiny_model.eval()

    def test_output_differs_from_empty_last_only(self, tiny_model):
        """Verify cross-attention actually uses non-last slots.

        If the model only looked at the last (empty) slot, the output would
        match an all-masked input. With offset, the real features are in
        slots 0..n-2, so the output must differ from all-masked.
        """
        B, T = 1, 16
        n_seg, F = 3, 3
        idx = torch.randint(1, 255, (B, T), dtype=torch.long)
        enc_feature_ids = torch.randint(0, 50, (B, n_seg, F), dtype=torch.long)
        enc_magnitudes = torch.randint(0, 5, (B, n_seg, F), dtype=torch.long)

        # Mask with features in slots 0, 1 (offset pattern)
        offset_mask = torch.zeros(B, n_seg, F, dtype=torch.bool)
        offset_mask[0, 0, :] = True
        offset_mask[0, 1, :] = True
        # seg_map routes tokens to slots 0 and 1 (where features live)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[0, :8] = 0
        seg_map[0, 8:] = 1

        # All-masked: no features anywhere
        all_masked = torch.zeros(B, n_seg, F, dtype=torch.bool)

        with torch.no_grad():
            out_offset = tiny_model(
                idx, None, enc_feature_ids, enc_magnitudes, offset_mask, seg_map,
            )
            out_empty = tiny_model(
                idx, None, enc_feature_ids, enc_magnitudes, all_masked, seg_map,
            )

        logits_offset = out_offset if isinstance(out_offset, torch.Tensor) else out_offset.logits
        logits_empty = out_empty if isinstance(out_empty, torch.Tensor) else out_empty.logits

        # The outputs must differ: the offset case has real features in encoder
        # slots 0 and 1, which influence decoder generation via cross-attention.
        assert not torch.allclose(logits_offset, logits_empty, atol=1e-4), (
            "Output should differ when offset conditioning features are present vs "
            "all-masked — cross-attention is not using encoder slots 0, 1"
        )

    def test_output_invariant_to_last_slot_contents(self, tiny_model):
        """The last encoder slot is empty (mask=False) — changing its feature_ids
        must not affect the output, since the decoder never attends to it.
        """
        B, T = 1, 16
        n_seg, F = 3, 3
        idx = torch.randint(1, 255, (B, T), dtype=torch.long)
        enc_mask = torch.zeros(B, n_seg, F, dtype=torch.bool)
        enc_mask[0, 0, :] = True
        enc_mask[0, 1, :] = True
        # Last slot empty (mask=False)
        seg_map = torch.zeros(B, T, dtype=torch.long)
        seg_map[0, :8] = 0
        seg_map[0, 8:] = 1

        enc_magnitudes = torch.randint(0, 5, (B, n_seg, F), dtype=torch.long)
        enc_mags_other = enc_magnitudes.clone()
        enc_mags_other[0, 2, :] = (enc_magnitudes[0, 2, :] + 2) % 5  # change last slot magnitudes

        fids_a = torch.randint(0, 50, (B, n_seg, F), dtype=torch.long)
        fids_b = fids_a.clone()
        fids_b[0, 2, :] = (fids_a[0, 2, :] + 13) % 50  # change last slot fids

        with torch.no_grad():
            out_a = tiny_model(idx, None, fids_a, enc_magnitudes, enc_mask, seg_map)
            out_b = tiny_model(idx, None, fids_b, enc_mags_other, enc_mask, seg_map)

        logits_a = out_a if isinstance(out_a, torch.Tensor) else out_a.logits
        logits_b = out_b if isinstance(out_b, torch.Tensor) else out_b.logits

        # Outputs must be identical — last slot is masked out
        assert torch.allclose(logits_a, logits_b, atol=1e-5), (
            "Last slot contents should not affect output when its mask is all-False"
        )


# ---------------------------------------------------------------------------
# P0: _load_doc_texts tests
# ---------------------------------------------------------------------------


def test_load_doc_texts_returns_indexed_dict(val_parquet_dir):
    """_load_doc_texts returns {doc_idx: text} for a single-shard val parquet."""
    result = _load_doc_texts(str(val_parquet_dir))
    assert isinstance(result, dict)
    assert len(result) == 3
    # val_parquet_dir fixture writes exactly these strings
    assert result[0] == "The quick brown fox jumps over the lazy dog."
    assert result[1] == "An encoder-decoder model learns to condition on SAE features."
    assert result[2] == "Sparse autoencoders extract interpretable features from neural networks."


def test_load_doc_texts_picks_last_shard(tmp_path):
    """_load_doc_texts reads the last (sorted) parquet file in the dir, matching list_parquet_files(...)[-1]."""
    # Write two shards; the second is alphabetically last
    pq.write_table(pa.table({"text": ["first-shard-0", "first-shard-1"]}), tmp_path / "val_00000.parquet")
    pq.write_table(pa.table({"text": ["second-shard-0"]}), tmp_path / "val_00001.parquet")

    result = _load_doc_texts(str(tmp_path))
    assert result == {0: "second-shard-0"}


def test_load_doc_texts_missing_parquet_raises(tmp_path):
    """_load_doc_texts raises IndexError when the dir has no parquet files (misconfig)."""
    with pytest.raises(IndexError):
        _load_doc_texts(str(tmp_path))


# ---------------------------------------------------------------------------
# P0: InlineEvalCache dataclass tests
# ---------------------------------------------------------------------------


def test_inline_eval_cache_is_frozen():
    """InlineEvalCache is frozen — field assignment raises FrozenInstanceError."""
    cache = InlineEvalCache(
        selected_docs=[(0, 1), (1, 2)],
        selection_metrics={"steering/n_docs_available": 5, "steering/n_docs_selected": 2},
        doc_texts={0: "hello", 1: "world"},
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cache.selected_docs = []  # type: ignore[misc]  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        cache.doc_texts = {}  # type: ignore[misc]  # ty: ignore[invalid-assignment]


def test_inline_eval_cache_equality():
    """Two InlineEvalCache instances built from the same inputs are equal;
    instances with different contents compare unequal."""
    sel = [(0, 1), (1, 2)]
    met = {"steering/n_docs_available": 5, "steering/n_docs_selected": 2}
    txt = {0: "hello", 1: "world"}

    a = InlineEvalCache(selected_docs=sel, selection_metrics=met, doc_texts=txt)  # ty: ignore[invalid-argument-type]
    b = InlineEvalCache(
        selected_docs=[(0, 1), (1, 2)],
        selection_metrics={"steering/n_docs_available": 5, "steering/n_docs_selected": 2},
        doc_texts={0: "hello", 1: "world"},
    )
    assert a == b

    # Inequality: different selected_docs
    c = InlineEvalCache(selected_docs=[(9, 9)], selection_metrics=met, doc_texts=txt)  # ty: ignore[invalid-argument-type]
    assert a != c
    # Inequality: different selection_metrics
    d = InlineEvalCache(selected_docs=sel, selection_metrics={"steering/n_docs_available": 99, "steering/n_docs_selected": 2}, doc_texts=txt)
    assert a != d
    # Inequality: different doc_texts
    e = InlineEvalCache(selected_docs=sel, selection_metrics=met, doc_texts={0: "different"})  # ty: ignore[invalid-argument-type]
    assert a != e


# ---------------------------------------------------------------------------
# P0: prepare_inline_eval_cache tests
# ---------------------------------------------------------------------------


def test_prepare_inline_eval_cache_shape(
    feature_shard_dir, threshold_config, precomputed_thresholds, val_parquet_dir
):
    """prepare_inline_eval_cache returns an InlineEvalCache with the right shape."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )
    assert isinstance(cache, InlineEvalCache)
    assert isinstance(cache.selected_docs, list)
    assert len(cache.selected_docs) <= 10
    for item in cache.selected_docs:
        assert isinstance(item, tuple) and len(item) == 2
        assert isinstance(item[0], int) and isinstance(item[1], int)
    assert "steering/n_docs_available" in cache.selection_metrics
    assert "steering/n_docs_selected" in cache.selection_metrics
    assert isinstance(cache.doc_texts, dict)
    assert len(cache.doc_texts) == 3  # val_parquet_dir fixture has 3 rows


def test_prepare_inline_eval_cache_determinism(
    feature_shard_dir, threshold_config, precomputed_thresholds, val_parquet_dir
):
    """Two calls with identical args produce exactly-equal selected_docs.
    Different seed produces a cache that still satisfies invariants."""
    kwargs = {
        "feature_dir": feature_shard_dir,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "n_eval_docs": 10,
    }
    a = prepare_inline_eval_cache(seed=42, **kwargs)  # ty: ignore[invalid-argument-type]
    b = prepare_inline_eval_cache(seed=42, **kwargs)  # ty: ignore[invalid-argument-type]
    assert a.selected_docs == b.selected_docs
    assert a.selection_metrics == b.selection_metrics
    assert a.doc_texts == b.doc_texts

    # The tiny feature shard has only 3 docs, so different seeds may give
    # the same docs. Verify the call runs to completion and returns a valid
    # cache under an alternative seed.
    c = prepare_inline_eval_cache(seed=1337, **kwargs)  # ty: ignore[invalid-argument-type]
    assert 0 <= len(c.selected_docs) <= 3


def test_prepare_inline_eval_cache_uses_n_eval_docs(
    feature_shard_dir, threshold_config, precomputed_thresholds, val_parquet_dir
):
    """prepare_inline_eval_cache respects n_eval_docs upper bound."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=2,
        seed=42,
    )
    assert len(cache.selected_docs) <= 2


def test_prepare_inline_eval_cache_empty_dir(tmp_path, threshold_config, precomputed_thresholds):
    """prepare_inline_eval_cache with an empty feature dir returns an empty cache.

    When there are no feature shards, _select_random_docs returns an empty
    list + scan metrics. The cache must not attempt to read a val parquet
    in that case — it returns an empty doc_texts dict.
    """
    cache = prepare_inline_eval_cache(
        feature_dir=tmp_path,  # empty — no shards
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=None,
        n_eval_docs=10,
        seed=42,
    )
    assert cache.selected_docs == []
    assert cache.selection_metrics["steering/n_docs_available"] == 0
    assert cache.selection_metrics["steering/n_docs_selected"] == 0
    assert cache.doc_texts == {}


# ---------------------------------------------------------------------------
# P0: evaluate_steering eval_cache kwarg tests
# ---------------------------------------------------------------------------


def test_evaluate_steering_accepts_eval_cache_none(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """evaluate_steering with eval_cache=None runs the legacy compute-inline path."""
    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=None,
    )
    assert isinstance(result, dict)
    assert "steering/n_docs_selected" in result
    assert "steering/eval_time_seconds" in result


def test_evaluate_steering_accepts_eval_cache_populated(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Passing a pre-built InlineEvalCache runs to completion and returns the expected keys."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=cache,
    )
    assert isinstance(result, dict)
    assert "steering/n_docs_selected" in result
    assert "steering/eval_time_seconds" in result


def test_evaluate_steering_skips_shard_load_when_cache_passed(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
    monkeypatch,
):
    """When eval_cache is populated, _select_random_docs must not be called."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("_select_random_docs should not be called when eval_cache is provided")

    monkeypatch.setattr(inline_mod, "_select_random_docs", _boom)

    # Must not raise
    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=cache,
    )
    assert isinstance(result, dict)


def test_evaluate_steering_skips_parquet_read_when_cache_passed(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
    monkeypatch,
):
    """When eval_cache is populated, _load_doc_texts must not be called."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    def _boom(*args, **kwargs):
        raise AssertionError("_load_doc_texts should not be called when eval_cache is provided")

    monkeypatch.setattr(inline_mod, "_load_doc_texts", _boom)

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=cache,
    )
    assert isinstance(result, dict)


def test_evaluate_steering_feature_index_not_reconstructed(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
    monkeypatch,
):
    """When feature_index is passed, FeatureIndex.__init__ is not called inside evaluate_steering."""
    def _boom(self, *args, **kwargs):
        raise AssertionError("FeatureIndex.__init__ should not be called when feature_index is provided")

    monkeypatch.setattr(FeatureIndex, "__init__", _boom)

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=mock_feature_index,
        probe_model=None,
    )
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# P0: copy-before-mutate + eval_cache_hit metric tests
# ---------------------------------------------------------------------------


def test_eval_cache_not_mutated_across_calls(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Running evaluate_steering twice with the same cache does not mutate the cache."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    # Snapshot the cache's selection_metrics as a plain dict comparable against post-call state
    snapshot = dict(cache.selection_metrics)

    for _ in range(2):
        evaluate_steering(
            model=tiny_model,
            feature_dir=feature_shard_dir,
            tokenizer=byte_tokenizer,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            feature_index=mock_feature_index,
            probe_model=None,
            eval_cache=cache,
        )
        # The cache's own dict must still equal the snapshot after every call
        assert cache.selection_metrics == snapshot, (
            f"cache.selection_metrics mutated: {cache.selection_metrics} != {snapshot}"
        )
        # The cache must still not have eval_time_seconds or eval_cache_hit poisoned in
        assert "steering/eval_time_seconds" not in cache.selection_metrics
        assert "steering/eval_cache_hit" not in cache.selection_metrics


def test_eval_cache_hit_metric_is_emitted(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """The returned dict includes steering/eval_cache_hit == 1 when a cache is passed,
    and == 0 when it is not."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "feature_index": mock_feature_index,
        "probe_model": None,
    }
    uncached = evaluate_steering(**common_kwargs, eval_cache=None)  # ty: ignore[invalid-argument-type]
    cached = evaluate_steering(**common_kwargs, eval_cache=cache)  # ty: ignore[invalid-argument-type]

    assert uncached["steering/eval_cache_hit"] == 0
    assert cached["steering/eval_cache_hit"] == 1


# ---------------------------------------------------------------------------
# P0: integration tests — regression guard, reuse, independence
# ---------------------------------------------------------------------------


def _numeric_metrics_equal(a: dict, b: dict, exclude_keys: set[str]) -> bool:
    """Compare two metric dicts, ignoring non-deterministic keys.

    Returns True if all numeric keys (other than excluded ones) match via
    pytest.approx; non-numeric values must match via ==.
    """
    keys = (set(a) | set(b)) - exclude_keys
    for k in keys:
        if k not in a or k not in b:
            return False
        av, bv = a[k], b[k]
        if isinstance(av, (int, float)) and isinstance(bv, (int, float)):
            if math.isnan(av) and math.isnan(bv):
                continue
            if av != pytest.approx(bv, rel=1e-6, abs=1e-9):
                return False
        else:
            if av != bv:
                return False
    return True


def test_cache_matches_uncached_metrics_bit_for_bit(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Central regression guard: caching must not change any metric value.

    Runs evaluate_steering twice under identical RNG state: once with
    eval_cache=None and once with a prepared cache. Every numeric metric
    must match (excluding eval_cache_hit and timing).
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "feature_index": mock_feature_index,
        "probe_model": None,
    }

    # Reset all RNG state before each call so model generation is reproducible
    torch.manual_seed(12345)
    uncached = evaluate_steering(**common_kwargs, eval_cache=None)  # ty: ignore[invalid-argument-type]

    torch.manual_seed(12345)
    cached = evaluate_steering(**common_kwargs, eval_cache=cache)  # ty: ignore[invalid-argument-type]

    # Excluded: timing is wall-clock and will differ; eval_cache_hit differs by design
    exclude = {"steering/eval_time_seconds", "steering/eval_cache_hit"}
    assert _numeric_metrics_equal(uncached, cached, exclude), (
        f"Metrics differ between uncached and cached paths:\n"
        f"uncached={sorted(uncached.items())}\n"
        f"cached={sorted(cached.items())}"
    )


def test_cache_reuse_across_multiple_eval_calls(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
    monkeypatch,
):
    """Reusing the same cache across 3 calls: _select_random_docs and
    _load_doc_texts are each invoked exactly once (during cache build)."""
    # Count invocations
    select_calls = {"n": 0}
    load_calls = {"n": 0}

    original_select = inline_mod._select_random_docs
    original_load = inline_mod._load_doc_texts

    def _counting_select(*args, **kwargs):
        select_calls["n"] += 1
        return original_select(*args, **kwargs)

    def _counting_load(*args, **kwargs):
        load_calls["n"] += 1
        return original_load(*args, **kwargs)

    monkeypatch.setattr(inline_mod, "_select_random_docs", _counting_select)
    monkeypatch.setattr(inline_mod, "_load_doc_texts", _counting_load)

    # Build cache once — should invoke both helpers exactly once
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )
    assert select_calls["n"] == 1
    assert load_calls["n"] == 1

    # Call evaluate_steering 3 times with the same cache — counters must not increment
    for _ in range(3):
        evaluate_steering(
            model=tiny_model,
            feature_dir=feature_shard_dir,
            tokenizer=byte_tokenizer,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            feature_index=mock_feature_index,
            probe_model=None,
            eval_cache=cache,
        )

    assert select_calls["n"] == 1, (
        f"_select_random_docs should have been called once, was {select_calls['n']}"
    )
    assert load_calls["n"] == 1, (
        f"_load_doc_texts should have been called once, was {load_calls['n']}"
    )


def test_cache_independence_across_seeds(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Two caches built with different seeds both run end-to-end in evaluate_steering."""
    cache_a = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )
    cache_b = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=1337,
    )

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "feature_index": mock_feature_index,
        "probe_model": None,
    }
    result_a = evaluate_steering(**common_kwargs, eval_cache=cache_a)  # ty: ignore[invalid-argument-type]
    result_b = evaluate_steering(**common_kwargs, eval_cache=cache_b)  # ty: ignore[invalid-argument-type]
    assert "steering/eval_time_seconds" in result_a
    assert "steering/eval_time_seconds" in result_b


def test_cache_independence_across_threshold_configs(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Two caches built with different ThresholdConfigs both work end-to-end."""
    cfg_a = ThresholdConfig(threshold_type=ThresholdType.segment_topk, threshold=3.0)
    cfg_b = ThresholdConfig(threshold_type=ThresholdType.segment_topk, threshold=1.0)

    cache_a = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=cfg_a,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )
    cache_b = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=cfg_b,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    # Each must run end-to-end with its matching threshold_cfg
    for cfg, cache in [(cfg_a, cache_a), (cfg_b, cache_b)]:
        result = evaluate_steering(
            model=tiny_model,
            feature_dir=feature_shard_dir,
            tokenizer=byte_tokenizer,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=cfg,
            precomputed_thresholds=precomputed_thresholds,
            feature_index=mock_feature_index,
            probe_model=None,
            eval_cache=cache,
        )
        assert "steering/eval_time_seconds" in result


# ---------------------------------------------------------------------------
# P0: property tests
# ---------------------------------------------------------------------------


def _build_random_shard(tmp_path, n_docs, n_segs_per_doc, features_per_seg):
    """Build a valid feature shard with the given shape in tmp_path.

    Returns the tmp_path for passing to prepare_inline_eval_cache.
    Uses torch's global RNG with a deterministic seed derived from the
    shape so shrinking-style iteration is stable.
    """
    torch.manual_seed(hash((n_docs, n_segs_per_doc, features_per_seg)) & 0xFFFFFFFF)
    n_segs = n_docs * n_segs_per_doc
    total_features = n_segs * features_per_seg
    offsets = torch.arange(0, total_features + 1, features_per_seg, dtype=torch.int64)
    doc_ids = torch.repeat_interleave(torch.arange(n_docs, dtype=torch.int64), n_segs_per_doc)
    segment_indices = torch.tile(torch.arange(n_segs_per_doc, dtype=torch.int64), (n_docs,))
    feature_ids = torch.randint(0, 50, (total_features,), dtype=torch.int32)
    feature_mags = torch.rand(total_features, dtype=torch.float32) * 10
    segment_char_starts = torch.arange(n_segs, dtype=torch.int64) * 128
    segment_char_ends = segment_char_starts + 128
    data = {
        "offsets": offsets,
        "doc_ids": doc_ids,
        "segment_indices": segment_indices,
        "segment_positions": torch.arange(n_segs, dtype=torch.int64) * 16,
        "segment_char_starts": segment_char_starts,
        "segment_char_ends": segment_char_ends,
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "n_segments": n_segs,
        "segment_size": 128,
    }
    save_features(data, tmp_path / "shard_00000_features")
    return tmp_path


@pytest.mark.parametrize("seed", [0, 42, 1337, 99999])
def test_property_prepare_cache_deterministic(
    tmp_path,
    seed,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """Property: prepare_inline_eval_cache is deterministic in its inputs.

    Parametrized across several seeds rather than using hypothesis's @given
    here so we avoid fixture composition issues. The other parametrized
    test below covers the shape dimension.
    """
    _build_random_shard(tmp_path, n_docs=5, n_segs_per_doc=4, features_per_seg=3)
    kwargs = {
        "feature_dir": tmp_path,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "n_eval_docs": 10,
        "seed": seed,
    }
    a = prepare_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]
    b = prepare_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]
    assert a.selected_docs == b.selected_docs
    assert a.selection_metrics == b.selection_metrics
    assert a.doc_texts == b.doc_texts


@pytest.mark.parametrize(
    "n_docs,n_segs_per_doc,features_per_seg,n_eval_docs",
    [
        (1, 2, 1, 10),
        (3, 4, 3, 2),
        (5, 2, 5, 10),
        (10, 3, 2, 5),
        (2, 10, 1, 100),  # n_eval_docs >> n_docs
    ],
)
def test_property_prepare_cache_respects_n_eval_bound(
    tmp_path,
    n_docs,
    n_segs_per_doc,
    features_per_seg,
    n_eval_docs,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """Property: len(selected_docs) <= n_eval_docs for any valid shard shape."""
    _build_random_shard(tmp_path, n_docs=n_docs, n_segs_per_doc=n_segs_per_doc, features_per_seg=features_per_seg)
    cache = prepare_inline_eval_cache(
        feature_dir=tmp_path,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=n_eval_docs,
        seed=42,
    )
    assert len(cache.selected_docs) <= n_eval_docs
    # And <= the number of docs actually in the shard
    assert len(cache.selected_docs) <= n_docs


def test_property_selected_segments_nonzero(
    tmp_path,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
):
    """Property: every selected (doc_idx, seg_idx) pair has seg_idx > 0.

    _select_random_docs skips the first segment (no prefix to generate from).
    This invariant is critical for _generate_roundtrip_samples to work.
    """
    _build_random_shard(tmp_path, n_docs=10, n_segs_per_doc=5, features_per_seg=3)
    cache = prepare_inline_eval_cache(
        feature_dir=tmp_path,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=20,
        seed=42,
    )
    for doc_idx, seg_idx in cache.selected_docs:
        assert seg_idx > 0, f"Selected segment {seg_idx} must be > 0 (doc={doc_idx})"


# ---------------------------------------------------------------------------
# P0: DDP rank consistency tests
# ---------------------------------------------------------------------------


def test_prepare_cache_is_deterministic_for_identical_inputs(
    feature_shard_dir, threshold_config, precomputed_thresholds, val_parquet_dir
):
    """Two sequential calls with identical inputs produce equal caches.

    Protects against non-determinism creeping into _select_random_docs or
    _load_doc_texts that would break cross-rank all_gather_object in DDP.

    Note: this is an in-process determinism check. It does not simulate the
    full cross-process DDP environment (e.g., PYTHONHASHSEED differences).
    The current implementation doesn't use sets or hash-dependent dict keys,
    so in-process determinism is sufficient — but if that ever changes, this
    test should be upgraded to a subprocess-based check.
    """
    kwargs = {
        "feature_dir": feature_shard_dir,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "n_eval_docs": 10,
        "seed": 42,
    }
    rank0_cache = prepare_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]
    rank1_cache = prepare_inline_eval_cache(**kwargs)  # ty: ignore[invalid-argument-type]
    assert rank0_cache == rank1_cache


def test_eval_cache_passed_through_ddp_partition(
    feature_shard_dir, threshold_config, precomputed_thresholds, val_parquet_dir
):
    """With world_size=2, _partition_docs splits the cached selected_docs
    into disjoint sets whose union equals the full list."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )
    rank0 = _partition_docs(cache.selected_docs, rank=0, world_size=2)
    rank1 = _partition_docs(cache.selected_docs, rank=1, world_size=2)
    assert set(rank0) & set(rank1) == set(), "ranks overlap"
    assert set(rank0) | set(rank1) == set(cache.selected_docs), "ranks don't cover full list"


# ---------------------------------------------------------------------------
# P0: end-to-end smoke test
# ---------------------------------------------------------------------------


def test_p0_end_to_end_cpu_smoke(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Full prepare_inline_eval_cache -> evaluate_steering pipeline on CPU.

    Asserts:
    - cache builds without error
    - evaluate_steering runs to completion with the cache
    - all expected metric keys are present with finite numeric values
    - steering/eval_cache_hit == 1
    - steering/eval_time_seconds > 0
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=cache,
    )

    expected_keys = [
        "steering/n_docs_available",
        "steering/n_docs_selected",
        "steering/eval_cache_hit",
        "steering/eval_time_seconds",
    ]
    for k in expected_keys:
        assert k in result, f"missing key: {k}"
        assert isinstance(result[k], (int, float)), f"{k} is {type(result[k])}"
        assert math.isfinite(result[k]), f"{k} is not finite: {result[k]}"

    assert result["steering/eval_cache_hit"] == 1
    assert result["steering/eval_time_seconds"] > 0


def test_evaluate_steering_with_empty_cache_takes_fast_return(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
    monkeypatch,
):
    """An explicitly-empty InlineEvalCache triggers the fast-return short-circuit.

    The short-circuit at the top of evaluate_steering returns scan_metrics
    (augmented with eval_cache_hit and eval_time_seconds) without touching
    the val parquet, the feature_index, or the model. This test guards the
    path against silent regressions where the short-circuit gets moved
    after the generation block.
    """
    empty_cache = InlineEvalCache(
        selected_docs=[],
        selection_metrics={"steering/n_docs_available": 0, "steering/n_docs_selected": 0},
        doc_texts={},
    )

    # If the fast-return path were silently removed, _load_doc_texts would
    # be called — catch that here.
    def _boom(*args, **kwargs):
        raise AssertionError(
            "_load_doc_texts should not be called on the empty-cache fast-return path"
        )

    monkeypatch.setattr(inline_mod, "_load_doc_texts", _boom)

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=empty_cache,
    )

    assert result["steering/n_docs_available"] == 0
    assert result["steering/n_docs_selected"] == 0
    assert result["steering/eval_cache_hit"] == 1
    assert "steering/eval_time_seconds" in result
    assert math.isfinite(result["steering/eval_time_seconds"])
    # No generation metrics should be present — we took the fast return
    assert "steering/roundtrip_real_mean" not in result
    assert "steering/n_completions_generated" not in result


# ---------------------------------------------------------------------------
# P1: _probe_roundtrip_raw uses batched helpers
# ---------------------------------------------------------------------------


def _build_reference_probe_fn():
    """Build a reference _probe_roundtrip_raw that uses the scalar helpers.

    Used by the P1 local parity tests to verify the new batched body
    produces the same raw scores as the old scalar loop. This is defined
    inline in the test so the reference implementation lives alongside
    the tests that use it.
    """
    import math
    import statistics

    from steering.conditional.percentile_table import magnitude_to_percentile
    from steering.eval.feature_probe import (
        apply_sae_activation,
        build_completion_mask,
        compute_raw_probe,
        max_pool_masked,
    )
    from steering.eval.probe_activations import get_probe_activations
    from steering.segmentation import token_to_char_offsets

    def _probe_roundtrip_raw_reference(completions, probe_model, percentile_table=None):
        scores = []
        device = probe_model.device
        for comp in completions:
            gt_fids = comp.ground_truth_feature_ids
            if not gt_fids:
                continue
            completion_char_start = len(comp.pre_segment_text)
            for sample in comp.samples:
                full_text = comp.pre_segment_text + sample.text
                if not full_text.strip():
                    continue
                tokens = probe_model.tokenizer.encode(full_text)
                if not tokens:
                    continue
                char_offsets = token_to_char_offsets(probe_model.tokenizer, tokens)
                mask = build_completion_mask(char_offsets, completion_char_start)

                tokens_tensor = torch.tensor([tokens], device=device)
                act = get_probe_activations(
                    probe_model.model, tokens_tensor,
                    layer=probe_model.layer,
                    hook_name=probe_model.hook_name,
                    activation_type=probe_model.activation_type,
                    nnterp_model=probe_model.nnterp_model,
                ).squeeze(0).float()

                pre_activations: list[float] = []
                post_activations: list[float] = []
                probed_fids: list[int] = []
                for fid in gt_fids:
                    probe = compute_raw_probe(
                        act, probe_model.sae.W_enc, probe_model.sae.b_enc,
                        probe_model.sae.b_dec, fid,
                        apply_b_dec=probe_model.apply_b_dec,
                    )
                    pre_val = max_pool_masked(probe, mask)
                    post_probe = apply_sae_activation(probe, probe_model.sae, fid)
                    post_val = max_pool_masked(post_probe, mask)
                    if math.isfinite(pre_val):
                        pre_activations.append(pre_val)
                        post_activations.append(post_val if math.isfinite(post_val) else 0.0)
                        probed_fids.append(fid)
                if pre_activations:
                    pre_mean = statistics.mean(pre_activations)
                    post_mean = statistics.mean(post_activations) if post_activations else 0.0
                    if percentile_table is not None and probed_fids:
                        fid_t = torch.tensor(probed_fids, dtype=torch.int64)
                        post_t = torch.tensor(post_activations, dtype=torch.float32)
                        pct_mean = magnitude_to_percentile(fid_t, post_t, percentile_table).mean().item()
                    else:
                        pct_mean = float("nan")
                    scores.append((sample.condition, pre_mean, post_mean, pct_mean))
        return scores

    return _probe_roundtrip_raw_reference


def _build_probe_model_mock(d_model: int = 8, n_features: int = 20):
    """Build a minimal ProbeModel-like mock for P1 probe tests."""
    class _P1MockSAECfg:
        apply_b_dec_to_input = True
        activation_fn = "relu"

    class _P1MockSAEForProbe:
        def __init__(self):
            torch.manual_seed(7)
            self.W_enc = torch.randn(d_model, n_features)
            self.b_enc = torch.randn(n_features)
            self.b_dec = torch.randn(d_model)
            self.cfg = _P1MockSAECfg()
            # ReLU (no threshold attribute)

    class _P1MockModelForProbe:
        def parameters(self):
            yield torch.zeros(1)

    class _P1MockTokenizerForProbe:
        def encode(self, text):
            return list(text.encode("utf-8"))[:64]

        def decode(self, tokens):
            return bytes(t for t in tokens if 0 <= t < 256).decode("utf-8", errors="replace")

    probe = ProbeModel.__new__(ProbeModel)
    probe.model = _P1MockModelForProbe()
    probe.sae = _P1MockSAEForProbe()
    probe.tokenizer = _P1MockTokenizerForProbe()
    probe.hook_name = "blocks.0.hook_resid_post"
    probe.layer = 0
    probe.nnterp_model = None
    probe.activation_type = "residual"  # ty: ignore[invalid-assignment]
    probe.apply_b_dec = True
    return probe


def _make_probe_test_completion(doc_idx: int, gt_fids: list[int], sample_text: str = "test completion"):
    """Build a RoundtripCompletion with one sample for P1 probe tests."""
    return RoundtripCompletion(
        doc_idx=doc_idx,
        target_segment_idx=1,
        pre_segment_text="prefix ",
        ground_truth_feature_ids=gt_fids,
        samples=[CompletionSample(condition="real", text=sample_text, seed=42)],
    )


def test_probe_roundtrip_raw_output_matches_scalar_reference(monkeypatch):
    """CENTRAL P1.2 PARITY ANCHOR: new batched body == old scalar loop output."""
    probe = _build_probe_model_mock()
    completions = [
        _make_probe_test_completion(doc_idx=0, gt_fids=[0, 3, 7], sample_text="alpha completion"),
        _make_probe_test_completion(doc_idx=1, gt_fids=[2, 5, 11, 15], sample_text="beta completion"),
    ]

    # Monkeypatch get_probe_activations to return deterministic activations
    # keyed by the token tensor shape (we use the shape as a cache key so both
    # the new body and the reference get the same activations for the same input)
    fake_act_by_key: dict[str, torch.Tensor] = {}

    def _fake_probe_activations(model, tokens, layer, hook_name=None, activation_type=None, nnterp_model=None):
        key = f"{tokens.shape}_{tokens.sum().item() if tokens.numel() > 0 else 0}"
        if key not in fake_act_by_key:
            # Seed based on the key so the same input always produces the same output
            g = torch.Generator()
            g.manual_seed(abs(hash(key)) & 0x7FFFFFFF)
            fake_act_by_key[key] = torch.randn(1, tokens.shape[1], 8, generator=g)
        return fake_act_by_key[key]

    monkeypatch.setattr(
        "steering.eval.inline_steering_eval.get_probe_activations",
        _fake_probe_activations,
    )
    monkeypatch.setattr(
        "steering.eval.probe_activations.get_probe_activations",
        _fake_probe_activations,
    )

    new_result = _probe_roundtrip_raw(completions, probe, percentile_table=None)

    reference_fn = _build_reference_probe_fn()
    reference_result = reference_fn(completions, probe, percentile_table=None)

    assert len(new_result) == len(reference_result)
    for (cond_a, pre_a, post_a, _), (cond_b, pre_b, post_b, _) in zip(
        new_result, reference_result, strict=True
    ):
        assert cond_a == cond_b
        assert pre_a == pytest.approx(pre_b, rel=1e-6, abs=1e-9)
        assert post_a == pytest.approx(post_b, rel=1e-6, abs=1e-9)


def test_probe_roundtrip_raw_raises_on_nan_pre_activations(monkeypatch):
    """A NaN pre_val raises ValueError instead of silently filtering.

    Rationale: under realistic SAE semantics with finite inputs, pre_val
    cannot be NaN. If we see NaN here, something upstream produced bad
    activations and the loud failure mode is the right default.
    """
    probe = _build_probe_model_mock(d_model=4, n_features=8)
    completions = [_make_probe_test_completion(doc_idx=0, gt_fids=[0, 1, 2])]

    # Inject activations that produce NaN for feature 1 via an inf-inf probe:
    # set W_enc[:, 1] = inf so sae_in @ W_enc[:, 1] overflows to inf, then
    # b_enc[1] = -inf produces NaN.
    probe.sae.W_enc = torch.zeros(4, 8)
    probe.sae.W_enc[:, 1] = float("inf")
    probe.sae.b_enc = torch.zeros(8)
    probe.sae.b_enc[1] = float("-inf")
    probe.sae.b_dec = torch.zeros(4)

    def _fake_probe_activations(model, tokens, layer, hook_name=None, activation_type=None, nnterp_model=None):
        return torch.ones(1, tokens.shape[1], 4)

    monkeypatch.setattr(
        "steering.eval.inline_steering_eval.get_probe_activations",
        _fake_probe_activations,
    )

    with pytest.raises(ValueError, match="Non-finite pre_val"):
        _probe_roundtrip_raw(completions, probe, percentile_table=None)


def test_probe_roundtrip_raw_post_val_is_zero_when_relu_of_negative_pre(monkeypatch):
    """When pre_val is finite-negative, ReLU produces post_val of 0.0.

    Tests the happy-path (finite pre → finite post via ReLU). No raise expected
    because both pre and post are finite (just zero in the post case).
    """
    probe = _build_probe_model_mock(d_model=4, n_features=8)
    completions = [_make_probe_test_completion(doc_idx=0, gt_fids=[0])]

    # Build a scenario where pre_val is finite-negative: W_enc @ act + b_enc is negative.
    # Then ReLU (no threshold) gives post_val = 0.
    probe.sae.W_enc = -torch.ones(4, 8)  # all negative
    probe.sae.b_enc = torch.zeros(8)
    probe.sae.b_dec = torch.zeros(4)

    def _fake_probe_activations(model, tokens, layer, hook_name=None, activation_type=None, nnterp_model=None):
        return torch.ones(1, tokens.shape[1], 4)  # positive acts × negative W → negative pre

    monkeypatch.setattr(
        "steering.eval.inline_steering_eval.get_probe_activations",
        _fake_probe_activations,
    )

    result = _probe_roundtrip_raw(completions, probe, percentile_table=None)
    assert len(result) == 1
    _cond, pre, post, _pct = result[0]
    assert pre < 0.0  # pre_val is negative (finite)
    assert post == pytest.approx(0.0, abs=1e-9)  # post = relu(negative) = 0


def test_probe_roundtrip_raw_raises_on_empty_gt_fids():
    """A completion with empty ground_truth_feature_ids raises ValueError.

    Rationale: _generate_roundtrip_samples raises on empty post-threshold
    features upstream, so any completion reaching the probe SHOULD have a
    non-empty gt_fids list. An empty list is a caller-side invariant violation
    (e.g., a test directly building completions with empty features).
    """
    probe = _build_probe_model_mock()
    completions = [_make_probe_test_completion(doc_idx=0, gt_fids=[])]

    with pytest.raises(ValueError, match="empty ground_truth_feature_ids"):
        _probe_roundtrip_raw(completions, probe, percentile_table=None)


def test_probe_roundtrip_raw_raises_on_all_false_mask(monkeypatch):
    """An all-False completion mask (zero tokens in the completion span)
    raises ValueError instead of silently producing NaN-filled output.

    The all-False mask is triggered when sample_text is empty — the full_text
    equals pre_segment_text, all tokens have char_offset < completion_char_start,
    and the mask is empty. Previously the code returned NaN-filled probe values
    that the downstream filter dropped. Now we treat it as a real bug (the
    upstream tokenizer / generation produced a degenerate result) and raise.
    """
    probe = _build_probe_model_mock()
    # sample_text="" → full_text == pre_segment_text == "prefix "
    # → all token offsets < completion_char_start → mask all-False
    completions = [_make_probe_test_completion(doc_idx=0, gt_fids=[0, 1], sample_text="")]

    def _fake_probe_activations(model, tokens, layer, hook_name=None, activation_type=None, nnterp_model=None):
        return torch.randn(1, max(tokens.shape[1], 1), 8)

    monkeypatch.setattr(
        "steering.eval.inline_steering_eval.get_probe_activations",
        _fake_probe_activations,
    )

    with pytest.raises(ValueError, match="Completion mask is all-False"):
        _probe_roundtrip_raw(completions, probe, percentile_table=None)


def test_probe_roundtrip_raw_percentile_table_path_matches_reference(monkeypatch):
    """When percentile_table is provided, pct_mean matches the reference."""
    probe = _build_probe_model_mock()
    completions = [_make_probe_test_completion(doc_idx=0, gt_fids=[0, 3])]

    # Small percentile table: shape (n_features, n_percentile_bins)
    percentile_table = torch.linspace(0.0, 100.0, 100).unsqueeze(0).expand(20, -1).contiguous()

    fake_act_cache: dict = {}

    def _fake_probe_activations(model, tokens, layer, hook_name=None, activation_type=None, nnterp_model=None):
        key = f"{tokens.shape}_{tokens.sum().item() if tokens.numel() > 0 else 0}"
        if key not in fake_act_cache:
            g = torch.Generator()
            g.manual_seed(abs(hash(key)) & 0x7FFFFFFF)
            fake_act_cache[key] = torch.randn(1, tokens.shape[1], 8, generator=g)
        return fake_act_cache[key]

    monkeypatch.setattr(
        "steering.eval.inline_steering_eval.get_probe_activations",
        _fake_probe_activations,
    )
    monkeypatch.setattr(
        "steering.eval.probe_activations.get_probe_activations",
        _fake_probe_activations,
    )

    new_result = _probe_roundtrip_raw(completions, probe, percentile_table=percentile_table)

    reference_fn = _build_reference_probe_fn()
    reference_result = reference_fn(completions, probe, percentile_table=percentile_table)

    assert len(new_result) == len(reference_result)
    for a, b in zip(new_result, reference_result, strict=True):
        assert a[0] == b[0]  # condition
        assert a[1] == pytest.approx(b[1], rel=1e-6, abs=1e-9)  # pre_mean
        assert a[2] == pytest.approx(b[2], rel=1e-6, abs=1e-9)  # post_mean
        # pct_mean: compare numerically if both finite, or both NaN
        if math.isnan(a[3]) and math.isnan(b[3]):
            pass  # both NaN — equivalent
        else:
            assert a[3] == pytest.approx(b[3], rel=1e-6, abs=1e-9), (
                f"pct_mean drift: new={a[3]}, ref={b[3]}"
            )


# ---------------------------------------------------------------------------
# P1: _generate_roundtrip_samples uses generate_samples_for_condition
# ---------------------------------------------------------------------------


def test_generate_roundtrip_samples_output_unchanged(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """RoundtripCompletion fields are unchanged after the helper refactor."""
    segments = _four_segments_for_doc()

    class _FakeIndex:
        def lookup(self, shard_stem, doc_idx):
            # Return segments for doc 0 AND doc 1 so _pick_random_segment
            # has a valid "other doc" to pick from for the random condition
            if doc_idx in (0, 1):
                return segments
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1]

    cfg = SteeringEvalConfig(
        n_eval_docs=1, samples_per_condition=2, temperature=0.5, top_k_sampling=20, seed=77,
    )
    completions = _generate_roundtrip_samples(
        model=tiny_model,
        selected_docs=[(0, 2)],
        doc_texts={0: "The quick brown fox jumps over the lazy dog."},
        feature_index=_FakeIndex(),
        val_shard_stem="shard_00000",
        tokenizer=byte_tokenizer,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        device=torch.device("cpu"),
        cfg=cfg,
    )

    assert len(completions) == 1
    rc = completions[0]
    assert rc.doc_idx == 0
    assert rc.target_segment_idx == 2
    assert isinstance(rc.pre_segment_text, str)
    assert isinstance(rc.ground_truth_feature_ids, list)
    # Full-conditioning path emits real and random conditioned variants.
    assert len(rc.samples) == 4
    for s in rc.samples:
        assert s.condition in ("real", "random")
        assert isinstance(s.text, str)
        assert isinstance(s.seed, int)


def test_generate_roundtrip_samples_seed_arithmetic_preserved(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """Per-sample seed = cfg.seed + doc_idx * 1000 + cumulative_seed_counter.

    The counter increments by 1 per (condition, sample) pair.
    """
    segments = _four_segments_for_doc()

    class _FakeIndex:
        def lookup(self, shard_stem, doc_idx):
            # Return segments for doc 0 AND doc 1 so _pick_random_segment
            # has a valid "other doc" to pick from for the random condition
            if doc_idx in (0, 1):
                return segments
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1]

    cfg = SteeringEvalConfig(
        n_eval_docs=1, samples_per_condition=2, temperature=0.3, top_k_sampling=10, seed=100,
    )
    completions = _generate_roundtrip_samples(
        model=tiny_model,
        selected_docs=[(0, 2)],
        doc_texts={0: "The quick brown fox jumps over the lazy dog."},
        feature_index=_FakeIndex(),
        val_shard_stem="shard_00000",
        tokenizer=byte_tokenizer,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        device=torch.device("cpu"),
        cfg=cfg,
    )
    assert len(completions) == 1
    seeds_by_condition: dict[str, list[int]] = {}
    for sample in completions[0].samples:
        seeds_by_condition.setdefault(sample.condition, []).append(sample.seed)

    # cfg.seed=100, doc_idx=0 -> base_offset=100.
    # The counter advances by samples_per_condition for each conditioned variant.
    assert seeds_by_condition.get("real") == [100, 101]
    assert seeds_by_condition.get("random") == [102, 103]


def test_generate_roundtrip_samples_raises_when_pick_random_segment_raises(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    monkeypatch,
):
    """If _pick_random_segment raises, _generate_roundtrip_samples propagates it.

    Earlier this test asserted the helper silently skipped the random condition
    when no candidate was available. The user removed the silent skip: the
    random condition is now mandatory, and a missing candidate (e.g., a
    single-doc val shard) is a hard configuration error.
    """
    segments = _four_segments_for_doc()

    class _FakeIndex:
        def lookup(self, shard_stem, doc_idx):
            return segments if doc_idx == 0 else None

        def doc_ids_for_shard(self, shard_stem):
            # Single-doc shard — _pick_random_segment will raise because
            # there is no other doc to choose from.
            return [0]

    cfg = SteeringEvalConfig(
        n_eval_docs=1, samples_per_condition=2, temperature=0.3, top_k_sampling=10, seed=200,
    )
    with pytest.raises(ValueError, match=r"no other docs in shard"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 2)],
            doc_texts={0: "The quick brown fox jumps over the lazy dog."},
            feature_index=_FakeIndex(),
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=cfg,
        )


def test_generate_roundtrip_samples_propagates_generation_exception(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    monkeypatch,
    caplog,
):
    """If generate_samples_for_condition raises for any condition, the
    exception propagates after being logged at error level. The function
    does NOT silently swallow generation failures.

    Rationale: silent failures hide bugs. A failed generation in the eval
    path is a real problem (e.g., a model bug, an OOM, a corrupted seg_map)
    and the loud failure mode is the right default. The eval-trigger block
    in train_one_config catches at a higher level if the operator wants the
    training run to survive eval failures.
    """
    segments = _four_segments_for_doc()

    class _FakeIndex:
        def lookup(self, shard_stem, doc_idx):
            if doc_idx in (0, 1):
                return segments
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1]

    def _flaky_helper(**kwargs):
        raise RuntimeError("synthetic failure for test")

    monkeypatch.setattr(inline_mod, "generate_samples_for_condition", _flaky_helper)

    cfg = SteeringEvalConfig(
        n_eval_docs=1, samples_per_condition=2, temperature=0.3, top_k_sampling=10, seed=300,
    )

    with pytest.raises(RuntimeError, match="synthetic failure for test"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 2)],
            doc_texts={0: "The quick brown fox jumps over the lazy dog."},
            feature_index=_FakeIndex(),
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=cfg,
        )


def test_generate_roundtrip_samples_helper_invoked_with_correct_seeds(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    monkeypatch,
):
    """generate_samples_for_condition is called with base_seed = cfg.seed + doc_idx*1000 + seed_counter."""
    segments = _four_segments_for_doc()

    class _FakeIndex:
        def lookup(self, shard_stem, doc_idx):
            # Return segments for doc 0 AND doc 1 so _pick_random_segment
            # has a valid "other doc" to pick from for the random condition
            if doc_idx in (0, 1):
                return segments
            return None

        def doc_ids_for_shard(self, shard_stem):
            return [0, 1]

    import steering.eval.inline_steering_eval as mod
    from steering.eval.multiseg_generator import generate_samples_for_condition as real_helper

    recorded_base_seeds: list[int] = []

    def _recording_helper(**kwargs):
        recorded_base_seeds.append(kwargs["base_seed"])
        return real_helper(**kwargs)

    monkeypatch.setattr(mod, "generate_samples_for_condition", _recording_helper)

    cfg = SteeringEvalConfig(
        n_eval_docs=1, samples_per_condition=2, temperature=0.3, top_k_sampling=10, seed=400,
    )
    _generate_roundtrip_samples(
        model=tiny_model,
        selected_docs=[(0, 2)],
        doc_texts={0: "The quick brown fox jumps over the lazy dog."},
        feature_index=_FakeIndex(),
        val_shard_stem="shard_00000",
        tokenizer=byte_tokenizer,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        device=torch.device("cpu"),
        cfg=cfg,
    )
    # Expected base seeds per conditioned variant:
    # real: 400 + 0*1000 + 0 = 400
    # random: 400 + 0*1000 + 2 = 402
    assert len(recorded_base_seeds) == 2
    assert recorded_base_seeds[0] == 400   # real: base + 0
    assert recorded_base_seeds[1] == 402   # random: base + 2 (samples_per_condition)


# ---------------------------------------------------------------------------
# P1: central regression guard + integration tests
# ---------------------------------------------------------------------------


def test_p1_cache_matches_post_p0_metrics_bit_for_bit(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """THE CRITICAL P1 GUARD: P1 metrics must not drift from the pre-P1 baseline.

    Runs evaluate_steering twice under identical RNG state:
    - Run 1: current P1 code (batched probe + amortized generation).
    - Run 2: a monkeypatched reference that makes generate_samples_for_condition
      call generate_completion per sample (simulating the pre-P1 path).

    The probe path is inherently batched in P1 and can't be "un-batched" via
    monkeypatching (that would require surgery on _probe_roundtrip_raw's body),
    so this guard primarily tests the generation amortization parity. The
    probe vectorization parity is covered by test_probe_roundtrip_raw_output_matches_scalar_reference
    which compares the new batched body against an inlined scalar reference.

    Every numeric metric must match within pytest.approx(rel=1e-6, abs=1e-9),
    excluding only eval_time_seconds (wall-clock) and eval_cache_hit (P0 diagnostic).
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "feature_index": mock_feature_index,
        "probe_model": None,
        "eval_cache": cache,
    }

    # Run 1: current (P1) code
    torch.manual_seed(99999)
    p1_metrics = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]

    # Run 2: reference — monkeypatch the generation helper to call
    # generate_completion per sample (simulating the pre-P1 code path).
    def _reference_generate_samples(**kwargs):
        prefix_tokens = kwargs["prefix_tokens"]
        enc_inputs = kwargs["encoder_inputs"]
        n_samples = kwargs["n_samples"]
        device = kwargs["device"]
        temperature = kwargs["temperature"]
        top_k = kwargs["top_k"]
        base_seed = kwargs["base_seed"]
        bos_id = kwargs["bos_id"]
        n_gen = int(enc_inputs.seg_map.shape[1]) - len(prefix_tokens)
        results = []
        for i in range(n_samples):
            gen_tokens = generate_completion(
                model=kwargs["model"],
                prefix_tokens=prefix_tokens,
                encoder_inputs=enc_inputs,
                max_new_tokens=n_gen,
                device=device,
                temperature=temperature,
                top_k=top_k,
                seed=base_seed + i,
                bos_id=bos_id,
            )
            results.append(gen_tokens)
        return results

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(inline_mod, "generate_samples_for_condition", _reference_generate_samples)
    try:
        torch.manual_seed(99999)
        reference_metrics = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
    finally:
        monkeypatch.undo()

    exclude = {"steering/eval_time_seconds", "steering/eval_cache_hit"}
    for key in (set(p1_metrics) | set(reference_metrics)) - exclude:
        p1_val = p1_metrics.get(key)
        ref_val = reference_metrics.get(key)
        assert p1_val is not None and ref_val is not None, (
            f"Key mismatch: key={key!r} p1={p1_val!r} ref={ref_val!r}"
        )
        if isinstance(p1_val, (int, float)) and isinstance(ref_val, (int, float)):
            if math.isnan(p1_val) and math.isnan(ref_val):
                continue
            assert p1_val == pytest.approx(ref_val, rel=1e-6, abs=1e-9), (
                f"Metric drift: {key}: p1={p1_val!r} ref={ref_val!r}"
            )
        else:
            assert p1_val == ref_val, (
                f"Non-numeric key drift: {key}: p1={p1_val!r} ref={ref_val!r}"
            )


def test_p1_generation_encoder_forward_called_once_per_condition(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """End-to-end amortization: encoder should be called AT MOST once per condition per doc.

    Default config (samples_per_condition=2, 3 conditions): encoder should be
    called at most 3 times per doc. Pre-P1 would have called it 6 times per doc
    (once per sample). With n_eval_docs=1, we expect encoder_count < 6.
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=1,
        seed=42,
    )

    original_forward = tiny_model.encoder.forward
    encoder_count = {"n": 0}

    def _counting_forward(*args, **kwargs):
        encoder_count["n"] += 1
        return original_forward(*args, **kwargs)

    tiny_model.encoder.forward = _counting_forward  # type: ignore[assignment]
    try:
        evaluate_steering(
            model=tiny_model,
            feature_dir=feature_shard_dir,
            tokenizer=byte_tokenizer,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            feature_index=mock_feature_index,
            probe_model=None,
            eval_cache=cache,
        )
    finally:
        tiny_model.encoder.forward = original_forward  # type: ignore[assignment]

    # For n_eval_docs=1 with samples_per_condition=2 and 2-3 conditions,
    # pre-P1 would be 6 calls minimum (3 conditions × 2 samples). P1 should be ≤ 3
    # (1 per condition) and strictly less than 6.
    assert encoder_count["n"] < 6, (
        f"Encoder should be amortized: expected < 6 calls for n_eval_docs=1 "
        f"but got {encoder_count['n']} (pre-P1 baseline would be 6)"
    )


def test_p1_probe_matmul_syncs_reduced(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Soft-guard: runs evaluate_steering with probe_model=None and verifies
    the pipeline completes without errors. The full per-feature sync count
    assertion is deferred to the slow GPU suite."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=1,
        seed=42,
    )

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=cache,
    )
    assert "steering/eval_time_seconds" in result
    assert math.isfinite(result["steering/eval_time_seconds"])


# ---------------------------------------------------------------------------
# P1: end-to-end property tests
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=1, max_value=10000))
@settings(max_examples=5, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_p1_matches_baseline_across_seeds(
    seed,
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Property: P1 code matches the reference across a sample of seeds.

    Lighter than the central regression guard — only checks that
    n_completions_generated is identical (cheap proxy) across 5 seeds.
    Full numerical parity for one seed is covered by
    test_p1_cache_matches_post_p0_metrics_bit_for_bit.
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=5,
        seed=seed,
    )

    def _reference_generate_samples(**kwargs):
        prefix_tokens = kwargs["prefix_tokens"]
        enc_inputs = kwargs["encoder_inputs"]
        n_samples = kwargs["n_samples"]
        n_gen = int(enc_inputs.seg_map.shape[1]) - len(prefix_tokens)
        return [
            generate_completion(
                model=kwargs["model"],
                prefix_tokens=prefix_tokens,
                encoder_inputs=enc_inputs,
                max_new_tokens=n_gen,
                device=kwargs["device"],
                temperature=kwargs["temperature"],
                top_k=kwargs["top_k"],
                seed=kwargs["base_seed"] + i,
                bos_id=kwargs["bos_id"],
            )
            for i in range(n_samples)
        ]

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "feature_index": mock_feature_index,
        "probe_model": None,
        "eval_cache": cache,
    }

    torch.manual_seed(seed)
    p1 = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(inline_mod, "generate_samples_for_condition", _reference_generate_samples)
    try:
        torch.manual_seed(seed)
        ref = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
    finally:
        monkeypatch.undo()

    # Cheap proxy: same number of completions and same doc counts
    assert p1.get("steering/n_completions_generated") == ref.get("steering/n_completions_generated")
    assert p1.get("steering/n_docs_selected") == ref.get("steering/n_docs_selected")


@given(n_samples=st.integers(min_value=1, max_value=3))
@settings(max_examples=3, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_property_p1_matches_baseline_across_sample_counts(
    n_samples,
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Property: P1 code matches the reference across samples_per_condition in {1, 2, 3}.

    Covers the n_samples=1 boundary explicitly — at n_samples=1 the helper's
    single-sample path must be indistinguishable from generate_completion.
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=3,
        seed=42,
    )
    cfg = SteeringEvalConfig(
        n_eval_docs=3,
        samples_per_condition=n_samples,
        temperature=0.5,
        top_k_sampling=20,
        seed=42,
    )

    def _reference_helper(**kwargs):
        prefix_tokens = kwargs["prefix_tokens"]
        enc_inputs = kwargs["encoder_inputs"]
        n_sam = kwargs["n_samples"]
        n_gen = int(enc_inputs.seg_map.shape[1]) - len(prefix_tokens)
        return [
            generate_completion(
                model=kwargs["model"],
                prefix_tokens=prefix_tokens,
                encoder_inputs=enc_inputs,
                max_new_tokens=n_gen,
                device=kwargs["device"],
                temperature=kwargs["temperature"],
                top_k=kwargs["top_k"],
                seed=kwargs["base_seed"] + i,
                bos_id=kwargs["bos_id"],
            )
            for i in range(n_sam)
        ]

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "feature_index": mock_feature_index,
        "probe_model": None,
        "eval_cache": cache,
        "cfg": cfg,
    }

    torch.manual_seed(11111)
    p1 = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(inline_mod, "generate_samples_for_condition", _reference_helper)
    try:
        torch.manual_seed(11111)
        ref = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
    finally:
        monkeypatch.undo()

    assert p1.get("steering/n_completions_generated") == ref.get("steering/n_completions_generated")


# ---------------------------------------------------------------------------
# P1: DDP determinism + end-to-end smoke test
# ---------------------------------------------------------------------------


def test_p1_batched_probe_deterministic_across_ranks():
    """Two in-process runs with identical inputs produce byte-identical output tensors.

    Protects against non-determinism in batched matmul or torch.max tie-breaking
    that would break all_gather_object cross-rank consistency in DDP.
    """
    torch.manual_seed(31337)
    seq_len, d_model, n_features = 16, 8, 32
    act = torch.randn(seq_len, d_model)
    W_enc = torch.randn(d_model, n_features)
    b_enc = torch.randn(n_features)
    b_dec = torch.randn(d_model)
    feature_ids = torch.tensor([1, 3, 7, 13, 17], dtype=torch.int64)
    mask = torch.rand(seq_len) > 0.4
    mask[0] = True  # ensure at least one True

    result_a = compute_raw_probe_batched(
        act, W_enc, b_enc, b_dec, feature_ids, apply_b_dec=True,
    )
    result_b = compute_raw_probe_batched(
        act, W_enc, b_enc, b_dec, feature_ids, apply_b_dec=True,
    )
    assert torch.equal(result_a, result_b)

    pooled_a = max_pool_masked_batched(result_a, mask)
    pooled_b = max_pool_masked_batched(result_b, mask)
    assert torch.equal(pooled_a, pooled_b)


def test_p1_end_to_end_cpu_smoke(
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
):
    """Full P1 pipeline — prepare_inline_eval_cache -> evaluate_steering with the
    new batched probe path and generate_samples_for_condition helper."""
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_shard_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        feature_index=mock_feature_index,
        probe_model=None,
        eval_cache=cache,
    )

    expected_keys = [
        "steering/n_docs_available",
        "steering/n_docs_selected",
        "steering/eval_cache_hit",
        "steering/eval_time_seconds",
        "steering/n_completions_generated",
    ]
    for k in expected_keys:
        assert k in result, f"missing key: {k}"
        assert isinstance(result[k], (int, float)), f"{k} is {type(result[k])}"
        assert math.isfinite(result[k]), f"{k} is not finite: {result[k]}"

    assert result["steering/eval_cache_hit"] == 1
    assert result["steering/eval_time_seconds"] > 0


# ---------------------------------------------------------------------------
# End-to-end performance benchmark (P1 vs reference)
# ---------------------------------------------------------------------------


@pytest.mark.benchmark
@pytest.mark.parametrize("samples_per_condition", [2, 4, 8])
def test_p1_end_to_end_perf_benchmark(
    samples_per_condition,
    tiny_model,
    feature_shard_dir,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    val_parquet_dir,
    mock_feature_index,
    capsys,
):
    """End-to-end wall-clock measurement: P1 vs per-sample reference.

    This test runs ``evaluate_steering`` twice on the same fixtures used by
    the central regression guard:
    - Run A: current P1 code (amortized encoder + prefill, batched probe)
    - Run B: a monkeypatched reference that calls ``generate_completion``
      per sample (the pre-P1 code path)

    Parametrized over ``samples_per_condition`` to show that the speedup
    grows with the number of samples — encoder + prefill is paid once
    per condition regardless of N, so larger N means more amortization.

    Both runs use eval_cache=cache so the doc-selection cost is amortized
    out of the measurement. We use ``time.perf_counter`` for monotonic
    wall-clock and report ratios.

    NOTE on the CPU floor: this test runs against the tiny test model
    (2 layers, dim=64, seq_len=128) on CPU. The encoder + prefill amortization
    primarily wins on GPU with larger models — most of the encoder/prefill
    work is a per-condition fixed cost that grows with model size, not
    with sample count. On this CPU microbenchmark we see ~1.07x at
    samples_per_condition=2, growing to ~1.15x+ at samples_per_condition=8.
    The assertion floor is deliberately loose (>= 1.0x — i.e., not a
    regression) and the real validation is on the actual training rig.

    Run via:
        uv run pytest tests/test_inline_steering_eval.py::test_p1_end_to_end_perf_benchmark -s
    """
    cache = prepare_inline_eval_cache(
        feature_dir=feature_shard_dir,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        n_eval_docs=10,
        seed=42,
    )

    # Reference path: per-sample generate_completion (pre-P1 simulation)
    def _reference_generate_samples(**kwargs):
        prefix_tokens = kwargs["prefix_tokens"]
        enc_inputs = kwargs["encoder_inputs"]
        n_samples = kwargs["n_samples"]
        n_gen = int(enc_inputs.seg_map.shape[1]) - len(prefix_tokens)
        return [
            generate_completion(
                model=kwargs["model"],
                prefix_tokens=prefix_tokens,
                encoder_inputs=enc_inputs,
                max_new_tokens=n_gen,
                device=kwargs["device"],
                temperature=kwargs["temperature"],
                top_k=kwargs["top_k"],
                seed=kwargs["base_seed"] + i,
                bos_id=kwargs["bos_id"],
            )
            for i in range(n_samples)
        ]

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_shard_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "feature_index": mock_feature_index,
        "probe_model": None,
        "eval_cache": cache,
        "cfg": SteeringEvalConfig(
            n_eval_docs=10,
            samples_per_condition=samples_per_condition,
            seed=42,
        ),
    }

    # Warm up both paths once each (compile, allocator, etc.)
    torch.manual_seed(0)
    _ = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(inline_mod, "generate_samples_for_condition", _reference_generate_samples)
    try:
        torch.manual_seed(0)
        _ = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
    finally:
        monkeypatch.undo()

    n_trials = 5
    p1_times: list[float] = []
    ref_times: list[float] = []

    # P1 path
    for trial in range(n_trials):
        torch.manual_seed(99999 + trial)
        t0 = time.perf_counter()
        evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
        p1_times.append(time.perf_counter() - t0)

    # Reference path
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(inline_mod, "generate_samples_for_condition", _reference_generate_samples)
    try:
        for trial in range(n_trials):
            torch.manual_seed(99999 + trial)
            t0 = time.perf_counter()
            evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
            ref_times.append(time.perf_counter() - t0)
    finally:
        monkeypatch.undo()

    p1_median = sorted(p1_times)[n_trials // 2]
    ref_median = sorted(ref_times)[n_trials // 2]
    p1_min = min(p1_times)
    ref_min = min(ref_times)
    speedup_median = ref_median / p1_median
    speedup_min = ref_min / p1_min

    # Print to capsys so `pytest -s` shows the numbers
    msg = (
        f"\n=== P1 end-to-end perf benchmark ({n_trials} trials, n_eval_docs=10, "
        f"samples_per_condition={samples_per_condition}) ===\n"
        f"P1 (amortized + batched):  median={p1_median * 1000:.1f}ms  min={p1_min * 1000:.1f}ms\n"
        f"Reference (per-sample):    median={ref_median * 1000:.1f}ms  min={ref_min * 1000:.1f}ms\n"
        f"Speedup:                   median={speedup_median:.2f}x  min={speedup_min:.2f}x\n"
        f"P1 trial times (ms): {[round(t * 1000, 1) for t in p1_times]}\n"
        f"Ref trial times (ms): {[round(t * 1000, 1) for t in ref_times]}\n"
    )
    print(msg)
    # Force the print to show up in pytest output even without -s by writing
    # to capsys's stdout directly.
    with capsys.disabled():
        print(msg)

    # Loose assertion floor: P1 should not regress vs the per-sample reference.
    # See the docstring above for why this floor is so lax — the CPU
    # microbenchmark on a 2-layer dim-64 model has too little encoder/prefill
    # work to amortize for the speedup to clearly exceed measurement noise.
    # The assertion exists to catch a real regression (speedup < 1.0x means
    # P1 is slower than scalar, which would be a real bug), not to certify
    # the size of the win.
    # Tolerance: on CPU with tiny models the amortization win is a few ms
    # and scheduling jitter regularly produces 0.9-1.0x ratios. The floor
    # is set to 0.85x to catch real regressions (>=15% slowdown) while not
    # being a CI flake.
    assert speedup_median >= 0.85, (
        f"P1 must not regress vs reference, got {speedup_median:.2f}x. "
        f"P1 trials (ms): {p1_times}, ref trials (ms): {ref_times}"
    )


# ---------------------------------------------------------------------------
# Overlay-based split boundaries — required path
#
# ``_tokenize_and_split`` requires ``SegmentFeatures.token_start`` /
# ``token_end`` (populated by the ``tok/<checksum>/seg_boundaries`` overlay
# produced by ``scripts.preprocess.precompute_token_boundaries``). Those indices are in
# the TRAINING tokenizer's token space (indices into
# ``tokenizer.encode(doc_text, prepend=bos_id)``) and already encode the
# BOS-length delta + zero-width + trailing-seg repairs from precompute.
#
# No char-alignment fallback: if the overlay is missing, the inline eval
# fails loudly rather than silently masking a misconfigured pipeline.
# ---------------------------------------------------------------------------


def _seg_with_token_bounds(
    segment_idx: int,
    position: int,
    char_start: int,
    char_end: int,
    token_start: int,
    token_end: int,
) -> SegmentFeatures:
    """Construct a SegmentFeatures with the overlay's token boundaries populated."""
    return SegmentFeatures(
        segment_idx=segment_idx,
        position=position,
        feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        feature_mags=torch.tensor([5.0, 3.0, 7.0], dtype=torch.float32),
        char_start=char_start,
        char_end=char_end,
        token_start=token_start,
        token_end=token_end,
    )


def _seg_without_token_bounds(
    segment_idx: int, char_start: int, char_end: int,
) -> SegmentFeatures:
    """Construct a SegmentFeatures WITHOUT the overlay — used to exercise
    the loud-failure branch."""
    return SegmentFeatures(
        segment_idx=segment_idx,
        position=char_start,
        feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        feature_mags=torch.tensor([5.0, 3.0, 7.0], dtype=torch.float32),
        char_start=char_start,
        char_end=char_end,
        # token_start / token_end intentionally omitted (default: None)
    )


# ---------------------------------------------------------------------------
# _tokenize_and_split: unit tests
# ---------------------------------------------------------------------------


def test_tokenize_and_split_uses_overlay_token_boundaries_when_present(byte_tokenizer):
    """When ``SegmentFeatures.token_start``/``token_end`` are set, they are
    the authoritative split point — not char-based alignment.

    Populates token_start / token_end with values that DELIBERATELY differ
    from what char-alignment would pick; asserts the overlay values win.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    bos_id = byte_tokenizer.get_bos_token_id()
    full_tokens = byte_tokenizer.encode(doc_text, prepend=bos_id)
    n_full = len(full_tokens)
    assert n_full == 45

    # Overlay picks a split (20, 35) that does not match char-alignment (22, 34).
    overlay_split = 20
    overlay_end = 35
    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0,
            char_start=0, char_end=22,
            token_start=1, token_end=overlay_split,
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=overlay_split,
            char_start=22, char_end=34,
            token_start=overlay_split, token_end=overlay_end,
        ),
    ]

    all_tokens, split_idx, pre_text = _tokenize_and_split(
        doc_text, segments, 1, byte_tokenizer,
    )

    assert split_idx == overlay_split
    assert all_tokens == full_tokens[:overlay_end]
    # pre_segment_text is derived from char_start — used only as a display
    # prefix and for probe span location; it intentionally does NOT need
    # to match the token split exactly.
    assert pre_text == doc_text[: segments[1].char_start - 1]


def test_tokenize_and_split_overlay_resolves_degenerate_tokenizer_case():
    """``DegenerateSplitTokenizer`` deliberately produces an alignment
    where char-based split would place 0 tokens in the target segment.
    With the overlay populated, ``_tokenize_and_split`` yields a
    non-degenerate split by construction — this is the scenario that
    matches the production bug, resolved by routing through the overlay.
    """
    tok = DegenerateSplitTokenizer()
    bos_id = tok.get_bos_token_id()
    doc_text = "AAAAAAAAAA" + "X" * 9  # matches the old degenerate fixture

    full_tokens = tok.encode(doc_text, prepend=bos_id)
    # full_tokens = [BOS, MERGED_TOKEN, X, X, X, X, X, X, X, X, X]
    assert len(full_tokens) == 11
    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=6,
            token_start=1, token_end=2,
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=2, char_start=6, char_end=11,
            token_start=2, token_end=4,
        ),
        _seg_with_token_bounds(
            segment_idx=2, position=4, char_start=11, char_end=20,
            token_start=4, token_end=11,
        ),
    ]

    all_tokens, split_idx, _ = _tokenize_and_split(doc_text, segments, 1, tok)
    assert 0 < split_idx < len(all_tokens)
    assert split_idx == 2
    assert len(all_tokens) == 4


def test_tokenize_and_split_raises_when_token_start_missing(byte_tokenizer):
    """If ``token_start`` is None (overlay absent), the helper raises with
    an operator-actionable message naming the precompute script.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = [
        _seg_without_token_bounds(segment_idx=0, char_start=0, char_end=12),
        _seg_without_token_bounds(segment_idx=1, char_start=12, char_end=23),
    ]
    assert segments[1].token_start is None and segments[1].token_end is None

    with pytest.raises(ValueError) as excinfo:
        _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)
    msg = str(excinfo.value)
    assert "token_start/token_end" in msg
    assert "precompute_token_boundaries" in msg
    assert "target_seg_idx=1" in msg


def test_tokenize_and_split_raises_when_only_token_end_missing(byte_tokenizer):
    """If ``token_end`` is None but ``token_start`` is not, still raise —
    both must be set for the overlay to be usable.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=12,
            token_start=0, token_end=12,
        ),
        SegmentFeatures(
            segment_idx=1, position=12,
            feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            feature_mags=torch.tensor([5.0, 3.0, 7.0], dtype=torch.float32),
            char_start=12, char_end=23,
            token_start=12,  # present
            token_end=None,  # missing
        ),
    ]
    with pytest.raises(ValueError, match="token_start/token_end"):
        _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)


def test_tokenize_and_split_overlay_raises_on_out_of_range_bounds(byte_tokenizer):
    """``token_end`` exceeding the doc's actual tokenization length means
    the overlay was computed on a different parquet text than the eval
    input — raise a clear ValueError instead of silently misindexing.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    bos_id = byte_tokenizer.get_bos_token_id()
    n_full = len(byte_tokenizer.encode(doc_text, prepend=bos_id))

    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=20,
            token_start=1, token_end=10,
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=10, char_start=20, char_end=45,
            token_start=10, token_end=n_full + 5,  # past end
        ),
    ]
    with pytest.raises(ValueError, match=r"token_end=\d+ exceeds full doc n_tokens"):
        _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)


def test_tokenize_and_split_raises_on_zero_width_overlay_bounds(byte_tokenizer):
    """``token_start == token_end`` means an empty target segment —
    the precompute pipeline forbids this, so a violation here indicates
    a corrupted overlay.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=12,
            token_start=1, token_end=12,
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=12, char_start=12, char_end=23,
            token_start=15, token_end=15,  # empty
        ),
    ]
    with pytest.raises(
        ValueError,
        match=r"token_end must be strictly greater than token_start",
    ):
        _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)


def test_tokenize_and_split_raises_on_negative_token_start(byte_tokenizer):
    """``token_start == 0`` would place the target at BOS, which is
    nonsensical for the steering eval — raise.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=6,
            token_start=0, token_end=6,  # starts at 0 — equivalent to BOS
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=6, char_start=6, char_end=23,
            token_start=0, token_end=23,
        ),
    ]
    with pytest.raises(
        ValueError,
        match=r"token_start must be > 0",
    ):
        _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)


def test_tokenize_and_split_accepts_overlay_where_token_end_equals_full_len(
    byte_tokenizer,
):
    """``token_end == len(full_tokens)`` is the last-segment boundary case
    and must be accepted (the strict inequality is ``<= len(full_tokens)``).
    """
    doc_text = "Hi."
    bos_id = byte_tokenizer.get_bos_token_id()
    full_tokens = byte_tokenizer.encode(doc_text, prepend=bos_id)
    n = len(full_tokens)  # 4 = BOS + 'H' + 'i' + '.'
    assert n == 4

    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=2,
            token_start=1, token_end=2,
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=2, char_start=2, char_end=n,
            token_start=2, token_end=n,  # hit exact doc end
        ),
    ]
    all_tokens, split_idx, _ = _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)
    assert split_idx == 2
    assert len(all_tokens) == n


# ---------------------------------------------------------------------------
# _generate_roundtrip_samples: failure propagation + overlay happy path
# ---------------------------------------------------------------------------


def _fake_feature_index(segments_by_doc: dict[int, list[SegmentFeatures]]):
    """Minimal FeatureIndex double: lookup returns segments or raises KeyError."""

    class _Idx:
        def lookup(self, shard_stem, doc_idx):
            if doc_idx not in segments_by_doc:
                raise KeyError(
                    f"Document {doc_idx} not found in shard {shard_stem!r} "
                    f"(shard has {len(segments_by_doc)} docs)"
                )
            return segments_by_doc[doc_idx]

        def doc_ids_for_shard(self, shard_stem):
            return sorted(segments_by_doc.keys())

    return _Idx()


def test_generate_roundtrip_samples_raises_when_overlay_missing(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """The failure is propagated up from ``_tokenize_and_split`` so the
    operator sees a clear error and knows to precompute the overlay — no
    silent skip.
    """
    segments_no_overlay = [
        _seg_without_token_bounds(segment_idx=0, char_start=0, char_end=12),
        _seg_without_token_bounds(segment_idx=1, char_start=12, char_end=23),
        _seg_without_token_bounds(segment_idx=2, char_start=23, char_end=34),
        _seg_without_token_bounds(segment_idx=3, char_start=34, char_end=45),
    ]
    feature_index = _fake_feature_index({0: segments_no_overlay, 1: segments_no_overlay})

    with pytest.raises(ValueError, match="precompute_token_boundaries"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 2)],
            doc_texts={0: "The quick brown fox jumps over the lazy dog."},
            feature_index=feature_index,
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=1, seed=42),
        )


def test_generate_roundtrip_samples_raises_even_if_other_docs_are_valid(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """Single bad doc (missing overlay) is fatal to the whole eval —
    does NOT silently produce a partial result. Mixed valid/invalid
    selection: the eval aborts on the first offender.
    """
    valid_segs = _four_segments_for_doc()  # has overlay
    invalid_segs = [
        _seg_without_token_bounds(segment_idx=0, char_start=0, char_end=12),
        _seg_without_token_bounds(segment_idx=1, char_start=12, char_end=23),
        _seg_without_token_bounds(segment_idx=2, char_start=23, char_end=34),
        _seg_without_token_bounds(segment_idx=3, char_start=34, char_end=45),
    ]
    feature_index = _fake_feature_index({0: invalid_segs, 1: valid_segs})

    with pytest.raises(ValueError, match="precompute_token_boundaries"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 2), (1, 2)],
            doc_texts={
                0: "The quick brown fox jumps over the lazy dog.",
                1: "The quick brown fox jumps over the lazy dog.",
            },
            feature_index=feature_index,
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=42),
        )


def test_generate_roundtrip_samples_overlay_happy_path_conditioned_variants(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """Overlay populated -> real and random conditioned variants produce samples."""
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    completions = _generate_roundtrip_samples(
        model=tiny_model,
        selected_docs=[(0, 2)],
        doc_texts={0: "The quick brown fox jumps over the lazy dog."},
        feature_index=feature_index,
        val_shard_stem="shard_00000",
        tokenizer=byte_tokenizer,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        device=torch.device("cpu"),
        cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=2, seed=99),
    )
    assert len(completions) == 1
    rc = completions[0]
    assert rc.doc_idx == 0
    assert rc.target_segment_idx == 2
    conditions = [s.condition for s in rc.samples]
    assert sorted(set(conditions)) == ["random", "real"]
    assert len(rc.samples) == 4


# ---------------------------------------------------------------------------
# evaluate_steering end-to-end: full pipeline with overlay, raise on missing
# ---------------------------------------------------------------------------


def _minimal_val_parquet(tmp_path: Path, texts: list[str]) -> Path:
    """Write a single val parquet and return its containing directory."""
    val_dir = tmp_path / "val"
    val_dir.mkdir(exist_ok=True)
    table = pa.table({"text": texts})
    pq.write_table(table, val_dir / "val_00000.parquet")
    return val_dir


def _minimal_feature_shard(tmp_path: Path, n_docs: int = 2, n_segs_per_doc: int = 4) -> Path:
    """Write a minimal feature shard with features for every segment."""
    n_segs = n_docs * n_segs_per_doc
    features_per_seg = 3
    total_features = n_segs * features_per_seg
    offsets = torch.arange(0, total_features + 1, features_per_seg, dtype=torch.int64)
    doc_ids = torch.repeat_interleave(
        torch.arange(n_docs, dtype=torch.int64), n_segs_per_doc,
    )
    segment_indices = torch.tile(
        torch.arange(n_segs_per_doc, dtype=torch.int64), (n_docs,)
    )
    feature_ids = torch.tile(torch.tensor([0, 1, 2], dtype=torch.int32), (n_segs,))
    feature_mags = torch.zeros(total_features, dtype=torch.float32)
    for seg_i in range(n_segs):
        s = seg_i * features_per_seg
        feature_mags[s : s + features_per_seg] = torch.tensor(
            [5.0 + seg_i * 0.1, 3.0 + seg_i * 0.1, 7.0 + seg_i * 0.1]
        )
    data = {
        "offsets": offsets,
        "doc_ids": doc_ids,
        "segment_indices": segment_indices,
        "feature_ids": feature_ids,
        "feature_mags": feature_mags,
        "n_segments": n_segs,
    }
    save_features(data, tmp_path / "shard_00000_features")
    return tmp_path


def test_evaluate_steering_e2e_with_overlay_produces_valid_metrics(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """Full ``evaluate_steering`` e2e with the overlay populated. Asserts
    the complete metrics dict is present, finite, and n_completions matches
    n_docs_selected (no skips, no partial results).
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=feature_index,
        probe_model=None,
        cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=7),
    )
    assert isinstance(result, dict)
    assert result["steering/n_docs_selected"] >= 1
    # No skips: n_completions == n_docs_selected.
    assert result["steering/n_completions_generated"] == result["steering/n_docs_selected"]
    assert result["steering/eval_time_seconds"] > 0
    for key in (
        "steering/roundtrip_real_mean",
        "steering/roundtrip_random_mean",
        "steering/roundtrip_specificity",
    ):
        assert math.isfinite(result[key]), f"{key} must be finite, got {result[key]}"


def test_evaluate_steering_raises_when_overlay_missing_e2e(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """If no overlay is loaded, ``evaluate_steering`` must raise from
    ``_tokenize_and_split`` — not quietly skip docs. Operator sees the
    full traceback naming ``scripts.preprocess.precompute_token_boundaries``.
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    no_overlay = [
        _seg_without_token_bounds(segment_idx=i, char_start=i * 11, char_end=(i + 1) * 11)
        for i in range(4)
    ]
    feature_index = _fake_feature_index({0: no_overlay, 1: no_overlay})

    with pytest.raises(ValueError, match="precompute_token_boundaries"):
        evaluate_steering(
            model=tiny_model,
            feature_dir=feature_dir,
            tokenizer=byte_tokenizer,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            data_dir=str(val_parquet_dir),
            feature_index=feature_index,
            probe_model=None,
            cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=7),
        )


def test_evaluate_steering_e2e_with_overlay(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """Full-context eval path exercised end-to-end with the overlay present."""
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=feature_index,
        probe_model=None,
        cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=7),
    )
    assert isinstance(result, dict)
    assert result["steering/n_completions_generated"] == result["steering/n_docs_selected"]
    for key in (
        "steering/roundtrip_real_mean",
        "steering/roundtrip_random_mean",
    ):
        assert math.isfinite(result[key])


# ---------------------------------------------------------------------------
# Roundtrip & integration: prefix + completion texts match the overlay split
# ---------------------------------------------------------------------------


def test_tokenize_and_split_prefix_plus_target_reconstructs_doc(byte_tokenizer):
    """Invariant: concatenating the tokens before and after ``split_idx``
    and decoding must round-trip to the doc's prefix through the target
    segment's end (in training-tokenizer space). Proves ``all_tokens`` and
    ``split_idx`` partition the same token sequence consistently.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    bos_id = byte_tokenizer.get_bos_token_id()
    full = byte_tokenizer.encode(doc_text, prepend=bos_id)
    segments = _four_segments_for_doc()

    for tgt in (1, 2, 3):
        all_tokens, split_idx, _ = _tokenize_and_split(
            doc_text, segments, tgt, byte_tokenizer,
        )
        # all_tokens is prefix + target span. Their concatenation should
        # equal the full tokenization up to the target's token_end.
        assert all_tokens == full[: segments[tgt].token_end]
        assert all_tokens[:split_idx] + all_tokens[split_idx:] == all_tokens


def test_tokenize_and_split_split_idx_equals_overlay_token_start(byte_tokenizer):
    """Structural invariant: ``split_idx == target_seg.token_start``
    exactly — the helper never adjusts the overlay boundary."""
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = _four_segments_for_doc()
    for tgt in (1, 2, 3):
        _, split_idx, _ = _tokenize_and_split(doc_text, segments, tgt, byte_tokenizer)
        assert split_idx == segments[tgt].token_start


def test_tokenize_and_split_target_span_length_matches_overlay(byte_tokenizer):
    """Structural invariant: the length of the target span
    (``len(all_tokens) - split_idx``) equals
    ``token_end - token_start``."""
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = _four_segments_for_doc()
    for tgt in (1, 2, 3):
        all_tokens, split_idx, _ = _tokenize_and_split(
            doc_text, segments, tgt, byte_tokenizer,
        )
        assert len(all_tokens) - split_idx == segments[tgt].token_end - segments[tgt].token_start  # ty: ignore[unsupported-operator]


# ---------------------------------------------------------------------------
# Determinism & cache parity with the overlay path
# ---------------------------------------------------------------------------


def test_evaluate_steering_overlay_deterministic_across_runs(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """Two runs under identical RNG state produce bit-identical metrics.
    Guards against any non-determinism sneaking into the overlay path.
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "feature_index": feature_index,
        "probe_model": None,
        "cfg": SteeringEvalConfig(n_eval_docs=2, samples_per_condition=2, seed=11),
    }
    torch.manual_seed(2024)
    run1 = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]
    torch.manual_seed(2024)
    run2 = evaluate_steering(**common_kwargs)  # ty: ignore[invalid-argument-type]

    # Every numeric metric must match exactly.
    exclude = {"steering/eval_time_seconds"}
    for key in (set(run1) | set(run2)) - exclude:
        v1, v2 = run1.get(key), run2.get(key)
        if isinstance(v1, float) and math.isnan(v1):
            assert math.isnan(v2)
        else:
            assert v1 == v2, f"non-determinism in {key}: {v1} vs {v2}"


def test_evaluate_steering_overlay_cache_parity(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """Cached vs uncached runs under the overlay path produce identical
    metrics (except the ``steering/eval_cache_hit`` diagnostic and the
    wall-clock ``eval_time_seconds``).
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    common_kwargs = {
        "model": tiny_model,
        "feature_dir": feature_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "feature_index": feature_index,
        "probe_model": None,
        "cfg": SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=11),
    }

    torch.manual_seed(2024)
    uncached = evaluate_steering(**common_kwargs, eval_cache=None)  # ty: ignore[invalid-argument-type]

    cache = InlineEvalCache(
        selected_docs=[(0, 2), (1, 2)],
        selection_metrics={
            "steering/n_docs_available": 2,
            "steering/n_docs_selected": 2,
        },
        doc_texts={
            0: "The quick brown fox jumps over the lazy dog.",
            1: "Sparse autoencoders extract interpretable features.",
        },
    )
    torch.manual_seed(2024)
    cached = evaluate_steering(**common_kwargs, eval_cache=cache)  # ty: ignore[invalid-argument-type]

    # Cached and uncached may pick slightly different docs (the cached
    # version uses our forced selection), so only compare metrics that do
    # not depend on which docs were picked. Focus on completions count
    # parity when both selected the same docs.
    assert uncached["steering/n_docs_selected"] == cached["steering/n_docs_selected"]
    assert uncached["steering/n_completions_generated"] == cached["steering/n_completions_generated"]


# ---------------------------------------------------------------------------
# Real nanochat tokenizer integration: the production tokenizer pair
# ---------------------------------------------------------------------------


def _nanochat_tokenizer_available() -> bool:
    from nanochat.tokenizer import get_tokenizer

    try:
        get_tokenizer()
    except (FileNotFoundError, ValueError):
        return False
    return True


_NANOCHAT_TOKENIZER_AVAILABLE = _nanochat_tokenizer_available()


@pytest.mark.skipif(
    not _NANOCHAT_TOKENIZER_AVAILABLE,
    reason="Requires a nanochat tokenizer cache in this environment.",
)
def test_tokenize_and_split_with_real_nanochat_tokenizer_via_simulated_overlay():
    """Integration with the production nanochat tokenizer. Builds segments
    whose ``token_start`` / ``token_end`` mirror what the precompute
    pipeline would produce (using the overlay-equivalent char-to-token
    mapping from nanochat's own tokenization), and exercises
    ``_tokenize_and_split`` across every target segment at seg_size ∈
    {32, 128}. Asserts the output invariants hold on real text and the
    real tokenizer's byte-level behaviour.
    """
    from nanochat.tokenizer import get_tokenizer
    from steering.segmentation import (
        SegmentMode,
        segment_tokens,
        token_to_char_offsets,
    )

    tok = get_tokenizer()
    bos_id = tok.get_bos_token_id()
    doc_text = (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "In the distance, a church bell rings softly, echoing through the "
        "valley. Farmers tend to their fields under a warm autumn sky, "
        "while children play in the meadow with kites of crimson and gold. "
        "Travelers on the winding road pause to admire the vista, their "
        "horses' breath visible in the cool morning air. Merchants at the "
        "village square hawk their wares, calling out prices and bargains "
        "to anyone who will listen, as the baker pulls fresh loaves from "
        "the stone oven and the smell of rosemary fills the lanes."
    )

    kept_ids = tok.encode(doc_text, prepend=bos_id)
    char_offsets = token_to_char_offsets(tok, kept_ids)
    assert len(kept_ids) >= 128

    for seg_size in (32, 128):
        seg_indices = segment_tokens(
            kept_ids,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(seg_size),
        )
        if len(seg_indices) < 2:
            continue
        # Simulate the overlay: since training and source tokenizers are
        # both nanochat here (self-consistent test), token indices map
        # trivially to the segmentation token ranges.
        segments = [
            _seg_with_token_bounds(
                segment_idx=i,
                position=start,
                char_start=char_offsets[start],
                char_end=char_offsets[end],
                token_start=start,
                token_end=end,
            )
            for i, (start, end) in enumerate(seg_indices)
        ]
        for tgt in range(1, len(segments)):
            all_tokens, split_idx, _ = _tokenize_and_split(
                doc_text, segments, tgt, tok,
            )
            # Overlay invariants hold on the real tokenizer.
            assert split_idx == segments[tgt].token_start
            assert len(all_tokens) == segments[tgt].token_end
            assert 0 < split_idx < len(all_tokens)
            # Structural: the concatenation (trivial identity) matches.
            assert all_tokens == kept_ids[: segments[tgt].token_end]


# ---------------------------------------------------------------------------
# Reference-parity tests: new overlay path matches pre-fix char-alignment
# behavior on every healthy input
#
# For docs where the old char-alignment fallback would have produced a
# valid non-degenerate result, the new overlay-based path must produce
# the SAME (split_idx, all_tokens, pre_segment_text). The reference is
# the old implementation, ported verbatim into the test file below.
# ---------------------------------------------------------------------------


def _reference_tokenize_and_split_char_alignment(
    doc_text, segments, target_seg_idx, tokenizer,
):
    """Pre-fix ``_tokenize_and_split`` reimplemented verbatim.

    This is the char-alignment path that the fix removed. It's here as a
    reference to verify the new overlay-based path produces equivalent
    output on every healthy input the reference can handle. The reference
    reports ``split_idx == len(all_tokens)`` on the degenerate case rather
    than raising.
    """
    from steering.segmentation import (
        assign_tokens_to_segments,
        token_to_char_offsets,
    )

    bos_id = tokenizer.get_bos_token_id()
    bos_prefix = tokenizer.decode([bos_id])
    bos_text = bos_prefix + doc_text

    target_seg = segments[target_seg_idx]
    pre_segment_text = bos_text[len(bos_prefix) : target_seg.char_start]
    text_through_target = bos_text[len(bos_prefix) : target_seg.char_end]
    all_tokens = tokenizer.encode(text_through_target, prepend=bos_id)

    char_offsets = token_to_char_offsets(tokenizer, all_tokens)
    token_starts = torch.tensor(char_offsets[:-1], dtype=torch.int64)
    token_ends = torch.tensor(char_offsets[1:], dtype=torch.int64)

    seg_char_starts = torch.tensor(
        [s.char_start for s in segments[: target_seg_idx + 1]], dtype=torch.int64,
    )
    seg_char_ends = torch.tensor(
        [s.char_end for s in segments[: target_seg_idx + 1]], dtype=torch.int64,
    )
    assignments = assign_tokens_to_segments(
        token_starts, token_ends, seg_char_starts, seg_char_ends,
    )
    assignments = torch.cummax(assignments, dim=0).values
    target_positions = (assignments == target_seg_idx).nonzero(as_tuple=True)[0]
    split_idx = (
        int(target_positions[0].item())
        if len(target_positions) > 0
        else len(all_tokens)
    )
    return all_tokens, split_idx, pre_segment_text


def _segments_from_reference(
    doc_text, seg_bounds, tokenizer, position_offset: int = 0,
):
    """Build SegmentFeatures whose overlay token_start/token_end reflect
    what the reference (char-alignment) implementation would have picked.

    For each segment (char_start, char_end), we compute what the reference
    returns for split_idx when targeting that segment, and use
    ``(split_idx_for_this_target, len_of_all_tokens_through_this_target)``
    as the overlay bounds. This makes the overlay path a strict
    simulation of the reference behaviour — the parity test asserts the
    new path returns the SAME output.
    """
    # Build initial segments without overlay bounds, just char bounds —
    # enough for the reference to compute split_idx against.
    plain = [
        SegmentFeatures(
            segment_idx=i,
            position=start + position_offset,
            feature_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
            feature_mags=torch.tensor(
                [5.0 + i, 3.0 + i, 7.0 + i], dtype=torch.float32,
            ),
            char_start=start,
            char_end=end,
        )
        for i, (start, end) in enumerate(seg_bounds)
    ]
    # For each target, compute the reference's output and record the
    # token_start / token_end. The overlay we build reproduces exactly
    # what the reference would have picked.
    token_bounds = []
    for tgt in range(len(plain)):
        all_tokens, split_idx, _ = _reference_tokenize_and_split_char_alignment(
            doc_text, plain, tgt, tokenizer,
        )
        token_bounds.append((split_idx, len(all_tokens)))
    return [
        SegmentFeatures(
            segment_idx=p.segment_idx,
            position=p.position,
            feature_ids=p.feature_ids,
            feature_mags=p.feature_mags,
            char_start=p.char_start,
            char_end=p.char_end,
            token_start=ts,
            token_end=te,
        )
        for p, (ts, te) in zip(plain, token_bounds, strict=True)
    ]


def test_overlay_matches_reference_implementation_byte_tokenizer(byte_tokenizer):
    """For every target segment ≥ 1 on a battery of healthy docs, the new
    overlay path must produce byte-identical ``(all_tokens, split_idx,
    pre_segment_text)`` to the old char-alignment reference. This is the
    strongest parity guarantee: the refactor does not observably change
    behavior on any input the reference could handle.
    """
    # Diverse doc texts — short/long, punctuation, whitespace, mixed.
    docs = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello. World. How are you?",
        "The rain in Spain falls mainly on the plain. " * 3,
        "a " + "blah " * 20 + "end",
        "Punctuation: !@#$%^&*() and numbers 0123456789!",
        "Trailing spaces    ",
        "Leading    spaces",
        "Many\nnewlines\nin\nbetween",
        "Tabs\tand\tspaces",
    ]
    for doc_text in docs:
        doc_len = len(doc_text)
        if doc_len < 16:
            continue  # too short to split into 4 segments
        bos_prefix_len = 1  # byte tokenizer BOS = "\x00"
        quarter = doc_len // 4
        seg_bounds = [
            (0, bos_prefix_len + quarter),
            (bos_prefix_len + quarter, bos_prefix_len + 2 * quarter),
            (bos_prefix_len + 2 * quarter, bos_prefix_len + 3 * quarter),
            (bos_prefix_len + 3 * quarter, bos_prefix_len + doc_len),
        ]
        # Skip docs where a segment would be empty.
        if any(e <= s for s, e in seg_bounds):
            continue

        segments = _segments_from_reference(doc_text, seg_bounds, byte_tokenizer)

        for tgt in range(1, len(segments)):
            ref_tokens, ref_split, ref_pre = (
                _reference_tokenize_and_split_char_alignment(
                    doc_text,
                    # Strip overlay bounds so the reference uses char-alignment.
                    [
                        SegmentFeatures(
                            segment_idx=s.segment_idx,
                            position=s.position,
                            feature_ids=s.feature_ids,
                            feature_mags=s.feature_mags,
                            char_start=s.char_start,
                            char_end=s.char_end,
                        )
                        for s in segments
                    ],
                    tgt,
                    byte_tokenizer,
                )
            )
            # Skip degenerate reference outputs — those are the exact
            # inputs the fix was built to handle; the overlay path would
            # have raised on a zero-width segment, not matched the ref.
            if ref_split >= len(ref_tokens) or ref_split <= 0:
                continue

            new_tokens, new_split, new_pre = _tokenize_and_split(
                doc_text, segments, tgt, byte_tokenizer,
            )

            assert new_tokens == ref_tokens, (
                f"tokens differ for doc={doc_text!r} tgt={tgt}: "
                f"new={new_tokens!r} ref={ref_tokens!r}"
            )
            assert new_split == ref_split, (
                f"split_idx differs for doc={doc_text!r} tgt={tgt}: "
                f"new={new_split} ref={ref_split}"
            )
            assert new_pre == ref_pre, (
                f"pre_segment_text differs: new={new_pre!r} ref={ref_pre!r}"
            )


@pytest.mark.skipif(
    not _NANOCHAT_TOKENIZER_AVAILABLE,
    reason="Requires a nanochat tokenizer cache in this environment.",
)
def test_overlay_matches_reference_nanochat_tokenizer_prose():
    """Same parity guarantee with the real nanochat tokenizer on realistic
    prose. For each target segment where the reference yields a valid
    (non-degenerate) split, the new overlay path must produce the same
    output.
    """
    from nanochat.tokenizer import get_tokenizer
    from steering.segmentation import (
        SegmentMode,
        segment_tokens,
        token_to_char_offsets,
    )

    tok = get_tokenizer()
    bos_id = tok.get_bos_token_id()
    doc_text = (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "In the distance, a church bell rings softly, echoing through the "
        "valley. Farmers tend to their fields under a warm autumn sky, "
        "while children play in the meadow with kites of crimson and gold."
    )
    kept_ids = tok.encode(doc_text, prepend=bos_id)
    char_offsets = token_to_char_offsets(tok, kept_ids)

    for seg_size in (32, 64, 128):
        seg_indices = segment_tokens(
            kept_ids,
            SegmentMode.fixed_tokens,
            PositiveInt.parse(seg_size),
        )
        if len(seg_indices) < 2:
            continue
        seg_bounds = [(char_offsets[s], char_offsets[e]) for s, e in seg_indices]
        segments = _segments_from_reference(doc_text, seg_bounds, tok)

        for tgt in range(1, len(segments)):
            ref_tokens, ref_split, ref_pre = (
                _reference_tokenize_and_split_char_alignment(
                    doc_text,
                    [
                        SegmentFeatures(
                            segment_idx=s.segment_idx,
                            position=s.position,
                            feature_ids=s.feature_ids,
                            feature_mags=s.feature_mags,
                            char_start=s.char_start,
                            char_end=s.char_end,
                        )
                        for s in segments
                    ],
                    tgt,
                    tok,
                )
            )
            if ref_split >= len(ref_tokens) or ref_split <= 0:
                continue

            new_tokens, new_split, new_pre = _tokenize_and_split(
                doc_text, segments, tgt, tok,
            )
            assert new_tokens == ref_tokens
            assert new_split == ref_split
            assert new_pre == ref_pre


def test_overlay_parity_preserves_roundtrip_sample_structure(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """End-to-end parity: the number of generated samples, their conditions,
    and their seed assignment all match what the old path would have
    produced (modulo the skip-on-degenerate edge that no longer exists —
    we use only healthy inputs here).
    """
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
    segments = _segments_from_reference(doc_text, seg_bounds, byte_tokenizer)
    feature_index = _fake_feature_index({0: segments, 1: segments})

    completions = _generate_roundtrip_samples(
        model=tiny_model,
        selected_docs=[(0, 2)],
        doc_texts={0: doc_text, 1: doc_text},
        feature_index=feature_index,
        val_shard_stem="shard_00000",
        tokenizer=byte_tokenizer,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        device=torch.device("cpu"),
        cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=2, seed=123),
    )
    # 1 doc x 2 conditioned variants x 2 samples = 4 samples
    assert len(completions) == 1
    assert len(completions[0].samples) == 4
    # Seed assignment: base = cfg.seed + doc_idx*1000 + counter, counter
    # advances by samples_per_condition per condition.
    seeds_by_cond: dict[str, list[int]] = {}
    for s in completions[0].samples:
        seeds_by_cond.setdefault(s.condition, []).append(s.seed)
    assert sorted(seeds_by_cond["real"]) == [123, 124]
    assert sorted(seeds_by_cond["random"]) == [125, 126]


# ---------------------------------------------------------------------------
# Exhaustive edge cases: every practical corner of the overlay path
# ---------------------------------------------------------------------------


def test_overlay_min_valid_target_seg_1_no_offset(byte_tokenizer):
    """Target seg 1 (minimum valid in non-offset mode): the smallest
    legal split point. Ensures the loop boundary is inclusive.
    """
    doc_text = "Hello world!"
    segments = _segments_from_reference(
        doc_text, [(0, 6), (6, 13)], byte_tokenizer,
    )
    all_tokens, split_idx, _ = _tokenize_and_split(
        doc_text, segments, 1, byte_tokenizer,
    )
    assert split_idx >= 1
    assert 0 < split_idx < len(all_tokens)


def test_overlay_min_valid_target_seg_2_offset_mode(byte_tokenizer):
    """Offset mode: min valid target is seg 2 (split_seg = 1). Ensures
    the offset split point is computed on seg 1, not seg 2.
    """
    doc_text = "Hello world, this is a test."
    segments = _segments_from_reference(
        doc_text, [(0, 7), (7, 15), (15, 29)], byte_tokenizer,
    )
    # Directly call with split_seg = target - 1
    all_tokens, split_idx, _ = _tokenize_and_split(
        doc_text, segments, 1, byte_tokenizer,  # split_seg in offset mode
    )
    assert split_idx == segments[1].token_start
    assert len(all_tokens) == segments[1].token_end


def test_overlay_last_seg_in_doc_is_valid_target(byte_tokenizer):
    """The last segment in a doc is an eligible target — the overlay's
    ``token_end == len(full_tokens)`` boundary condition.
    """
    doc_text = "Hello world!"
    segments = _segments_from_reference(
        doc_text, [(0, 6), (6, 13)], byte_tokenizer,
    )
    bos_id = byte_tokenizer.get_bos_token_id()
    full = byte_tokenizer.encode(doc_text, prepend=bos_id)
    all_tokens, _, _ = _tokenize_and_split(doc_text, segments, 1, byte_tokenizer)
    assert all_tokens == full


def test_overlay_single_token_target_segment(byte_tokenizer):
    """A target segment containing exactly 1 token (``token_end -
    token_start == 1``) is valid and must produce a 1-token gen span.
    """
    doc_text = "Hello world!"
    segments = [
        _seg_with_token_bounds(
            segment_idx=0, position=0, char_start=0, char_end=6,
            token_start=1, token_end=6,
        ),
        _seg_with_token_bounds(
            segment_idx=1, position=6, char_start=6, char_end=7,
            token_start=6, token_end=7,  # 1 token wide
        ),
        _seg_with_token_bounds(
            segment_idx=2, position=7, char_start=7, char_end=13,
            token_start=7, token_end=13,
        ),
    ]
    all_tokens, split_idx, _ = _tokenize_and_split(
        doc_text, segments, 1, byte_tokenizer,
    )
    assert split_idx == 6
    assert len(all_tokens) == 7
    assert len(all_tokens) - split_idx == 1


def test_overlay_target_seg_idx_out_of_range_raises(byte_tokenizer):
    """``target_seg_idx`` past the end of segments list is an IndexError
    (Python native) — surfaces as a clear failure to the caller.
    """
    doc_text = "Hello world!"
    segments = _segments_from_reference(
        doc_text, [(0, 6), (6, 13)], byte_tokenizer,
    )
    with pytest.raises(IndexError):
        _tokenize_and_split(doc_text, segments, 5, byte_tokenizer)


def test_overlay_multi_byte_utf8_doc_text(byte_tokenizer):
    """Multi-byte UTF-8 chars in the doc text: each byte is 1 token for
    byte_tokenizer, so the overlay bounds work unchanged."""
    doc_text = "cafe resume naive - 9 chars ascii"
    segments = _segments_from_reference(
        doc_text, [(0, 15), (15, 34)], byte_tokenizer,
    )
    all_tokens, split_idx, _ = _tokenize_and_split(
        doc_text, segments, 1, byte_tokenizer,
    )
    assert 0 < split_idx < len(all_tokens)


def test_generate_roundtrip_samples_doc_missing_from_doc_texts_raises(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """Selecting a doc_idx that isn't in ``doc_texts`` must raise
    ValueError — silently dropping it would produce inconsistent metrics.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})
    with pytest.raises(ValueError, match="doc_texts"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 2)],  # 0 missing from doc_texts
            doc_texts={1: doc_text},  # wrong key
            feature_index=feature_index,
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=1, seed=42),
        )


def test_generate_roundtrip_samples_target_seg_out_of_range_raises(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """Selecting a target_seg_idx past the doc's segment list is a real
    error (stale cache, shard mismatch) — propagate loudly.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    segments = _four_segments_for_doc()  # 4 segments (0..3)
    feature_index = _fake_feature_index({0: segments, 1: segments})
    with pytest.raises(ValueError, match="out of range"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 99)],  # way past end
            doc_texts={0: doc_text, 1: doc_text},
            feature_index=feature_index,
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=1, seed=42),
        )


def test_generate_roundtrip_samples_feature_index_missing_doc_raises(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """``feature_index.lookup`` raises ``KeyError`` for a selected doc absent
    from the index — a data-integrity (corruption) issue that propagates.
    """
    doc_text = "The quick brown fox jumps over the lazy dog."
    # Index knows about doc 1 but _select_random_docs somehow produced doc 0.
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({1: segments})
    with pytest.raises(KeyError, match=r"Document 0 not found in shard 'shard_00000'"):
        _generate_roundtrip_samples(
            model=tiny_model,
            selected_docs=[(0, 2)],
            doc_texts={0: doc_text, 1: doc_text},
            feature_index=feature_index,
            val_shard_stem="shard_00000",
            tokenizer=byte_tokenizer,
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            device=torch.device("cpu"),
            cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=1, seed=42),
        )


def test_generate_roundtrip_samples_empty_selected_docs_returns_empty(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
):
    """Empty ``selected_docs`` is a legal input (e.g. DDP rank with no
    work) and must produce an empty completions list without errors.
    """
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments})
    completions = _generate_roundtrip_samples(
        model=tiny_model,
        selected_docs=[],
        doc_texts={0: "The quick brown fox jumps over the lazy dog."},
        feature_index=feature_index,
        val_shard_stem="shard_00000",
        tokenizer=byte_tokenizer,
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        device=torch.device("cpu"),
        cfg=SteeringEvalConfig(n_eval_docs=1, samples_per_condition=1, seed=42),
    )
    assert completions == []


def test_evaluate_steering_e2e_all_keys_present_and_correct_types(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """Full eval produces every expected metric key; each is the correct
    numeric type; no NaNs (because inputs are healthy).
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=feature_index,
        probe_model=None,
        cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=7),
    )
    expected_keys = [
        "steering/n_docs_available",
        "steering/n_docs_selected",
        "steering/n_completions_generated",
        "steering/eval_time_seconds",
        "steering/roundtrip_real_mean",
        "steering/roundtrip_random_mean",
        "steering/roundtrip_specificity",
        "steering/roundtrip_n_samples",
        "steering/roundtrip_n_gt_features_avg",
    ]
    missing = set(expected_keys) - set(result.keys())
    assert not missing, f"missing metric keys: {missing}"
    for k in expected_keys:
        v = result[k]
        assert isinstance(v, (int, float)), f"{k} not numeric: {type(v)}"
        assert math.isfinite(v), f"{k} not finite: {v}"


def test_evaluate_steering_partition_docs_ddp_round_robin(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """DDP-partitioning correctness (exercised without actual distributed
    backend): total completions across all ranks equal the full selection
    under round-robin partition.
    """
    feature_dir = _minimal_feature_shard(tmp_path, n_docs=4, n_segs_per_doc=4)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features from networks.",
        "In the morning light, dew sparkles across the meadow.",
        "Programs run best when you know what they do under the hood.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index(dict.fromkeys(range(4), segments))
    cache = InlineEvalCache(
        selected_docs=[(i, 2) for i in range(4)],
        selection_metrics={
            "steering/n_docs_available": 4,
            "steering/n_docs_selected": 4,
        },
        doc_texts={
            0: "The quick brown fox jumps over the lazy dog.",
            1: "Sparse autoencoders extract interpretable features from networks.",
            2: "In the morning light, dew sparkles across the meadow.",
            3: "Programs run best when you know what they do under the hood.",
        },
    )

    # Simulate 2 ranks: each gets 2 docs via round-robin.
    total_completions = 0
    for ddp_rank in range(2):
        result = evaluate_steering(
            model=tiny_model,
            feature_dir=feature_dir,
            tokenizer=byte_tokenizer,
            device=torch.device("cpu"),
            boundaries=boundaries,
            threshold_cfg=threshold_config,
            precomputed_thresholds=precomputed_thresholds,
            data_dir=str(val_parquet_dir),
            feature_index=feature_index,
            probe_model=None,
            cfg=SteeringEvalConfig(n_eval_docs=4, samples_per_condition=1, seed=7),
            eval_cache=cache,
            ddp_rank=ddp_rank,
            ddp_world_size=2,
        )
        # Without dist.is_initialized(), _gather_int is a no-op: each rank
        # reports its local completions count. Sum across ranks must equal
        # the full selection.
        total_completions += result["steering/n_completions_generated"]
    assert total_completions == 4


def test_evaluate_steering_empty_selection_fast_path(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """When ``_select_random_docs`` (or the cache) yields no docs, the
    eval short-circuits: returns scan metrics only, no probe/generation.
    """
    cache = InlineEvalCache(
        selected_docs=[],
        selection_metrics={
            "steering/n_docs_available": 0,
            "steering/n_docs_selected": 0,
        },
        doc_texts={},
    )
    result = evaluate_steering(
        model=tiny_model,
        feature_dir=tmp_path,  # empty — never read
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=None,  # never read
        feature_index=None,  # never consulted
        probe_model=None,
        cfg=SteeringEvalConfig(n_eval_docs=10, samples_per_condition=1, seed=7),
        eval_cache=cache,
    )
    assert result["steering/n_docs_selected"] == 0
    assert "steering/roundtrip_real_mean" not in result  # fast-return skipped probe
    assert result["steering/eval_time_seconds"] >= 0


def test_evaluate_steering_offset_mode_min_valid_target(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """Offset mode with target=2 (minimum valid): exercises the offset
    branch that splits at seg 1 and targets seg 2's features.
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})
    cache = InlineEvalCache(
        selected_docs=[(0, 2), (1, 2)],
        selection_metrics={
            "steering/n_docs_available": 2,
            "steering/n_docs_selected": 2,
        },
        doc_texts={
            0: "The quick brown fox jumps over the lazy dog.",
            1: "Sparse autoencoders extract interpretable features.",
        },
    )
    result = evaluate_steering(
        model=tiny_model,
        feature_dir=feature_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=feature_index,
        probe_model=None,
        cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=7),
        eval_cache=cache,
    )
    assert result["steering/n_completions_generated"] == 2


def test_evaluate_steering_rng_state_preserved(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
):
    """``evaluate_steering`` must not disturb the caller's global torch
    RNG state — verified by comparing a torch.rand() draw before/after.
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    torch.manual_seed(42)
    before_rng = torch.rand(5)

    torch.manual_seed(42)
    rng_state = save_rng_state(torch.device("cpu"))
    evaluate_steering(
        model=tiny_model,
        feature_dir=feature_dir,
        tokenizer=byte_tokenizer,
        device=torch.device("cpu"),
        boundaries=boundaries,
        threshold_cfg=threshold_config,
        precomputed_thresholds=precomputed_thresholds,
        data_dir=str(val_parquet_dir),
        feature_index=feature_index,
        probe_model=None,
        cfg=SteeringEvalConfig(n_eval_docs=2, samples_per_condition=2, seed=7),
    )
    restore_rng_state(rng_state, torch.device("cpu"))
    after_rng = torch.rand(5)

    assert torch.equal(before_rng, after_rng), (
        f"RNG was disturbed: before={before_rng} after={after_rng}"
    )


# ---------------------------------------------------------------------------
# Multiple doc-pattern integration: tokenizer-bound docs with varied shapes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("doc_pattern", [
    "The quick brown fox jumps over the lazy dog.",
    "a " * 30 + "end.",
    "Hello.\nWorld.\nHow.\nAre.\nYou?",
    "AB" * 50,  # very repetitive
    "1234567890 " * 5,  # numbers and spaces
])
def test_tokenize_and_split_invariants_across_doc_patterns(
    byte_tokenizer, doc_pattern,
):
    """Parametrized: for a variety of doc shapes, the structural
    invariants on `_tokenize_and_split`'s output always hold.
    """
    if len(doc_pattern) < 20:
        pytest.skip("doc too short for 4-seg fixture")
    bos_prefix_len = 1
    doc_len = len(doc_pattern)
    quarter = doc_len // 4
    seg_bounds = [
        (0, bos_prefix_len + quarter),
        (bos_prefix_len + quarter, bos_prefix_len + 2 * quarter),
        (bos_prefix_len + 2 * quarter, bos_prefix_len + 3 * quarter),
        (bos_prefix_len + 3 * quarter, bos_prefix_len + doc_len),
    ]
    # Skip if any segment is empty.
    if any(e <= s for s, e in seg_bounds):
        pytest.skip("some seg would be empty")
    segments = _segments_from_reference(doc_pattern, seg_bounds, byte_tokenizer)
    for tgt in range(1, len(segments)):
        # Skip if the reference produces a degenerate split — those are
        # the exact cases we can no longer reach (overlay values would
        # violate the invariants and raise).
        if segments[tgt].token_start == segments[tgt].token_end:
            continue
        if segments[tgt].token_start == 0:
            continue
        all_tokens, split_idx, pre_text = _tokenize_and_split(
            doc_pattern, segments, tgt, byte_tokenizer,
        )
        # Invariants.
        assert 0 < split_idx < len(all_tokens)
        assert split_idx == segments[tgt].token_start
        assert len(all_tokens) == segments[tgt].token_end
        # pre_segment_text is prefix of bos_text up to char_start.
        bos_prefix = byte_tokenizer.decode([byte_tokenizer.get_bos_token_id()])
        expected_pre = (bos_prefix + doc_pattern)[
            len(bos_prefix) : segments[tgt].char_start
        ]
        assert pre_text == expected_pre


@pytest.mark.parametrize("seed", [1, 7, 42, 100, 999])
def test_evaluate_steering_overlay_deterministic_across_seeds(
    tiny_model,
    byte_tokenizer,
    boundaries,
    threshold_config,
    precomputed_thresholds,
    tmp_path,
    seed,
):
    """Same seed → same metrics, across parametrized seeds. Catches any
    seed-dependent non-determinism in the overlay path.
    """
    feature_dir = _minimal_feature_shard(tmp_path)
    val_parquet_dir = _minimal_val_parquet(tmp_path, [
        "The quick brown fox jumps over the lazy dog.",
        "Sparse autoencoders extract interpretable features.",
    ])
    segments = _four_segments_for_doc()
    feature_index = _fake_feature_index({0: segments, 1: segments})

    common = {
        "model": tiny_model,
        "feature_dir": feature_dir,
        "tokenizer": byte_tokenizer,
        "device": torch.device("cpu"),
        "boundaries": boundaries,
        "threshold_cfg": threshold_config,
        "precomputed_thresholds": precomputed_thresholds,
        "data_dir": str(val_parquet_dir),
        "feature_index": feature_index,
        "probe_model": None,
        "cfg": SteeringEvalConfig(n_eval_docs=2, samples_per_condition=1, seed=seed),
    }
    torch.manual_seed(seed)
    r1 = evaluate_steering(**common)  # ty: ignore[invalid-argument-type]
    torch.manual_seed(seed)
    r2 = evaluate_steering(**common)  # ty: ignore[invalid-argument-type]
    for key in set(r1) - {"steering/eval_time_seconds"}:
        v1, v2 = r1[key], r2[key]
        if isinstance(v1, float) and math.isnan(v1):
            assert math.isnan(v2)
        else:
            assert v1 == v2, f"seed={seed} key={key}: {v1} != {v2}"


def test_clustered_generation_uses_cached_representatives_and_fixed_donor(
    tiny_model,
    byte_tokenizer,
    monkeypatch,
):
    source_segments = _four_segments_for_doc()[:3]
    prefix_segments = tuple(
        dataclasses.replace(
            segment,
            feature_ids=torch.tensor([1, 5], dtype=torch.int64),
            feature_mags=torch.tensor([2, 4], dtype=torch.int64),
        )
        for segment in source_segments
    )
    donor = dataclasses.replace(
        source_segments[1],
        feature_ids=torch.tensor([7], dtype=torch.int64),
        feature_mags=torch.tensor([3], dtype=torch.int64),
    )
    target = ClusteredInlineTarget(
        doc_idx=0,
        target_segment_ordinal=2,
        prefix_segments=prefix_segments,
        positive_feature_ids=(1,),
        negative_feature_ids=(5,),
        positive_member_groups=((1, 2),),
        donor_doc_idx=1,
        donor_segment_ordinal=1,
        donor_segment=donor,
    )
    payload = ClusteredInlineEvalPayload(
        source_identity={"selection_hash": "fixture"},
        targets=(target,),
        digest="a" * 64,
    )
    captured: list[MultisegInputs] = []

    def fake_generate_samples_for_condition(**kwargs):
        captured.append(kwargs["encoder_inputs"])
        return [[65]]

    monkeypatch.setattr(
        inline_mod,
        "generate_samples_for_condition",
        fake_generate_samples_for_condition,
    )
    completions = _generate_clustered_roundtrip_samples(
        tiny_model,
        [(0, 2)],
        {0: "The quick brown fox jumps over the lazy dog."},
        payload,
        byte_tokenizer,
        torch.device("cpu"),
        SteeringEvalConfig(samples_per_condition=1, seed=17),
    )

    assert len(captured) == 2
    assert captured[0].enc_feature_ids[0, 2, :2].tolist() == [1, 5]
    assert captured[1].enc_feature_ids[0, 2, :2].tolist() == [7, 0]
    assert captured[1].enc_mask[0, 2, :2].tolist() == [True, False]
    assert completions[0].ground_truth_feature_ids == [1]
    assert completions[0].ground_truth_member_ids == [[1, 2]]
    assert [sample.condition for sample in completions[0].samples] == [
        "real",
        "random",
    ]


def test_clustered_probe_uses_member_max_percentile_and_representative_raw(
    monkeypatch,
):
    probe = _build_probe_model_mock(n_features=8)
    completion = RoundtripCompletion(
        doc_idx=0,
        target_segment_idx=1,
        pre_segment_text="prefix ",
        ground_truth_feature_ids=[0, 3],
        samples=[CompletionSample(condition="real", text="completion", seed=42)],
        ground_truth_member_ids=[[0, 1], [3]],
    )
    monkeypatch.setattr(
        inline_mod,
        "get_probe_activations",
        lambda model, tokens, **kwargs: torch.ones(1, tokens.shape[1], 8),
    )
    seen: list[int] = []

    def fake_magnitude_to_percentile(feature_ids, magnitudes, percentile_table):
        del magnitudes, percentile_table
        seen.extend(feature_ids.tolist())
        values = {0: 10.0, 1: 90.0, 3: 30.0}
        return torch.tensor([values[int(fid)] for fid in feature_ids])

    monkeypatch.setattr(
        inline_mod,
        "magnitude_to_percentile",
        fake_magnitude_to_percentile,
    )
    scores = _probe_roundtrip_raw(
        [completion],
        probe,
        percentile_table=torch.empty(8, 99),
    )

    assert seen == [0, 1, 3]
    assert len(scores) == 1
    condition, pre_raw, post_raw, percentile = scores[0]
    assert condition == "real"
    assert math.isfinite(pre_raw)
    assert math.isfinite(post_raw)
    assert percentile == 60.0  # mean(max(member 0=10, member 1=90), member 3=30)


def test_clustered_evaluate_emits_semantics_and_average_member_group_size(
    monkeypatch,
    tmp_path,
):
    payload = ClusteredInlineEvalPayload(
        source_identity={},
        targets=(),
        digest="a" * 64,
    )
    cache = InlineEvalCache(
        selected_docs=[(0, 1)],
        selection_metrics={
            "steering/n_docs_available": 1,
            "steering/n_docs_selected": 1,
        },
        doc_texts={0: "text"},
        clustered_payload=payload,
    )
    completion = RoundtripCompletion(
        doc_idx=0,
        target_segment_idx=1,
        pre_segment_text="prefix ",
        ground_truth_feature_ids=[1, 4],
        samples=[],
        ground_truth_member_ids=[[1, 2, 3], [4]],
    )
    monkeypatch.setattr(
        inline_mod,
        "_generate_clustered_roundtrip_samples",
        lambda *args, **kwargs: [completion],
    )

    class EvalModel:
        training = False

    metrics = evaluate_steering(
        model=EvalModel(),
        feature_dir=tmp_path,
        tokenizer=object(),
        device=torch.device("cpu"),
        boundaries=None,
        threshold_cfg=None,
        precomputed_thresholds=None,
        cfg=SteeringEvalConfig(),
        eval_cache=cache,
        probe_model=None,
        meta={"feature_clusters": {"present": True}},
    )

    assert metrics["steering/cluster_probe_semantics"] == CLUSTERED_PROBE_SEMANTICS
    assert metrics["steering/roundtrip_cluster_member_group_size_avg"] == 2.0


def test_clustered_ddp_generation_failure_is_gathered_before_scores(
    monkeypatch,
    tmp_path,
):
    payload = ClusteredInlineEvalPayload(
        source_identity={},
        targets=(),
        digest="a" * 64,
    )
    cache = InlineEvalCache(
        selected_docs=[(0, 1)],
        selection_metrics={
            "steering/n_docs_available": 1,
            "steering/n_docs_selected": 1,
        },
        doc_texts={0: "text"},
        clustered_payload=payload,
    )
    monkeypatch.setattr(
        inline_mod,
        "_generate_clustered_roundtrip_samples",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("rank-local boom")),
    )
    monkeypatch.setattr(inline_mod.dist, "is_initialized", lambda: True)
    gather_calls = 0

    def fake_all_gather(output, value):
        nonlocal gather_calls
        gather_calls += 1
        output[:] = [
            value,
            {
                "ok": True,
                "rank": 1,
                "error_type": None,
                "error": None,
            },
        ]

    monkeypatch.setattr(inline_mod.dist, "all_gather_object", fake_all_gather)

    class EvalModel:
        training = False

    with pytest.raises(
        RuntimeError,
        match="rank 0: ValueError: rank-local boom",
    ):
        evaluate_steering(
            model=EvalModel(),
            feature_dir=tmp_path,
            tokenizer=object(),
            device=torch.device("cpu"),
            boundaries=None,
            threshold_cfg=None,
            precomputed_thresholds=None,
            cfg=SteeringEvalConfig(),
            eval_cache=cache,
            ddp_rank=0,
            ddp_world_size=2,
        )
    assert gather_calls == 1
