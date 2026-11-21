"""End-to-end integration for full-sequence conditioned SFT — WITHOUT patching.

Every existing ``make_sft_data_generator`` test monkey-patches
``load_sft_features`` to inject an in-memory ``{idx: [SegmentFeatures]}`` dict
(see ``tests/steering/test_encdec_sft_conditioned_integration.py``).  That
bypasses the on-disk extractor↔router coordinate contract: the parquet
``original_index`` mapping, the feature-shard char ranges, and the coverage-gap
preflight are never exercised together.

These two tests close that gap.  They build *real* feature shards
(``steering.feature_io.save_features`` — the extractor's format) plus real
parquet shards with an ``original_index`` column, then run the UNPATCHED
``load_sft_features`` -> ``make_sft_data_generator`` -> tiny ``EncoderDecoderGPT``
forward path:

(a) the routed batch produces a finite loss AND the encoder actually received
    non-empty routed segments (``enc_mask`` has True entries, ``seg_map`` is
    non-trivial);
(b) the routed-feature conditioning changes the loss versus an all-zero
    ``seg_map`` + empty ``enc_mask`` — catching a regression where SFT routing
    silently degrades to a no-op while the loss stays finite/positive.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_sft import SFTState, make_sft_data_generator
from steering.conditional.encoder_decoder.full_sequence_routing import (
    source_aligned_char_offsets,
)
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.feature_io import save_features
from steering.sft.mixture_fingerprint import (
    MIXTURE_FINGERPRINT_KEY,
    SFT_MIXTURE_FINGERPRINT_METADATA_KEY,
    compute_mixture_fingerprint,
    mixture_fingerprint_json,
)
from steering.sft_schema import SFTSplit
from tasks.common import Task
from tests.conftest import (
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    MockTokenizer,
    _make_encdec,
)

# ---------------------------------------------------------------------------
# Force SDPA: flex_attention lacks a CPU backward and trips Inductor on CPU.
# Mirrors tests/steering/test_encdec_sft_conditioned_integration.py.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Real tiny task + on-disk shard/parquet builder
# ---------------------------------------------------------------------------


class _TinyTask(Task):
    """Deterministic single-turn task; rendered conversations stay short."""

    def __init__(self, n: int, **kw) -> None:
        super().__init__(**kw)
        self._n = n

    def num_examples(self) -> int:
        return self._n

    def get_example(self, index: int) -> dict:
        return {
            "messages": [
                {"role": "user", "content": f"q{index}xx"},
                {"role": "assistant", "content": f"ans{index}yy"},
            ]
        }


def _build_real_sft_artifacts(
    tmp_path: Path,
    task: _TinyTask,
    tokenizer: MockTokenizer,
    *,
    feats_per_seg: int = 3,
) -> tuple[str, str]:
    """Write a real feature shard + parquet covering every task example.

    Each example gets one segment spanning ``[0, total_source_chars)`` where the
    char count is computed with ``source_aligned_char_offsets`` exactly as the
    SFT generator's coverage-gap preflight does — so the strict (<= 20 char)
    coverage check passes for real instead of being bypassed by a patch.

    The parquet ``original_index`` column maps local doc i -> task index i, and
    the feature shard's ``doc_ids`` are the same i, so the unpatched
    ``load_sft_features`` indexes by mixture index correctly.

    Returns ``(feature_dir, parquet_dir)`` as strings.
    """
    feature_dir = tmp_path / "features"
    parquet_dir = tmp_path / "parquet"
    feature_dir.mkdir()
    parquet_dir.mkdir()

    n = len(task)
    fingerprint = compute_mixture_fingerprint(
        task,
        mixture_name="full-sequence-test",
        split=SFTSplit.TRAIN,
    )
    doc_ids: list[int] = []
    seg_char_starts: list[int] = []
    seg_char_ends: list[int] = []
    offsets: list[int] = [0]
    feature_ids: list[int] = []
    feature_mags: list[float] = []
    for idx in range(n):
        ids, _mask = tokenizer.render_conversation(task[idx])
        total_chars = source_aligned_char_offsets(tokenizer, ids)[-1]
        assert total_chars > 0
        doc_ids.append(idx)
        seg_char_starts.append(0)
        seg_char_ends.append(total_chars)
        # Distinct, non-zero feature ids per example so conditioning varies.
        fids = [
            1 + (idx * feats_per_seg + j) % (TINY_N_SAE_FEATURES - 1) for j in range(feats_per_seg)
        ]
        feature_ids.extend(fids)
        feature_mags.extend(float(feats_per_seg - j) for j in range(feats_per_seg))
        offsets.append(len(feature_ids))

    shard = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.tensor(feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(feature_mags, dtype=t.float32),
        "doc_ids": t.tensor(doc_ids, dtype=t.int32),
        "segment_indices": t.tensor([0] * n, dtype=t.int32),
        "segment_positions": t.tensor([0] * n, dtype=t.int32),
        "doc_token_counts": t.tensor([256] * n, dtype=t.int32),
        "segment_char_starts": t.tensor(seg_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(seg_char_ends, dtype=t.int32),
        "n_docs": n,
        "n_segments": n,
        "layer": 12,
        "context_window": 1024,
        "pooling": "mean_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        SFT_MIXTURE_FINGERPRINT_METADATA_KEY: mixture_fingerprint_json(
            fingerprint
        ),
    }
    save_features(shard, feature_dir / "shard_00000_features")

    table = pa.table(
        {
            "original_index": pa.array(list(range(n)), type=pa.int64()),
            "text": pa.array([f"doc-{i}" for i in range(n)]),
        }
    )
    pq.write_table(table, parquet_dir / "shard_00000.parquet")
    (parquet_dir / "conversion_meta.json").write_text(
        json.dumps(
            {
                "mixture": "full-sequence-test",
                "split": "train",
                "format": "full_nanochat_text",
                "completed_shards": 1,
                "total_shards": 1,
                "total_conversations": n,
                MIXTURE_FINGERPRINT_KEY: fingerprint,
            }
        ),
        encoding="utf-8",
    )
    return str(feature_dir), str(parquet_dir)


def _make_real_generator(
    tmp_path: Path,
    task: _TinyTask,
    tokenizer: MockTokenizer,
    *,
    B: int = 2,
    T: int = 64,
):
    """Build a conditioned generator backed by real on-disk artifacts."""
    feature_dir, parquet_dir = _build_real_sft_artifacts(tmp_path, task, tokenizer)
    boundaries = (
        t.linspace(0.5, float(TINY_NUM_BUCKETS) - 0.5, TINY_NUM_BUCKETS - 1)
        .unsqueeze(0)
        .expand(TINY_N_SAE_FEATURES, -1)
        .contiguous()
    )
    return make_sft_data_generator(
        task=task,
        tokenizer=tokenizer,
        B=B,
        T=T,
        device="cpu",
        sft_state=SFTState(),
        feature_dir=feature_dir,
        parquet_dir=parquet_dir,
        threshold_cfg=ThresholdConfig(threshold_type=ThresholdType.segment_topk, threshold=1000),
        boundaries=boundaries,
        min_loss_tokens=1,
        alignment_probe_samples=0,
        mixture_name="full-sequence-test",
        expected_split=SFTSplit.TRAIN,
    )


def _forward_loss(model, batch, *, enc_mask=None, seg_map=None) -> float:
    """Run a forward pass and return the scalar loss.

    Optional ``enc_mask`` / ``seg_map`` overrides let a caller feed the same
    tokens/features through with neutralised conditioning.
    """
    inputs, targets, enc_fids, enc_mags, batch_enc_mask, batch_seg_map, _fp = batch
    use_mask = batch_enc_mask if enc_mask is None else enc_mask
    use_seg = batch_seg_map if seg_map is None else seg_map
    with t.no_grad():
        result = model(inputs, targets, enc_fids, enc_mags, use_mask, use_seg)
    loss = result.loss if hasattr(result, "loss") else result
    return float(loss.item())


# ---------------------------------------------------------------------------
# (a) real load -> route -> forward roundtrip
# ---------------------------------------------------------------------------


class TestRealLoadRouteForward:
    """Unpatched load -> route -> tiny-model forward yields a finite loss and
    non-empty routed conditioning."""

    def test_roundtrip_finite_loss_and_nonempty_routing(self, tmp_path: Path) -> None:
        tokenizer = MockTokenizer()
        task = _TinyTask(n=6)
        gen = _make_real_generator(tmp_path, task, tokenizer, B=2, T=64)

        batch = next(gen)
        inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp = batch

        # Shapes line up with the requested (B, T).
        assert inputs.shape == (2, 64)
        assert targets.shape == (2, 64)
        assert seg_map.shape == (2, 64)
        assert enc_fids.shape == enc_mask.shape == enc_mags.shape
        assert enc_fids.shape[0] == 2

        # The router actually placed features: non-empty encoder mask, real
        # (non-zero) feature ids, and a non-trivial position->segment map.
        assert bool(enc_mask.any()), "encoder mask must have at least one True entry"
        assert bool((enc_fids > 0).any()), "encoder must receive real feature ids"
        assert bool((seg_map > 0).any()), "seg_map must route some positions"

        model = _make_encdec(device="cpu")
        model.eval()
        loss = _forward_loss(model, batch)
        assert math.isfinite(loss), f"forward loss must be finite, got {loss}"
        assert loss > 0.0, f"forward loss should be positive at init, got {loss}"

    def test_load_sft_features_is_not_patched(self, tmp_path: Path) -> None:
        """Sanity guard: the module-level ``load_sft_features`` reference used by
        the SFT script is the genuine function, not a test double.  If a future
        edit accidentally patches it module-wide, this test fails loudly so the
        'real on-disk' guarantee of this file is not silently lost."""
        import scripts.train.conditional_encdec_sft as sft_mod
        from steering.conditional.encoder_decoder import sft_features as real_mod

        assert sft_mod.load_sft_features is real_mod.load_sft_features
        # And it is a plain function (unittest.mock would swap in a MagicMock).
        assert callable(sft_mod.load_sft_features)
        assert type(sft_mod.load_sft_features).__name__ == "function"


# ---------------------------------------------------------------------------
# (b) conditioning changes the loss
# ---------------------------------------------------------------------------


class TestConditioningChangesLoss:
    """The routed-feature conditioning must measurably change the loss versus a
    neutralised (all-zero seg_map + empty enc_mask) pass over identical tokens.

    A no-op regression (routing silently produces empty conditioning) would
    leave the loss finite and positive but identical to the neutralised pass —
    exactly what this test catches.
    """

    def test_routed_vs_empty_conditioning_differ(self, tmp_path: Path) -> None:
        tokenizer = MockTokenizer()
        task = _TinyTask(n=6)
        gen = _make_real_generator(tmp_path, task, tokenizer, B=2, T=64)
        batch = next(gen)
        _inp, _tgt, _fids, _mags, enc_mask, seg_map, _fp = batch

        # Precondition: the real batch is genuinely conditioned.
        assert bool(enc_mask.any())
        assert bool((seg_map > 0).any())

        model = _make_encdec(device="cpu")
        model.eval()

        routed_loss = _forward_loss(model, batch)
        empty_loss = _forward_loss(
            model,
            batch,
            enc_mask=t.zeros_like(enc_mask),
            seg_map=t.zeros_like(seg_map),
        )

        assert math.isfinite(routed_loss) and math.isfinite(empty_loss)
        assert routed_loss != empty_loss, (
            "Routed-feature conditioning produced the SAME loss as empty "
            f"conditioning ({routed_loss}); SFT routing may be a silent no-op."
        )

    def test_conditioning_effect_is_deterministic(self, tmp_path: Path) -> None:
        """Two forward passes of the routed batch give the identical loss
        (eval mode, no dropout), so the (a)/(b) difference above is a real
        conditioning effect rather than nondeterminism."""
        tokenizer = MockTokenizer()
        task = _TinyTask(n=6)
        gen = _make_real_generator(tmp_path, task, tokenizer, B=2, T=64)
        batch = next(gen)

        model = _make_encdec(device="cpu")
        model.eval()
        loss_a = _forward_loss(model, batch)
        loss_b = _forward_loss(model, batch)
        assert loss_a == loss_b, f"eval forward must be deterministic: {loss_a} vs {loss_b}"
