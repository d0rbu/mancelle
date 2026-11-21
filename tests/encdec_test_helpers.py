"""Shared helpers for encoder-decoder integration, E2E, and AxBench tests.

Provides model building, precomputed dataloader creation, and training step
utilities used by test_encdec_integration.py, test_encdec_e2e.py,
and test_encdec_performance.py, plus the one ``MethodSetup``-shaped double
every AxBench-upstream eval suite builds its setups from
(tests/scripts/eval/test_axbench_*.py).
"""

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from nanochat.gpt import GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.dataloader import (
    precomputed_data_loader,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.encoder_decoder.sft_selection import SegmentSelectionContext
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    ThresholdSelectionConfig,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    SamplingBundle,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval.axbench_upstream.methods import (
    PROMPT_CONTEXT_WINDOW,
    PROMPT_SEGMENT_SIZE,
    _build_turn_prompt_selection_context,
)
from steering.packed_batch_io import PackedRowShard, SelectionShard, save_packed_row_shard
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode

# ---------------------------------------------------------------------------
# AxBench-upstream MethodSetup double
# ---------------------------------------------------------------------------

TURN_FIELD_DEFAULTS: dict[str, Any] = {
    "segment_mode": SegmentMode.fixed_tokens,
    "prompt_segment_size": PROMPT_SEGMENT_SIZE,
    "prompt_context_window": PROMPT_CONTEXT_WINDOW,
    "turn_self_cond_segment_size_rustbpe": None,
    "prompt_selection_context": None,
}


TURN_SELECTION_SEED = 4242


def turn_prompt_selection_context(
    *,
    seed: int = TURN_SELECTION_SEED,
    policy: str = "lowest",
) -> SegmentSelectionContext:
    """The REAL turn prompt selector the AxBench driver doubles carry.

    Built through the production ``_build_turn_prompt_selection_context`` so the
    policy and seed a manifest publishes are the ones the selector would
    actually condition prompts with, not a restated pair of literals.
    """

    rng = torch.Generator(device="cpu").manual_seed(seed)
    pipeline = FeaturePipelineConfig(
        selection_config=SamplingBundle(
            selection=SamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0,
                std=2.0,
                delta_weights=None,
                min_k=1,
                max_k=4,
                mode=SamplingMode.top_k_percentile,
            ),
            percentile_table=torch.zeros((4, 99), dtype=torch.float64),
            rng=rng,
        ),
        boundaries=torch.zeros((4, 0), dtype=torch.float64),
        negative_bundle=None,
        effective_buckets=1,
        feature_mask=None,
    )
    return _build_turn_prompt_selection_context(
        pipeline=pipeline,
        missing_percentile_policy=policy,
        consolidation=None,
    )


# The exact turn geometry block both AxBench drivers must publish for the
# turn setup the driver suites stage. Hand-written on purpose: deriving it from
# `_turn_self_conditioning_segment_geometry` would make both manifest tests
# tautological.
EXPECTED_TURN_GEOMETRY: dict[str, Any] = {
    "prompt_segment_mode": "turn_delimited_fixed_tokens",
    "prompt_sae_model": "gemma-3-4b-pt",
    "prompt_sae_layer": 22,
    "prompt_segment_size_gemma_tokens": 17,
    "prompt_context_window_gemma_tokens": 333,
    "generated_cadence_rustbpe_tokens": 37,
    "prompt_missing_percentile_policy": "lowest",
    "prompt_selection_seed": TURN_SELECTION_SEED,
    "prompt_statistics_raw_sha256": "a" * 64,
}


def method_setup_double(**overrides: Any) -> SimpleNamespace:
    """A ``MethodSetup``-shaped double that always carries the five turn fields.

    ``MethodSetup`` declares them without defaults precisely so a partially
    built setup cannot masquerade as a fixed-token checkpoint. Stating that
    shape once here is what keeps every AxBench double honest instead of
    letting one of them drift into relying on an absent attribute.
    """

    return SimpleNamespace(
        **{**TURN_FIELD_DEFAULTS, "prompt_statistics_raw_sha256": "a" * 64, **overrides}
    )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VOCAB_SIZE = 100
N_EMBD = 64
N_LAYER = 4
N_HEAD = 2
N_KV_HEAD = 2
HEAD_DIM = 32
SEQ_LEN = 64
SEGMENT_SIZE = 16
NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)
N_SAE_FEATURES: PositiveInt = PositiveInt.parse(200)


# ---------------------------------------------------------------------------
# Pipeline helpers
# ---------------------------------------------------------------------------


def build_threshold_pipeline(boundaries, num_buckets=NUM_BUCKETS, threshold=3):
    """Build a FeaturePipelineConfig with threshold-based selection."""
    return FeaturePipelineConfig(
        selection_config=ThresholdSelectionConfig(
            selection=ThresholdSelection(ThresholdType.segment_topk, threshold),
            precomputed_thresholds=None,
        ),
        boundaries=boundaries,
        negative_bundle=None,
        effective_buckets=num_buckets,
    )


# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------


def build_tiny_model(
    n_sae_features=N_SAE_FEATURES,
    num_buckets=NUM_BUCKETS,
    feature_pred_config=None,
    n_layer=N_LAYER,
    n_embd=N_EMBD,
    seq_len=SEQ_LEN,
    xattn_frequency=2,
    encoder_depth=1,
    feature_vocab: FeatureVocab | None = None,
    vocab_size: int = VOCAB_SIZE,
):
    """Build a tiny EncoderDecoderGPT for testing.

    ``encoder_depth`` sizes the FEATURE ENCODER's block stack (default 1 keeps
    every existing caller byte-identical). Pass ``>= 3`` to obtain more than one
    ``has_ve`` block — required for gradient tests that must pin the STE value-
    embedding term across EVERY value-embedding layer, not just a single one.

    ``vocab_size`` defaults to this module's tiny ``VOCAB_SIZE``; pass the real
    tokenizer's size when the batch carries real token ids.
    """
    dec_config = GPTConfig(
        sequence_len=seq_len,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        n_embd=n_embd,
        window_pattern="L",
    )
    enc_config = EncoderConfig(
        n_embd=n_embd // 2,
        depth=encoder_depth,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
        head_dim=HEAD_DIM // 2,
    )
    xattn_config = CrossAttentionConfig(
        frequency=xattn_frequency,
        n_head=N_HEAD,
        n_kv_head=N_KV_HEAD,
    )
    model = EncoderDecoderGPT(
        dec_config, enc_config, xattn_config,
        n_sae_features=n_sae_features,
        num_buckets=num_buckets,
        feature_pred_config=feature_pred_config,
        feature_vocab=feature_vocab,
    )
    model.init_weights()
    return model


def build_model_with_feature_prediction(n_sae_features=N_SAE_FEATURES, **kwargs):
    """Build a tiny model with feature prediction auxiliary loss enabled."""
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.FULL_SEGMENT,
        loss_fn=FeaturePredLossFn.BCE,
    )
    return build_tiny_model(
        n_sae_features=n_sae_features,
        feature_pred_config=fp_config,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Precomputed packed-row helpers
# ---------------------------------------------------------------------------


def _make_identity_selection(shard: PackedRowShard) -> SelectionShard:
    """Create a SelectionShard that selects all features from the base shard."""
    return SelectionShard(
        sel_offsets=shard.seg_feat_offsets.to(torch.int32),
        sel_fids=shard.feature_ids,
        sel_buckets=torch.ones(len(shard.feature_ids), dtype=torch.uint8),
        sel_pcts=shard.feature_pcts,
        neg_offsets=torch.zeros(shard.total_segments + 1, dtype=torch.int32),
        neg_fids=torch.tensor([], dtype=torch.uint16),
    )


def _make_test_shard(
    n_rows=8,
    context_len=SEQ_LEN,
    segs_per_row: int | list[int] = 3,
    feats_per_seg=20,
    n_features=N_SAE_FEATURES,
    seed=42,
):
    """Create a synthetic PackedRowShard for testing.

    Args:
        segs_per_row: If int, every row has this many segments. If list[int],
            row r has segs_per_row[r] segments (heterogeneous; useful for
            cross-shard tests that need variable n_seg).
    """
    torch.manual_seed(seed)
    T = context_len

    # Normalize segs_per_row to a list of per-row counts
    if isinstance(segs_per_row, int):
        per_row_counts = [segs_per_row] * n_rows
    else:
        assert len(segs_per_row) == n_rows, (
            f"segs_per_row list length {len(segs_per_row)} != n_rows {n_rows}"
        )
        per_row_counts = list(segs_per_row)

    total_segs = sum(per_row_counts)
    total_feats = total_segs * feats_per_seg

    tokens = torch.randint(1, VOCAB_SIZE, (n_rows, T + 1), dtype=torch.uint16)

    # CSR row offsets from cumulative per-row counts
    row_seg_offsets = torch.zeros(n_rows + 1, dtype=torch.int32)
    cum = 0
    for r in range(n_rows):
        cum += per_row_counts[r]
        row_seg_offsets[r + 1] = cum

    seg_token_start = torch.zeros(total_segs, dtype=torch.uint16)
    seg_token_end = torch.zeros(total_segs, dtype=torch.uint16)
    seg_doc_id = torch.zeros(total_segs, dtype=torch.uint8)

    for r in range(n_rows):
        n_seg_r = per_row_counts[r]
        seg_size = T // n_seg_r
        start_idx = int(row_seg_offsets[r].item())
        for s in range(n_seg_r):
            idx = start_idx + s
            seg_token_start[idx] = s * seg_size
            # Last segment in each row extends to T (no remainder gap)
            seg_token_end[idx] = T if s == n_seg_r - 1 else (s + 1) * seg_size
            seg_doc_id[idx] = 0

    seg_feat_offsets = torch.arange(
        0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=torch.int32,
    )
    feature_ids = torch.randint(0, n_features, (total_feats,), dtype=torch.uint16)
    feature_pcts = torch.randint(1, 255, (total_feats,), dtype=torch.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def make_precomputed_dataloader(
    tmp_path,
    B=2,
    n_rows=8,
    segs_per_row=3,
    feats_per_seg=20,
    n_sae_features=N_SAE_FEATURES,
    pipeline=None,
    label_dropout=0.0,
    seed=42,
):
    """Write temporary packed row shards and return a precomputed_data_loader.

    Just pass a pytest tmp_path -- no mock patches needed.
    """
    if pipeline is None:
        boundaries = (
            torch.linspace(
                1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1,
                dtype=torch.float64,
            )
            .unsqueeze(0)
            .expand(n_sae_features, -1)
            .contiguous()
        )
        pipeline = build_threshold_pipeline(boundaries)

    shard_dir = Path(tmp_path) / "packed"
    shard_dir.mkdir(parents=True, exist_ok=True)

    shard = _make_test_shard(
        n_rows=n_rows,
        segs_per_row=segs_per_row,
        feats_per_seg=feats_per_seg,
        n_features=n_sae_features,
        seed=seed,
    )
    save_packed_row_shard(shard, shard_dir / "shard_00000")

    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {}}))

    return precomputed_data_loader(
        packed_row_dir=shard_dir,
        B=B,
        device="cpu",
        label_dropout=label_dropout,
        feature_pipeline=pipeline,
        n_sae_features=n_sae_features,
    )


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------


def train_steps(model, optimizer, dataloader, n_steps, grad_accum=1):
    """Run n_steps of training, return list of per-step losses.

    For models with feature prediction, passes seg_doc_id.
    The returned loss is the combined loss (LM + FP) for FP models, or just LM loss.
    """
    model.train()
    losses = []
    for _ in range(n_steps):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            x, y, enc_fids, enc_mags, enc_mask, _enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = next(dataloader)
            result = model(
                x, y, enc_fids, enc_mags, enc_mask, seg_map,
                seg_doc_id=seg_doc_id,
            )
            if isinstance(result, ForwardResult):
                loss = result.loss / grad_accum
            else:
                loss = result / grad_accum
            loss.backward()
            step_loss += loss.item()
        optimizer.step()
        losses.append(step_loss)
    return losses


def setup_optimizer_simple(model, lr=0.001):
    """Simple AdamW optimizer for tests (avoids Muon complexity)."""
    return torch.optim.AdamW(model.parameters(), lr=lr)
