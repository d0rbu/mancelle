"""Deterministic tiny-model and multi-document batch builders for self-conditioning tests."""

from __future__ import annotations

from dataclasses import dataclass

import torch as t

from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from tests.encdec_test_helpers import VOCAB_SIZE, build_tiny_model


def tiny_head_model(
    pool_mode: FeaturePredLossMode,
    *,
    n_sae: int = 32,
    num_buckets: int = 2,
    seed: int = 0,
    encoder_depth: int = 1,
) -> EncoderDecoderGPT:
    """Build a deterministic tiny model whose FP head uses ``pool_mode``.

    ``encoder_depth >= 3`` yields more than one ``has_ve`` block so gradient
    tests can pin the STE value-embedding term across every value-embedding
    layer (the default 1 keeps a single block, matching earlier callers).
    """
    t.manual_seed(seed)
    feature_pred_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=pool_mode,
        loss_fn=FeaturePredLossFn.BCE,
    )
    return build_tiny_model(
        n_sae_features=n_sae,
        num_buckets=num_buckets,
        feature_pred_config=feature_pred_config,
        n_layer=2,
        xattn_frequency=1,
        encoder_depth=encoder_depth,
    )


def perturb_residual_projections(module: t.nn.Module, *, seed: int = 0, scale: float = 0.2) -> None:
    """Add small deterministic noise to every zero-initialized residual/gate projection.

    ``init_weights`` zero-inits ``attn.c_proj``, ``mlp.c_proj`` and ``ve_gate``
    for residual-friendly startup. At that degenerate point the gradient of an
    upstream weight (``c_qkv``/``c_fc``/``value_embeds``) routes through a zero
    projection and is itself exactly zero — so a gradient test built on a fresh
    tiny model cannot observe the value-embedding path (or most block weights)
    at all. Perturbing these projections moves the model off that point, exactly
    as one optimizer step would, making the whole gradient surface live.
    """
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for name, parameter in module.named_parameters():
            if name.endswith(("attn.c_proj.weight", "mlp.c_proj.weight", "ve_gate.weight")):
                parameter.add_(
                    t.randn(parameter.shape, generator=generator, dtype=parameter.dtype) * scale
                )


@dataclass
class SelfcondBatch:
    """The eight tensor fields consumed from an encoder-decoder train batch."""

    inputs: t.Tensor
    targets: t.Tensor
    enc_fids: t.Tensor
    enc_mags: t.Tensor
    enc_mask: t.Tensor
    enc_pcts: t.Tensor
    seg_map: t.Tensor
    seg_doc_id: t.Tensor


def make_batch(
    layout: list[list[int]],
    *,
    T: int,
    F: int,
    n_sae: int,
    seed: int,
) -> SelfcondBatch:
    """Build exact-tiling rows whose documents own the requested segment counts.

    ``layout[b]`` lists the number of segments in each document on row ``b``.
    Rows are padded to the largest total segment count with ``seg_doc_id=-1``.
    """
    if not layout:
        raise ValueError("layout must contain at least one batch row")
    if T <= 0 or F <= 0 or n_sae <= 0:
        raise ValueError(f"T, F, and n_sae must be positive; got {(T, F, n_sae)}")
    if any(not row or any(count <= 0 for count in row) for row in layout):
        raise ValueError("every layout row must contain positive document segment counts")

    row_segment_counts = [sum(row) for row in layout]
    if any(count > T for count in row_segment_counts):
        raise ValueError(
            f"every real segment needs a decoder token: counts={row_segment_counts}, T={T}"
        )
    batch_size = len(layout)
    n_seg = max(row_segment_counts)
    generator = t.Generator().manual_seed(seed)

    inputs = t.randint(2, VOCAB_SIZE, (batch_size, T), generator=generator, dtype=t.int64)
    targets = t.randint(0, VOCAB_SIZE, (batch_size, T), generator=generator, dtype=t.int64)
    enc_fids = t.zeros(batch_size, n_seg, F, dtype=t.int64)
    enc_mags = t.zeros(batch_size, n_seg, F, dtype=t.int64)
    enc_mask = t.zeros(batch_size, n_seg, F, dtype=t.bool)
    enc_pcts = t.zeros(batch_size, n_seg, F, dtype=t.float32)
    seg_map = t.zeros(batch_size, T, dtype=t.int64)
    seg_doc_id = t.full((batch_size, n_seg), -1, dtype=t.int64)

    positive_columns = (F + 1) // 2
    for row_index, (documents, real_segments) in enumerate(
        zip(layout, row_segment_counts, strict=True)
    ):
        cursor = 0
        for document_id, count in enumerate(documents):
            seg_doc_id[row_index, cursor : cursor + count] = document_id
            cursor += count

        token_positions = t.arange(T, dtype=t.int64)
        seg_map[row_index] = (token_positions * real_segments) // T
        enc_fids[row_index, :real_segments] = t.randint(
            0,
            n_sae,
            (real_segments, F),
            generator=generator,
            dtype=t.int64,
        )
        enc_mags[row_index, :real_segments, positive_columns:] = 1
        enc_mask[row_index, :real_segments] = True
        enc_pcts[row_index, :real_segments] = t.rand(
            real_segments,
            F,
            generator=generator,
            dtype=t.float32,
        )
        enc_pcts[row_index, :real_segments, positive_columns:] = 1.0

    return SelfcondBatch(
        inputs=inputs,
        targets=targets,
        enc_fids=enc_fids,
        enc_mags=enc_mags,
        enc_mask=enc_mask,
        enc_pcts=enc_pcts,
        seg_map=seg_map,
        seg_doc_id=seg_doc_id,
    )
