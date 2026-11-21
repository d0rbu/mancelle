"""Test-owned padded SDPA references for CUDA-only encoder/attention code."""

from collections.abc import Iterator
from contextlib import contextmanager
from functools import wraps
from typing import cast
from unittest.mock import patch

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from errors import SelfCondTrainingError
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer
from steering.conditional.encoder_decoder.cross_attention import norm as cross_attention_norm
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    SteForm,
    SteInputs,
    norm,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    mark_packed_encoder_output_dynamic,
    pack_encoder_layout,
)


def pack_dense_encoder_output_reference(
    encoder_output: t.Tensor,
    mask: t.Tensor,
) -> PackedEncoderOutput:
    """Pack the removed dense contract for differential tests only."""
    if encoder_output.ndim != 4:
        raise ValueError(
            f"encoder_output must have shape (B, S, F, C), got {tuple(encoder_output.shape)}"
        )
    layout = pack_encoder_layout(mask)
    expected_prefix = (
        layout.batch_size,
        layout.n_segments,
        layout.feature_count,
    )
    if tuple(encoder_output.shape[:3]) != expected_prefix:
        raise ValueError(
            "encoder_output leading dimensions disagree with mask: "
            f"{tuple(encoder_output.shape[:3])} != {expected_prefix}"
        )
    flat = encoder_output.reshape(-1, encoder_output.size(-1))
    gathered = flat[layout.flat_indices]
    values = t.where(
        layout.valid_mask[:, None],
        gathered,
        t.zeros_like(gathered),
    ).unsqueeze(0)
    packed = PackedEncoderOutput(values=values, layout=layout)
    mark_packed_encoder_output_dynamic(packed)
    return packed


def densify_packed_encoder_output_reference(
    packed: PackedEncoderOutput,
) -> tuple[t.Tensor, t.Tensor]:
    """Reconstruct the padded CPU oracle tensor and mask from packed owners."""
    layout = packed.layout
    padded_features = (
        (layout.feature_count + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT
    ) * FP8_ALIGNMENT
    valid = layout.valid_mask
    feature_slots = layout.flat_indices[valid] % layout.feature_count
    dense_indices = (
        layout.batch_indices[valid] * layout.n_segments
        + layout.segment_indices[valid]
    ) * padded_features + feature_slots
    dense = packed.values.new_zeros(
        layout.batch_size * layout.n_segments * padded_features,
        packed.values.size(-1),
    )
    dense = dense.index_add(0, dense_indices, packed.values[0, valid])
    dense = dense.view(
        layout.batch_size,
        layout.n_segments,
        padded_features,
        packed.values.size(-1),
    )
    mask = t.zeros(
        layout.batch_size * layout.n_segments * padded_features,
        dtype=t.bool,
        device=packed.values.device,
    )
    mask[dense_indices] = True
    return dense, mask.view(
        layout.batch_size,
        layout.n_segments,
        padded_features,
    )


def padded_cross_attention_reference(
    layer: CrossAttentionLayer,
    decoder_hidden: t.Tensor,
    encoder_kv: t.Tensor,
    encoder_mask: t.Tensor,
    seg_map: t.Tensor,
    static_max_seg_size: int | None = None,
) -> t.Tensor:
    """Run the removed dense CPU SDPA implementation as a test-only oracle."""
    batch_size, sequence_length, width = decoder_hidden.size()
    if encoder_kv.ndim != 4:
        raise ValueError(
            f"padded encoder_kv must have shape (B, S, F, C), got {tuple(encoder_kv.shape)}"
        )
    if encoder_mask.shape != encoder_kv.shape[:3]:
        raise ValueError(
            "padded encoder mask must match encoder_kv leading dimensions: "
            f"{tuple(encoder_mask.shape)} != {tuple(encoder_kv.shape[:3])}"
        )
    n_segments = encoder_mask.size(1)
    feature_count = encoder_mask.size(2)

    if not t.compiler.is_compiling():
        assert seg_map.min() >= 0, (
            f"seg_map values must be >= 0; got seg_map.min()={int(seg_map.min())}"
        )
        assert seg_map.max() < n_segments, (
            f"seg_map values must be < n_seg ({n_segments}); got seg_map.max()={int(seg_map.max())}"
        )
        if sequence_length > 1:
            non_zero_seg_max = (
                t.where(seg_map > 0, seg_map, t.zeros_like(seg_map))
                .cummax(dim=1)
                .values
            )
            assert ((seg_map == 0) | (seg_map >= non_zero_seg_max)).all(), (
                "seg_map non-zero values must be monotonically non-decreasing "
                "within each row; a non-zero value followed by a smaller "
                "non-zero value violates the ordered-segment contract."
            )

    position = t.arange(sequence_length, device=seg_map.device).unsqueeze(0)
    position = position.expand(batch_size, -1)
    is_segment_start = t.zeros(
        batch_size,
        sequence_length,
        dtype=t.bool,
        device=seg_map.device,
    )
    is_segment_start[:, 0] = True
    is_segment_start[:, 1:] = seg_map[:, 1:] != seg_map[:, :-1]
    segment_start = t.where(is_segment_start, position, t.zeros_like(position))
    segment_start = segment_start.cummax(dim=1).values
    position_in_segment = position - segment_start
    required_max_seg_size = position_in_segment.amax() + 1

    if static_max_seg_size is not None:
        assert static_max_seg_size > 0, (
            f"static_max_seg_size must be > 0; got {static_max_seg_size}"
        )
        if t.compiler.is_compiling():
            t._assert_async(
                required_max_seg_size <= static_max_seg_size,
                "static_max_seg_size is smaller than required "
                "(= pos_in_seg.amax() + 1); packed-row manifests store "
                "the raw observed segment maximum.",
            )
        else:
            required = int(required_max_seg_size.item())
            assert required <= static_max_seg_size, (
                f"static_max_seg_size={static_max_seg_size} is smaller than "
                f"required={required} (= pos_in_seg.amax() + 1); "
                "the batch violates its packed-row manifest bound."
            )
        raw_max_seg_size = static_max_seg_size
    else:
        raw_max_seg_size = required_max_seg_size.clamp(min=1)

    max_seg_size = (
        (raw_max_seg_size + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT
    ) * FP8_ALIGNMENT
    padded_indices = seg_map * max_seg_size + position_in_segment
    padded_length = n_segments * max_seg_size
    padded_hidden = decoder_hidden.new_zeros(
        batch_size,
        padded_length,  # ty: ignore[invalid-argument-type]
        width,
    )
    padded_hidden.scatter_(
        1,
        padded_indices.unsqueeze(-1).expand(-1, -1, width),
        decoder_hidden,
    )
    padded_hidden = padded_hidden.view(
        batch_size,
        n_segments,
        max_seg_size,
        width,
    )

    query = layer.c_q(
        padded_hidden.reshape(batch_size * n_segments, max_seg_size, width)
    )
    query = query.view(
        batch_size * n_segments,
        max_seg_size,
        layer.n_head,
        layer.head_dim,
    )
    encoder_flat = encoder_kv.reshape(
        batch_size * n_segments,
        feature_count,
        layer.enc_n_embd,
    )
    key_value = layer.c_kv(encoder_flat)
    key, value = key_value.split((layer.kv_dim, layer.kv_dim), dim=-1)
    key = key.view(
        batch_size * n_segments,
        feature_count,
        layer.n_kv_head,
        layer.head_dim,
    )
    value = value.view_as(key)
    query = cross_attention_norm(query).transpose(1, 2).nan_to_num()
    key = cross_attention_norm(key).transpose(1, 2).nan_to_num()
    value = value.transpose(1, 2)
    attention_mask = (
        encoder_mask.reshape(
            batch_size * n_segments,
            feature_count,
        )
        .unsqueeze(1)
        .unsqueeze(1)
    )
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        is_causal=False,
        enable_gqa=layer.n_head != layer.n_kv_head,
    ).nan_to_num()
    output = output.transpose(1, 2).reshape(
        batch_size,
        padded_length,  # ty: ignore[invalid-argument-type]
        layer.n_head * layer.head_dim,
    )
    output = layer.c_proj(output)
    return output.gather(
        1,
        padded_indices.unsqueeze(-1).expand(-1, -1, output.size(-1)),
    )


def padded_encoder_reference(
    encoder: FeatureEncoder,
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
    mask: t.Tensor,
    ste: SteInputs | None = None,
) -> tuple[t.Tensor, t.Tensor]:
    """Run the pre-packing dense implementation as a differential oracle."""
    batch_size, n_segments, feature_count = feature_ids.shape
    padded_feature_count = (
        (feature_count + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT
    ) * FP8_ALIGNMENT
    pad_feature_count = padded_feature_count - feature_count
    feature_ids = F.pad(feature_ids, (0, pad_feature_count))
    magnitudes = F.pad(magnitudes, (0, pad_feature_count))
    mask = F.pad(mask, (0, pad_feature_count))

    ids_flat = feature_ids.reshape(batch_size * n_segments, padded_feature_count)
    mags_flat = magnitudes.reshape(batch_size * n_segments, padded_feature_count)
    mask_flat = mask.reshape(batch_size * n_segments, padded_feature_count)
    ids_flat = encoder._to_compact(ids_flat, mask_flat)
    x = encoder.feature_embed(ids_flat) + encoder.magnitude_embed(mags_flat)

    sign_flat: t.Tensor
    scores_flat: t.Tensor
    if ste is not None:
        # Mirrors ``EncoderDecoderGPT`` encoder: a compacted encoder emits STE
        # scores in compact space (``feature_vocab.size`` embedding rows), so
        # the reference must accept the same widths the real encoder does.
        expected_scores_shape = (
            batch_size,
            n_segments,
            encoder.config.n_sae_features
            if encoder.feature_vocab is None
            else encoder.feature_vocab.size,
        )
        if tuple(ste.scores.shape) != expected_scores_shape:
            # Message must match ``EncoderDecoderGPT``'s encoder exactly: the
            # compact case names the compacted axis so a width mismatch is
            # attributable, and the reference is contract-compared against it.
            if encoder.feature_vocab is not None:
                raise SelfCondTrainingError(
                    "ste.scores shape for a compacted encoder must be "
                    f"{expected_scores_shape}, got {tuple(ste.scores.shape)}"
                )
            raise SelfCondTrainingError(
                f"ste.scores shape must be {expected_scores_shape}, got {tuple(ste.scores.shape)}"
            )
        expected_sign_shape = (batch_size, n_segments, feature_count)
        if tuple(ste.sign.shape) != expected_sign_shape:
            raise SelfCondTrainingError(
                f"ste.sign shape must be {expected_sign_shape}, got {tuple(ste.sign.shape)}"
            )

        sign_flat = F.pad(ste.sign, (0, pad_feature_count)).reshape(
            batch_size * n_segments,
            padded_feature_count,
        )
        scores_flat = ste.scores.reshape(batch_size * n_segments, -1)
        if ste.form == SteForm.DENSE:
            mix = scores_flat.to(x.dtype) @ encoder.feature_embed.weight.detach()
            x = x + sign_flat.to(x.dtype).unsqueeze(-1) * (
                mix - mix.detach()
            ).unsqueeze(1)
        else:
            col_scores = scores_flat.gather(1, ids_flat)
            rows = encoder.feature_embed.weight.detach()[ids_flat]
            term = col_scores.to(x.dtype).unsqueeze(-1) * rows
            x = x + sign_flat.to(x.dtype).unsqueeze(-1) * (term - term.detach())

    for index, block in enumerate(encoder.blocks):
        value_embed = (
            cast(nn.Embedding, encoder.value_embeds[str(index)])
            if str(index) in encoder.value_embeds
            else None
        )
        value = value_embed(ids_flat) if value_embed is not None else None
        if ste is not None and value is not None:
            assert value_embed is not None
            value_weight = value_embed.weight.detach()
            if ste.form == SteForm.DENSE:
                mix = scores_flat.to(value.dtype) @ value_weight
                value = value + sign_flat.to(value.dtype).unsqueeze(-1) * (
                    mix - mix.detach()
                ).unsqueeze(1)
            else:
                col_scores = scores_flat.gather(1, ids_flat)
                rows = value_weight[ids_flat]
                term = col_scores.to(value.dtype).unsqueeze(-1) * rows
                value = value + sign_flat.to(value.dtype).unsqueeze(-1) * (
                    term - term.detach()
                )
        x = block(x, ve=value, mask=mask_flat)

    x = norm(x).reshape(
        batch_size,
        n_segments,
        padded_feature_count,
        encoder.config.n_embd,
    )
    return x, mask


production_feature_encoder_forward = FeatureEncoder.forward
production_feature_encoder_forward_packed = FeatureEncoder.forward_packed


@wraps(production_feature_encoder_forward)
def feature_encoder_forward_with_cpu_reference(
    encoder: FeatureEncoder,
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
    mask: t.Tensor,
    ste: SteInputs | None = None,
) -> PackedEncoderOutput:
    """Preserve inexpensive CPU semantic tests without restoring CPU production."""
    if feature_ids.is_cuda:
        return production_feature_encoder_forward(
            encoder,
            feature_ids,
            magnitudes,
            mask,
            ste=ste,
        )
    dense, dense_mask = padded_encoder_reference(
        encoder,
        feature_ids,
        magnitudes,
        mask,
        ste=ste,
    )
    return pack_dense_encoder_output_reference(dense, dense_mask)


@wraps(production_feature_encoder_forward_packed)
def feature_encoder_forward_packed_with_cpu_reference(
    encoder: FeatureEncoder,
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
    mask: t.Tensor,
    ste: SteInputs | None = None,
) -> PackedEncoderOutput:
    """Support model-level CPU tests without a production CPU branch."""
    if feature_ids.is_cuda:
        return production_feature_encoder_forward_packed(
            encoder,
            feature_ids,
            magnitudes,
            mask,
            ste=ste,
        )
    dense, dense_mask = padded_encoder_reference(
        encoder,
        feature_ids,
        magnitudes,
        mask,
        ste=ste,
    )
    return pack_dense_encoder_output_reference(dense, dense_mask)


production_cross_attention_forward = CrossAttentionLayer.forward


@wraps(production_cross_attention_forward)
def cross_attention_forward_with_cpu_reference(
    layer: CrossAttentionLayer,
    decoder_hidden: t.Tensor,
    encoder_kv: t.Tensor | PackedEncoderOutput,
    encoder_mask_or_seg_map: t.Tensor | None,
    seg_map: t.Tensor | None = None,
    static_max_seg_size: int | None = None,
    packed_block_mask=None,
) -> t.Tensor:
    """Install test-only dense/CPU behavior without widening production's API."""
    if isinstance(encoder_kv, PackedEncoderOutput):
        if seg_map is None:
            if not isinstance(encoder_mask_or_seg_map, t.Tensor):
                raise TypeError("packed test cross-attention requires seg_map")
            resolved_seg_map = encoder_mask_or_seg_map
        else:
            if encoder_mask_or_seg_map is not None:
                raise TypeError("packed test cross-attention does not accept encoder_mask")
            resolved_seg_map = seg_map
        if decoder_hidden.is_cuda:
            return production_cross_attention_forward(
                layer,
                decoder_hidden,
                encoder_kv,
                resolved_seg_map,
                static_max_seg_size=static_max_seg_size,
                packed_block_mask=packed_block_mask,
            )
        dense, dense_mask = densify_packed_encoder_output_reference(encoder_kv)
        return padded_cross_attention_reference(
            layer,
            decoder_hidden,
            dense,
            dense_mask,
            resolved_seg_map,
            static_max_seg_size=static_max_seg_size,
        )

    if not isinstance(encoder_mask_or_seg_map, t.Tensor) or seg_map is None:
        raise TypeError("dense test cross-attention requires encoder_mask and seg_map")
    return padded_cross_attention_reference(
        layer,
        decoder_hidden,
        encoder_kv,
        encoder_mask_or_seg_map,
        seg_map,
        static_max_seg_size=static_max_seg_size,
    )


@contextmanager
def use_test_owned_cpu_encoder_reference() -> Iterator[None]:
    """Install the padded CPU oracle in the current process for one scope.

    Pytest's autouse monkeypatch fixture installs these references in the parent
    process, but the ``spawn`` multiprocessing start method imports modules in a
    fresh interpreter. Distributed CPU tests must therefore install the same
    test-owned boundary explicitly inside each worker. CUDA inputs continue to
    dispatch to the captured production implementations.
    """
    with (
        patch.object(
            FeatureEncoder,
            "forward",
            feature_encoder_forward_with_cpu_reference,
        ),
        patch.object(
            FeatureEncoder,
            "forward_packed",
            feature_encoder_forward_packed_with_cpu_reference,
        ),
        patch.object(
            CrossAttentionLayer,
            "forward",
            cross_attention_forward_with_cpu_reference,
        ),
    ):
        yield
