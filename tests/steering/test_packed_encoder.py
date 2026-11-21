"""Differential tests for the globally packed CUDA encoder path."""

import copy
import inspect

import pytest
import torch as t
import torch.nn.functional as F
from torch._dynamo.utils import counters
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from steering.conditional.encoder_decoder import cross_attention as cross_attention_module
from steering.conditional.encoder_decoder import model as model_module
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
    build_packed_cross_attention_block_mask,
)
from steering.conditional.encoder_decoder.cross_attention import (
    norm as xattn_norm,
)
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
    SteForm,
    SteInputs,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    PACKED_ENCODER_ALIGNMENT,
    PACKED_ENCODER_BLOCK_SIZE,
    PackedEncoderLayout,
    PackedEncoderOutput,
    append_packed_encoder_segment,
    build_block_mask_from_dense_routing,
    build_packed_encoder_block_mask,
    pack_encoder_inputs,
    pack_encoder_layout,
    repeat_packed_encoder_output,
    stack_packed_encoder_outputs,
)
from tests.encdec_test_helpers import build_tiny_model
from tests.padded_encoder_reference import (
    densify_packed_encoder_output_reference,
    pack_dense_encoder_output_reference,
    padded_encoder_reference,
    production_cross_attention_forward,
    production_feature_encoder_forward,
)

requires_cuda = pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")


def _reference_encoder_block_mask(layout: PackedEncoderLayout) -> BlockMask:
    valid_mask = layout.valid_mask
    batch_indices = layout.batch_indices
    segment_indices = layout.segment_indices

    def mask_mod(
        batch_index: t.Tensor,
        head_index: t.Tensor,
        query_index: t.Tensor,
        kv_index: t.Tensor,
        **kwargs: object,
    ) -> t.Tensor:
        del batch_index, head_index, kwargs
        return (
            valid_mask[query_index]
            & valid_mask[kv_index]
            & (batch_indices[query_index] == batch_indices[kv_index])
            & (segment_indices[query_index] == segment_indices[kv_index])
        ) | (~valid_mask[query_index] & (query_index == kv_index))

    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=layout.packed_count,
        KV_LEN=layout.packed_count,
        device=valid_mask.device,
        BLOCK_SIZE=PACKED_ENCODER_BLOCK_SIZE,
    )


def _reference_cross_attention_block_mask(
    seg_map: t.Tensor,
    layout: PackedEncoderLayout,
) -> BlockMask:
    batch_size, sequence_length = seg_map.shape
    flat_seg_map = seg_map.reshape(-1)
    flat_batch_indices = (
        t.arange(batch_size, device=seg_map.device)[:, None].expand_as(seg_map).reshape(-1)
    )
    valid_mask = layout.valid_mask
    encoder_batch_indices = layout.batch_indices
    encoder_segment_indices = layout.segment_indices

    def mask_mod(
        batch_index: t.Tensor,
        head_index: t.Tensor,
        query_index: t.Tensor,
        kv_index: t.Tensor,
        **kwargs: object,
    ) -> t.Tensor:
        del batch_index, head_index, kwargs
        return (
            valid_mask[kv_index]
            & (encoder_batch_indices[kv_index] == flat_batch_indices[query_index])
            & (encoder_segment_indices[kv_index] == flat_seg_map[query_index])
        )

    return create_block_mask(
        mask_mod,
        B=1,
        H=None,
        Q_LEN=batch_size * sequence_length,
        KV_LEN=layout.packed_count,
        device=seg_map.device,
        BLOCK_SIZE=PACKED_ENCODER_BLOCK_SIZE,
    )


def _assert_block_mask_metadata_equal(actual: BlockMask, expected: BlockMask) -> None:
    assert actual.seq_lengths == expected.seq_lengths
    assert actual.BLOCK_SIZE == expected.BLOCK_SIZE
    for name in BlockMask._TENSOR_ATTRS:
        actual_tensor = getattr(actual, name)
        expected_tensor = getattr(expected, name)
        assert (actual_tensor is None) == (expected_tensor is None), name
        if actual_tensor is not None:
            assert t.equal(actual_tensor, expected_tensor), name


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize(
    "shape,kind,noncontiguous",
    [
        ((128, 128), "full", False),
        ((129, 129), "partial", False),
        ((129, 257), "partial", True),
        ((127, 127), "empty", False),
    ],
)
def test_dense_routing_metadata_matches_scalar_reference(
    device: str, shape: tuple[int, int], kind: str, noncontiguous: bool
) -> None:
    query_length, kv_length = shape
    routing = t.ones(shape, dtype=t.bool, device=device)
    if kind == "partial":
        routing = t.arange(query_length * kv_length, device=device).reshape(shape) % 3 == 0
    elif kind == "empty":
        routing.zero_()
    if noncontiguous:
        backing = t.zeros(query_length, 2 * kv_length + 1, dtype=t.bool, device=device)
        backing[:, 1::2] = routing
        routing = backing[:, 1::2]
        assert not routing.is_contiguous()
        assert routing.storage_offset() == 1

    def mask_mod(batch_index, head_index, query_index, kv_index):
        del batch_index, head_index
        return (
            (query_index < query_length)
            & (kv_index < kv_length)
            & routing[query_index.clamp(0, query_length - 1), kv_index.clamp(0, kv_length - 1)]
        )

    actual = build_block_mask_from_dense_routing(routing, mask_mod=mask_mod)
    expected = create_block_mask(
        mask_mod,
        B=1,
        H=1,
        Q_LEN=query_length,
        KV_LEN=kv_length,
        device=device,
        BLOCK_SIZE=PACKED_ENCODER_BLOCK_SIZE,
    )
    _assert_block_mask_metadata_equal(actual, expected)
    if kind == "full":
        # A full 128 x 128 tile contains 16,384 true entries.
        assert actual.full_kv_num_blocks is not None
        assert int(actual.full_kv_num_blocks.sum()) == 1


def test_production_feature_encoder_rejects_cpu_inputs() -> None:
    config = FeatureEncoderConfig(
        n_sae_features=16,
        num_buckets=4,
        n_embd=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        head_dim=16,
    )
    encoder = FeatureEncoder(config)
    ids = t.zeros(1, 1, 1, dtype=t.int64)
    magnitudes = t.zeros_like(ids)
    mask = t.ones_like(ids, dtype=t.bool)

    with pytest.raises(RuntimeError, match="requires CUDA inputs"):
        production_feature_encoder_forward(encoder, ids, magnitudes, mask)


def test_production_cross_attention_rejects_cpu_packed_state() -> None:
    layer = CrossAttentionLayer(
        CrossAttentionLayerConfig(
            dec_n_embd=32,
            enc_n_embd=16,
            n_head=2,
            n_kv_head=2,
            head_dim=16,
        )
    )
    decoder = t.randn(1, 2, 32)
    packed = pack_dense_encoder_output_reference(
        t.randn(1, 1, 2, 16),
        t.ones(1, 1, 2, dtype=t.bool),
    )
    seg_map = t.zeros(1, 2, dtype=t.long)

    with pytest.raises(RuntimeError, match="requires CUDA packed state"):
        production_cross_attention_forward(
            layer,
            decoder,
            packed,
            seg_map,
        )


def test_production_cross_attention_contract_is_packed_only() -> None:
    signature = inspect.signature(production_cross_attention_forward)
    assert "encoder_mask" not in signature.parameters
    assert signature.parameters["encoder_kv"].annotation is PackedEncoderOutput

    source = inspect.getsource(production_cross_attention_forward)
    assert "scaled_dot_product_attention" not in source
    assert "encoder_mask" not in source


def _encoder() -> FeatureEncoder:
    config = FeatureEncoderConfig(
        n_sae_features=128,
        num_buckets=8,
        n_embd=64,
        n_layer=2,
        n_head=4,
        n_kv_head=2,
        head_dim=16,
    )
    encoder = FeatureEncoder(config).cuda()
    encoder.init_weights()
    # Production initialization makes both residual projections zero. Exercise
    # the attention and MLP numerics rather than comparing two identity stacks.
    with t.no_grad():
        for block in encoder.blocks:
            block.attn.c_proj.weight.normal_(std=0.02)
            block.mlp.c_proj.weight.normal_(std=0.02)
    return encoder


def _encoder_inputs() -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    ids = t.randint(0, 128, (2, 4, 13), device="cuda")
    magnitudes = t.randint(0, 8, ids.shape, device="cuda")
    counts = ((3, 11, 0, 7), (13, 1, 8, 5))
    mask = t.zeros_like(ids, dtype=t.bool)
    for batch_index, row in enumerate(counts):
        for segment_index, count in enumerate(row):
            mask[batch_index, segment_index, :count] = True
    return ids, magnitudes, mask


def _assert_close(actual: t.Tensor, expected: t.Tensor, *, name: str) -> None:
    t.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2, msg=name)


def _assert_gradient_close(actual: t.Tensor, expected: t.Tensor, *, name: str) -> None:
    """Compare BF16 attention gradients by direction and relative L2 error."""
    actual_float = actual.float().flatten()
    expected_float = expected.float().flatten()
    assert t.isfinite(actual_float).all(), f"{name}: actual gradient is not finite"
    assert t.isfinite(expected_float).all(), f"{name}: reference gradient is not finite"
    expected_norm = expected_float.norm()
    if expected_norm == 0:
        assert actual_float.norm() == 0, f"{name}: expected an exactly zero gradient"
        return
    relative_l2 = (actual_float - expected_float).norm() / expected_norm
    cosine = F.cosine_similarity(actual_float, expected_float, dim=0)
    assert relative_l2 < 3e-2, f"{name}: relative L2 error {relative_l2.item():.6f}"
    assert cosine > 0.999, f"{name}: cosine similarity {cosine.item():.6f}"


def _cross_attention_reference(
    layer: CrossAttentionLayer,
    decoder_hidden: t.Tensor,
    encoder_kv: t.Tensor,
    encoder_mask: t.Tensor,
    seg_map: t.Tensor,
) -> t.Tensor:
    """Independent gather-per-query SDPA oracle for packed cross-attention."""
    batch_size, sequence_length, width = decoder_hidden.shape
    feature_count = encoder_kv.size(2)
    batch_indices = t.arange(batch_size, device=seg_map.device)[:, None]
    routed_kv = encoder_kv[batch_indices, seg_map]
    routed_mask = encoder_mask[batch_indices, seg_map]

    q = layer.c_q(decoder_hidden).view(
        batch_size * sequence_length,
        1,
        layer.n_head,
        layer.head_dim,
    )
    kv = layer.c_kv(routed_kv).view(
        batch_size * sequence_length,
        feature_count,
        2 * layer.kv_dim,
    )
    k, v = kv.split((layer.kv_dim, layer.kv_dim), dim=-1)
    k = k.view(
        batch_size * sequence_length,
        feature_count,
        layer.n_kv_head,
        layer.head_dim,
    )
    v = v.view_as(k)
    q = xattn_norm(q).transpose(1, 2).nan_to_num()
    k = xattn_norm(k).transpose(1, 2).nan_to_num()
    v = v.transpose(1, 2)
    y = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=routed_mask.reshape(batch_size * sequence_length, 1, 1, feature_count),
        enable_gqa=layer.n_head != layer.n_kv_head,
    ).nan_to_num()
    y = y.transpose(1, 2).reshape(batch_size, sequence_length, width)
    return layer.c_proj(y)


class TestPackedEncoderLayout:
    def test_global_pack_preserves_owner_order_and_has_one_short_tail(self) -> None:
        mask = t.tensor(
            [
                [[True, False, True, False], [False, True, False, False]],
                [[True, True, True, False], [False, False, True, False]],
            ]
        )
        layout = pack_encoder_layout(mask)
        real = mask.reshape(-1).nonzero().flatten()

        assert t.equal(layout.flat_indices[: real.numel()], real)
        assert layout.valid_mask.sum().item() == real.numel()
        assert layout.packed_count % PACKED_ENCODER_ALIGNMENT == 0
        assert layout.packed_count - real.numel() < PACKED_ENCODER_ALIGNMENT
        assert layout.batch_indices[: real.numel()].tolist() == [0, 0, 0, 1, 1, 1, 1]
        assert layout.segment_indices[: real.numel()].tolist() == [0, 0, 1, 0, 0, 0, 1]

    def test_empty_batch_uses_one_safe_alignment_block(self) -> None:
        layout = pack_encoder_layout(t.zeros(2, 3, 7, dtype=t.bool))
        assert layout.packed_count == PACKED_ENCODER_ALIGNMENT
        assert not layout.valid_mask.any()


@requires_cuda
@pytest.mark.parametrize(
    ("shape", "mask_kind", "sequence_length"),
    [
        ((2, 3, 7), "empty", 17),
        ((1, 2, 192), "full", 257),
        ((3, 5, 43), "irregular", 389),
    ],
)
def test_vectorized_block_masks_match_create_block_mask_metadata_exactly(
    shape: tuple[int, int, int],
    mask_kind: str,
    sequence_length: int,
) -> None:
    batch_size, n_segments, feature_count = shape
    positions = t.arange(batch_size * n_segments * feature_count, device="cuda").view(
        shape
    )
    if mask_kind == "empty":
        encoder_mask = t.zeros(shape, dtype=t.bool, device="cuda")
    elif mask_kind == "full":
        encoder_mask = t.ones(shape, dtype=t.bool, device="cuda")
    else:
        encoder_mask = (positions % 5 != 0) & (positions % 11 != 0)
    layout = pack_encoder_layout(encoder_mask)
    seg_map = (
        t.arange(sequence_length, device="cuda")
        .mul(n_segments)
        .div(sequence_length, rounding_mode="floor")
        .repeat(batch_size, 1)
    )

    _assert_block_mask_metadata_equal(
        build_packed_encoder_block_mask(layout),
        _reference_encoder_block_mask(layout),
    )
    _assert_block_mask_metadata_equal(
        build_packed_cross_attention_block_mask(seg_map, layout),
        _reference_cross_attention_block_mask(seg_map, layout),
    )


def _assert_same_consumed_packed_state(actual, reference) -> None:
    """Compare every field consumed by packed cross-attention exactly."""
    assert t.equal(actual.values, reference.values)
    assert t.equal(actual.layout.valid_mask, reference.layout.valid_mask)
    assert t.equal(actual.layout.batch_indices, reference.layout.batch_indices)
    assert t.equal(actual.layout.segment_indices, reference.layout.segment_indices)
    assert actual.layout.batch_size == reference.layout.batch_size
    assert actual.layout.n_segments == reference.layout.n_segments

    actual_valid = actual.layout.valid_mask
    reference_valid = reference.layout.valid_mask
    actual_features = (
        actual.layout.flat_indices[actual_valid] % actual.layout.feature_count
    )
    reference_features = (
        reference.layout.flat_indices[reference_valid]
        % reference.layout.feature_count
    )
    assert t.equal(actual_features, reference_features)


class TestPackedEncoderStateOperations:
    def test_stack_is_bitwise_equal_to_removed_dense_batch_contract(self) -> None:
        generator = t.Generator().manual_seed(1948)
        first_dense = t.randn(1, 3, 5, 7, generator=generator)
        first_mask = t.tensor(
            [
                [
                    [True, False, True, False, False],
                    [False, True, False, True, False],
                    [True, False, False, False, True],
                ]
            ]
        )
        second_dense = t.randn(1, 3, 9, 7, generator=generator)
        second_mask = t.tensor(
            [
                [
                    [False, True, False, False, True, False, False, False, True],
                    [True, False, False, True, False, False, True, False, False],
                    [False, False, True, False, False, True, False, True, False],
                ]
            ]
        )
        first = pack_dense_encoder_output_reference(first_dense, first_mask)
        second = pack_dense_encoder_output_reference(second_dense, second_mask)

        actual = stack_packed_encoder_outputs((first, second))
        reference = pack_dense_encoder_output_reference(
            t.cat((F.pad(first_dense, (0, 0, 0, 4)), second_dense), dim=0),
            t.cat((F.pad(first_mask, (0, 4)), second_mask), dim=0),
        )

        _assert_same_consumed_packed_state(actual, reference)
        assert actual.layout.batch_size == 2
        assert actual.layout.n_segments == 3
        assert actual.layout.feature_count == 9
        assert actual.layout.packed_count - int(actual.layout.valid_mask.sum()) < 32

    def test_stack_preserves_row_order_and_does_not_alias_inputs(self) -> None:
        rows = tuple(
            pack_dense_encoder_output_reference(
                t.full((1, 2, width, 4), float(row + 1)),
                t.ones(1, 2, width, dtype=t.bool),
            )
            for row, width in enumerate((2, 5, 3))
        )
        pinned = tuple(row.values.clone() for row in rows)

        actual = stack_packed_encoder_outputs(rows)
        dense, mask = densify_packed_encoder_output_reference(actual)

        assert actual.layout.feature_count == 5
        assert dense.shape == (3, 2, 16, 4)
        assert mask.sum(dim=(1, 2)).tolist() == [4, 10, 6]
        for row, width in enumerate((2, 5, 3)):
            assert t.equal(dense[row, :, :width], t.full((2, width, 4), float(row + 1)))
        actual.values.zero_()
        assert all(
            t.equal(source.values, expected) for source, expected in zip(rows, pinned, strict=True)
        )

    def test_append_is_bitwise_equal_to_removed_dense_concat_contract(self) -> None:
        generator = t.Generator().manual_seed(1448)
        old_dense = t.randn(1, 2, 5, 7, generator=generator)
        old_mask = t.tensor(
            [[[True, False, False, True, False], [False, True, False, False, True]]]
        )
        new_dense = t.randn(1, 1, 3, 7, generator=generator)
        new_mask = t.tensor([[[True, False, True]]])
        old = pack_dense_encoder_output_reference(old_dense, old_mask)
        new = pack_dense_encoder_output_reference(new_dense, new_mask)

        actual = append_packed_encoder_segment(old, new)

        old_padded, old_padded_mask = densify_packed_encoder_output_reference(old)
        new_padded, new_padded_mask = densify_packed_encoder_output_reference(new)
        reference = pack_dense_encoder_output_reference(
            t.cat((old_padded, new_padded), dim=1),
            t.cat((old_padded_mask, new_padded_mask), dim=1),
        )
        _assert_same_consumed_packed_state(actual, reference)
        assert actual.layout.feature_count == 5
        assert reference.layout.feature_count == 16
        assert actual.layout.packed_count - int(actual.layout.valid_mask.sum()) < 32

    def test_repeat_is_bitwise_equal_to_removed_dense_expand_contract(self) -> None:
        generator = t.Generator().manual_seed(2448)
        dense = t.randn(1, 3, 6, 9, generator=generator)
        mask = t.tensor(
            [
                [
                    [True, False, True, False, False, False],
                    [False, True, False, True, False, True],
                    [True, False, False, False, True, False],
                ]
            ]
        )
        source = pack_dense_encoder_output_reference(dense, mask)

        actual = repeat_packed_encoder_output(source, 4)

        padded, padded_mask = densify_packed_encoder_output_reference(source)
        reference = pack_dense_encoder_output_reference(
            padded.expand(4, -1, -1, -1).contiguous(),
            padded_mask.expand(4, -1, -1).contiguous(),
        )
        _assert_same_consumed_packed_state(actual, reference)
        assert actual.layout.batch_size == 4
        assert repeat_packed_encoder_output(source, 1) is source

    def test_repeated_append_across_width_boundaries_matches_removed_contract(self) -> None:
        """Catch cumulative owner drift while the removed path repeatedly repads F."""
        generator = t.Generator().manual_seed(3448)
        widths = (17, 33, 4, 65, 32)
        segments: list[PackedEncoderOutput] = []
        for width in widths:
            dense = t.randn(1, 1, width, 7, generator=generator)
            mask = t.zeros(1, 1, width, dtype=t.bool)
            mask[0, 0, [0, width // 2, width - 1]] = True
            segments.append(pack_dense_encoder_output_reference(dense, mask))

        actual = segments[0]
        reference_dense, reference_mask = densify_packed_encoder_output_reference(actual)
        for segment_index, new_segment in enumerate(segments[1:], start=1):
            actual = append_packed_encoder_segment(actual, new_segment)

            new_dense, new_mask = densify_packed_encoder_output_reference(new_segment)
            common_width = max(reference_dense.size(2), new_dense.size(2))
            reference_dense = F.pad(
                reference_dense,
                (0, 0, 0, common_width - reference_dense.size(2)),
            )
            reference_mask = F.pad(
                reference_mask,
                (0, common_width - reference_mask.size(2)),
            )
            new_dense = F.pad(
                new_dense,
                (0, 0, 0, common_width - new_dense.size(2)),
            )
            new_mask = F.pad(
                new_mask,
                (0, common_width - new_mask.size(2)),
            )
            reference_dense = t.cat((reference_dense, new_dense), dim=1)
            reference_mask = t.cat((reference_mask, new_mask), dim=1)
            reference = pack_dense_encoder_output_reference(
                reference_dense,
                reference_mask,
            )

            _assert_same_consumed_packed_state(actual, reference)
            assert actual.layout.n_segments == segment_index + 1
            assert actual.layout.feature_count == max(widths[: segment_index + 1])
            assert actual.layout.packed_count - int(actual.layout.valid_mask.sum()) < 32

        actual_repeat = repeat_packed_encoder_output(actual, 3)
        reference_repeat = pack_dense_encoder_output_reference(
            reference_dense.expand(3, -1, -1, -1).contiguous(),
            reference_mask.expand(3, -1, -1).contiguous(),
        )
        _assert_same_consumed_packed_state(actual_repeat, reference_repeat)

    def test_detach_and_move_keep_values_and_owners_atomic(self) -> None:
        dense = t.randn(1, 2, 3, 5, requires_grad=True)
        mask = t.tensor([[[True, False, True], [False, True, False]]])
        packed = pack_dense_encoder_output_reference(dense, mask)

        detached = packed.detach()
        assert not detached.values.requires_grad
        assert detached.values.data_ptr() == packed.values.data_ptr()
        assert detached.layout is packed.layout
        assert packed.to(t.device("cpu")) is packed

    def test_invalid_append_and_repeat_contracts_fail_loudly(self) -> None:
        values = t.randn(1, 1, 2, 4)
        nonempty = pack_dense_encoder_output_reference(
            values,
            t.ones(1, 1, 2, dtype=t.bool),
        )
        empty = pack_dense_encoder_output_reference(
            values,
            t.zeros(1, 1, 2, dtype=t.bool),
        )
        two_segments = pack_dense_encoder_output_reference(
            t.randn(1, 2, 2, 4),
            t.ones(1, 2, 2, dtype=t.bool),
        )
        batched = repeat_packed_encoder_output(nonempty, 2)

        with pytest.raises(ValueError, match="all-masked"):
            append_packed_encoder_segment(nonempty, empty)
        with pytest.raises(ValueError, match="exactly one new segment"):
            append_packed_encoder_segment(nonempty, two_segments)
        with pytest.raises(ValueError, match="batch size 1"):
            append_packed_encoder_segment(batched, nonempty)
        with pytest.raises(TypeError, match="must be an integer"):
            repeat_packed_encoder_output(nonempty, True)
        with pytest.raises(ValueError, match="must be positive"):
            repeat_packed_encoder_output(nonempty, 0)
        with pytest.raises(ValueError, match="batch-one source"):
            repeat_packed_encoder_output(batched, 2)
        with pytest.raises(ValueError, match="at least one row"):
            stack_packed_encoder_outputs(())
        with pytest.raises(ValueError, match="batch-one rows"):
            stack_packed_encoder_outputs((nonempty, batched))
        with pytest.raises(ValueError, match="one segment count"):
            stack_packed_encoder_outputs((nonempty, two_segments))
        with pytest.raises(TypeError, match="one values dtype"):
            stack_packed_encoder_outputs(
                (nonempty, PackedEncoderOutput(nonempty.values.double(), nonempty.layout))
            )
        with pytest.raises(ValueError, match="one encoder width"):
            stack_packed_encoder_outputs(
                (nonempty, PackedEncoderOutput(nonempty.values[..., :-1], nonempty.layout))
            )
        with pytest.raises(ValueError, match="one device"):
            stack_packed_encoder_outputs((nonempty, nonempty.to("meta")))

    @requires_cuda
    def test_move_transfers_every_tensor_and_roundtrips_exactly(self) -> None:
        dense = t.randn(1, 2, 4, 6)
        mask = t.tensor(
            [[[True, False, True, False], [False, True, False, True]]]
        )
        cpu = pack_dense_encoder_output_reference(dense, mask)
        cuda = cpu.to("cuda")
        for tensor in (
            cuda.values,
            cuda.layout.flat_indices,
            cuda.layout.valid_mask,
            cuda.layout.batch_indices,
            cuda.layout.segment_indices,
        ):
            assert tensor.device.type == "cuda"
        roundtrip = cuda.to("cpu")
        _assert_same_consumed_packed_state(roundtrip, cpu)
        assert t.equal(roundtrip.layout.flat_indices, cpu.layout.flat_indices)


def test_model_rejects_dense_or_redundantly_masked_precomputed_state() -> None:
    model = build_tiny_model(n_layer=2, xattn_frequency=1)
    decoder_ids = t.ones(1, 2, dtype=t.long)
    seg_map = t.zeros_like(decoder_ids)
    dense = t.zeros(1, 1, 1, model.encoder.config.n_embd)
    mask = t.ones(1, 1, 1, dtype=t.bool)

    with pytest.raises(TypeError, match="must be PackedEncoderOutput"):
        model(
            decoder_ids,
            None,
            seg_map=seg_map,
            encoder_out=dense,
            encoder_strict=True,
        )

    packed = pack_dense_encoder_output_reference(dense, mask)
    with pytest.raises(ValueError, match="enc_mask must be omitted"):
        model(
            decoder_ids,
            None,
            enc_mask=mask,
            seg_map=seg_map,
            encoder_out=packed,
            encoder_strict=True,
        )


@requires_cuda
def test_cuda_packed_second_pass_matches_removed_densify_repack_ste_reference() -> None:
    """Exercise the exact self-conditioning STE-to-loss path end to end."""
    t.manual_seed(1448)
    actual_model = build_tiny_model(n_layer=2, xattn_frequency=1).cuda()
    actual_model.init_weights()
    reference_model = copy.deepcopy(actual_model)

    feature_ids = t.tensor(
        [[[1, 5, 9, 0, 0], [2, 7, 3, 11, 0]]],
        dtype=t.long,
        device="cuda",
    )
    magnitudes = t.tensor(
        [[[0, 1, 2, 0, 0], [1, 0, 3, 2, 0]]],
        dtype=t.long,
        device="cuda",
    )
    mask = t.tensor(
        [[[True, True, True, False, False], [True, True, True, True, False]]],
        device="cuda",
    )
    sign = t.where(mask, 1.0, 0.0)
    actual_scores = t.randn(1, 2, 200, device="cuda", requires_grad=True)
    reference_scores = actual_scores.detach().clone().requires_grad_(True)
    decoder_ids = t.randint(0, 100, (1, 12), device="cuda")
    targets = t.randint(0, 100, (1, 12), device="cuda")
    seg_map = t.tensor([[0] * 6 + [1] * 6], device="cuda")

    actual_encoder = actual_model.encoder(
        feature_ids,
        magnitudes,
        mask,
        ste=SteInputs(
            scores=actual_scores,
            sign=sign,
            form=SteForm.DENSE,
        ),
    )
    reference_encoder = reference_model.encoder(
        feature_ids,
        magnitudes,
        mask,
        ste=SteInputs(
            scores=reference_scores,
            sign=sign,
            form=SteForm.DENSE,
        ),
    )
    reference_dense, reference_mask = densify_packed_encoder_output_reference(
        reference_encoder
    )
    reference_repacked = pack_dense_encoder_output_reference(
        reference_dense,
        reference_mask,
    )

    actual_loss = actual_model(
        decoder_ids,
        targets,
        seg_map=seg_map,
        encoder_out=actual_encoder,
        encoder_strict=True,
    )
    reference_loss = reference_model(
        decoder_ids,
        targets,
        seg_map=seg_map,
        encoder_out=reference_repacked,
        encoder_strict=True,
    )
    assert t.equal(actual_loss, reference_loss)

    actual_loss.backward()
    reference_loss.backward()
    assert actual_scores.grad is not None
    assert reference_scores.grad is not None
    assert bool(t.isfinite(actual_scores.grad).all())
    assert bool(actual_scores.grad.abs().sum() > 0)
    _assert_gradient_close(
        actual_scores.grad,
        reference_scores.grad,
        name="packed self-conditioning second-pass STE gradient",
    )


@requires_cuda
def test_cuda_append_and_repeat_match_removed_dense_cache_contract_exactly() -> None:
    """Pin cached-generation logits/loss after both packed state mutations."""
    t.manual_seed(2448)
    model = build_tiny_model(n_layer=2, xattn_frequency=1).cuda().eval()
    model.init_weights()

    old_ids = t.tensor(
        [[[1, 4, 0, 0, 0], [7, 2, 9, 0, 0]]],
        dtype=t.long,
        device="cuda",
    )
    old_magnitudes = t.tensor(
        [[[0, 2, 0, 0, 0], [3, 1, 2, 0, 0]]],
        dtype=t.long,
        device="cuda",
    )
    old_mask = old_ids != 0
    new_ids = t.tensor([[[5, 11, 3]]], dtype=t.long, device="cuda")
    new_magnitudes = t.tensor([[[1, 0, 3]]], dtype=t.long, device="cuda")
    new_mask = t.ones_like(new_ids, dtype=t.bool)

    with t.no_grad():
        old = model.encoder(old_ids, old_magnitudes, old_mask)
        new = model.encoder(new_ids, new_magnitudes, new_mask)
        actual_append = append_packed_encoder_segment(old, new)

        old_dense, old_dense_mask = densify_packed_encoder_output_reference(old)
        new_dense, new_dense_mask = densify_packed_encoder_output_reference(new)
        reference_append = pack_dense_encoder_output_reference(
            t.cat((old_dense, new_dense), dim=1),
            t.cat((old_dense_mask, new_dense_mask), dim=1),
        )
        _assert_same_consumed_packed_state(actual_append, reference_append)

        decoder_ids = t.randint(0, 100, (1, 12), device="cuda")
        targets = t.randint(0, 100, decoder_ids.shape, device="cuda")
        seg_map = t.tensor([[0] * 4 + [1] * 3 + [2] * 5], device="cuda")
        actual_logits = model(
            decoder_ids,
            None,
            seg_map=seg_map,
            encoder_out=actual_append,
            encoder_strict=True,
        )
        reference_logits = model(
            decoder_ids,
            None,
            seg_map=seg_map,
            encoder_out=reference_append,
            encoder_strict=True,
        )
        assert t.equal(actual_logits, reference_logits)
        actual_loss = model(
            decoder_ids,
            targets,
            seg_map=seg_map,
            encoder_out=actual_append,
            encoder_strict=True,
        )
        reference_loss = model(
            decoder_ids,
            targets,
            seg_map=seg_map,
            encoder_out=reference_append,
            encoder_strict=True,
        )
        assert t.equal(actual_loss, reference_loss)

        batch_size = 3
        actual_repeat = repeat_packed_encoder_output(actual_append, batch_size)
        reference_dense, reference_mask = densify_packed_encoder_output_reference(
            reference_append
        )
        reference_repeat = pack_dense_encoder_output_reference(
            reference_dense.expand(batch_size, -1, -1, -1).contiguous(),
            reference_mask.expand(batch_size, -1, -1).contiguous(),
        )
        _assert_same_consumed_packed_state(actual_repeat, reference_repeat)
        batched_ids = decoder_ids.expand(batch_size, -1).contiguous()
        batched_targets = targets.expand(batch_size, -1).contiguous()
        batched_seg_map = seg_map.expand(batch_size, -1).contiguous()
        actual_batched_logits = model(
            batched_ids,
            None,
            seg_map=batched_seg_map,
            encoder_out=actual_repeat,
            encoder_strict=True,
        )
        reference_batched_logits = model(
            batched_ids,
            None,
            seg_map=batched_seg_map,
            encoder_out=reference_repeat,
            encoder_strict=True,
        )
        assert t.equal(actual_batched_logits, reference_batched_logits)
        actual_batched_loss = model(
            batched_ids,
            batched_targets,
            seg_map=batched_seg_map,
            encoder_out=actual_repeat,
            encoder_strict=True,
        )
        reference_batched_loss = model(
            batched_ids,
            batched_targets,
            seg_map=batched_seg_map,
            encoder_out=reference_repeat,
            encoder_strict=True,
        )
        assert t.equal(actual_batched_loss, reference_batched_loss)


@requires_cuda
def test_model_validates_shared_cross_attention_routing_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hoisted validation is once-per-forward and output-exact."""
    model = build_tiny_model(n_layer=4, xattn_frequency=1).cuda().eval()
    decoder_ids = t.randint(0, 100, (1, 8), device="cuda")
    dense_encoder = t.randn(
        1,
        2,
        5,
        model.encoder.config.n_embd,
        device="cuda",
    )
    encoder_mask = t.ones(1, 2, 5, dtype=t.bool, device="cuda")
    encoder_out = pack_dense_encoder_output_reference(dense_encoder, encoder_mask)
    seg_map = t.tensor([[0] * 4 + [1] * 4], device="cuda")

    validate = cross_attention_module.validate_packed_cross_attention_inputs
    call_count = 0

    def counting_validate(*args, **kwargs) -> None:
        nonlocal call_count
        call_count += 1
        validate(*args, **kwargs)

    # The model imports the helper directly, while direct layer calls resolve
    # the cross_attention module global. Patch both so a regression to checked
    # per-layer calls is observable.
    monkeypatch.setattr(
        model_module,
        "validate_packed_cross_attention_inputs",
        counting_validate,
    )
    monkeypatch.setattr(
        cross_attention_module,
        "validate_packed_cross_attention_inputs",
        counting_validate,
    )

    with t.no_grad(), t.amp.autocast("cuda", dtype=t.bfloat16):
        logits = model(
            decoder_ids,
            None,
            seg_map=seg_map,
            encoder_out=encoder_out,
            encoder_strict=True,
            static_max_seg_size=4,
        )
    assert logits.shape == (1, 8, 100)
    assert call_count == 1

    prevalidated_forward = CrossAttentionLayer._forward_prevalidated

    def checked_per_layer(
        layer: CrossAttentionLayer,
        decoder_hidden: t.Tensor,
        encoder_kv: PackedEncoderOutput,
        packed_block_mask: BlockMask | None,
    ) -> t.Tensor:
        validate(
            decoder_hidden,
            encoder_kv,
            seg_map,
            static_max_seg_size=4,
        )
        return prevalidated_forward(
            layer,
            decoder_hidden,
            encoder_kv,
            packed_block_mask,
        )

    monkeypatch.setattr(
        CrossAttentionLayer,
        "_forward_prevalidated",
        checked_per_layer,
    )
    with t.no_grad(), t.amp.autocast("cuda", dtype=t.bfloat16):
        checked_logits = model(
            decoder_ids,
            None,
            seg_map=seg_map,
            encoder_out=encoder_out,
            encoder_strict=True,
            static_max_seg_size=4,
        )
    t.testing.assert_close(logits, checked_logits, rtol=0.0, atol=0.0)


@requires_cuda
class TestPackedEncoderDifferential:
    def test_synthetic_outputs_and_parameter_gradients_match_padded_reference(self) -> None:
        t.manual_seed(20260817)
        packed_encoder = _encoder()
        reference_encoder = _encoder()
        reference_encoder.load_state_dict(packed_encoder.state_dict())
        ids, magnitudes, mask = _encoder_inputs()

        with t.amp.autocast("cuda", dtype=t.bfloat16):
            packed = packed_encoder.forward_packed(ids, magnitudes, mask)
            dense, dense_mask = padded_encoder_reference(
                reference_encoder,
                ids,
                magnitudes,
                mask,
            )
            reference = pack_dense_encoder_output_reference(dense, dense_mask)

        real_count = int(mask.sum().item())
        _assert_close(
            packed.values[:, :real_count],
            reference.values[:, :real_count],
            name="packed encoder output",
        )

        probe = t.randn_like(packed.values[:, :real_count])
        (packed.values[:, :real_count] * probe).sum().backward()
        (reference.values[:, :real_count] * probe).sum().backward()
        for (packed_name, packed_parameter), (reference_name, reference_parameter) in zip(
            packed_encoder.named_parameters(),
            reference_encoder.named_parameters(),
            strict=True,
        ):
            assert packed_name == reference_name
            assert packed_parameter.grad is not None
            assert reference_parameter.grad is not None
            _assert_gradient_close(
                packed_parameter.grad,
                reference_parameter.grad,
                name=f"encoder gradient {packed_name}",
            )

    @pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
    def test_ste_score_gradients_match_padded_reference(self, form: SteForm) -> None:
        t.manual_seed(91)
        packed_encoder = _encoder()
        reference_encoder = _encoder()
        reference_encoder.load_state_dict(packed_encoder.state_dict())
        ids, magnitudes, mask = _encoder_inputs()
        sign = t.where(mask, t.randint(0, 2, mask.shape, device="cuda") * 2 - 1, 0).float()
        packed_scores = t.randn(2, 4, 128, device="cuda", requires_grad=True)
        reference_scores = packed_scores.detach().clone().requires_grad_(True)

        with t.amp.autocast("cuda", dtype=t.bfloat16):
            packed = packed_encoder.forward_packed(
                ids,
                magnitudes,
                mask,
                ste=SteInputs(scores=packed_scores, sign=sign, form=form),
            )
            dense, dense_mask = padded_encoder_reference(
                reference_encoder,
                ids,
                magnitudes,
                mask,
                ste=SteInputs(scores=reference_scores, sign=sign, form=form),
            )
            reference = pack_dense_encoder_output_reference(dense, dense_mask)

        real_count = int(mask.sum().item())
        probe = t.randn_like(packed.values[:, :real_count])
        (packed.values[:, :real_count] * probe).sum().backward()
        (reference.values[:, :real_count] * probe).sum().backward()
        assert packed_scores.grad is not None
        assert reference_scores.grad is not None
        _assert_gradient_close(
            packed_scores.grad,
            reference_scores.grad,
            name=f"{form} STE gradient",
        )

    def test_static_outer_compile_reuses_dynamic_packed_graph_across_lengths(self) -> None:
        t.manual_seed(7)
        encoder = t.compile(_encoder(), dynamic=False)
        ids = t.randint(0, 128, (2, 4, 13), device="cuda")
        magnitudes = t.randint(0, 8, ids.shape, device="cuda")
        flat_positions = t.arange(ids.numel(), device="cuda").reshape(ids.shape)
        active_counts = (24, 55, 88, 31, 63, 96)

        t._dynamo.reset()
        counters.clear()
        try:
            with t.no_grad(), t.amp.autocast("cuda", dtype=t.bfloat16):
                first = encoder(ids, magnitudes, flat_positions < active_counts[0])
            t.cuda.synchronize()
            first_graph_count = counters["stats"]["unique_graphs"]
            assert first_graph_count > 0

            for active_count in active_counts[1:]:
                with t.no_grad(), t.amp.autocast("cuda", dtype=t.bfloat16):
                    output = encoder(ids, magnitudes, flat_positions < active_count)
                assert output.values.shape[0] == first.values.shape[0] == 1
                assert output.values.shape[-1] == first.values.shape[-1]
                assert int(output.layout.valid_mask.sum()) == active_count
            t.cuda.synchronize()
            assert counters["stats"]["unique_graphs"] == first_graph_count
        finally:
            t._dynamo.reset()


@requires_cuda
class TestPackedCrossAttentionDifferential:
    def test_synthetic_outputs_and_all_gradients_match_independent_reference(self) -> None:
        t.manual_seed(314159)
        config = CrossAttentionLayerConfig(
            dec_n_embd=64,
            enc_n_embd=48,
            n_head=4,
            n_kv_head=2,
            head_dim=16,
        )
        packed_layer = CrossAttentionLayer(config).cuda()
        packed_layer.init_weights()
        reference_layer = CrossAttentionLayer(config).cuda()
        reference_layer.load_state_dict(packed_layer.state_dict())

        batch_size, sequence_length, n_segments, feature_count = 2, 17, 4, 11
        decoder = t.randn(
            batch_size, sequence_length, config.dec_n_embd, device="cuda", requires_grad=True
        )
        reference_decoder = decoder.detach().clone().requires_grad_(True)
        encoder = t.randn(
            batch_size,
            n_segments,
            feature_count,
            config.enc_n_embd,
            device="cuda",
            requires_grad=True,
        )
        reference_encoder = encoder.detach().clone().requires_grad_(True)
        mask = t.zeros(batch_size, n_segments, feature_count, dtype=t.bool, device="cuda")
        counts = ((3, 11, 1, 0), (7, 2, 9, 4))
        for batch_index, row in enumerate(counts):
            for segment_index, count in enumerate(row):
                mask[batch_index, segment_index, :count] = True
        seg_map = t.tensor(
            [[0] * 3 + [1] * 6 + [2] * 4 + [3] * 4, [0] * 5 + [1] * 2 + [2] * 7 + [3] * 3],
            device="cuda",
        )
        packed = pack_dense_encoder_output_reference(encoder, mask)
        block_mask = build_packed_cross_attention_block_mask(seg_map, packed.layout)

        with t.amp.autocast("cuda", dtype=t.bfloat16):
            actual = packed_layer(
                decoder,
                packed,
                seg_map,
                packed_block_mask=block_mask,
            )
            expected = _cross_attention_reference(
                reference_layer,
                reference_decoder,
                reference_encoder,
                mask,
                seg_map,
            )
        _assert_close(actual, expected, name="packed cross-attention output")

        probe = t.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        assert decoder.grad is not None and reference_decoder.grad is not None
        assert encoder.grad is not None and reference_encoder.grad is not None
        _assert_gradient_close(decoder.grad, reference_decoder.grad, name="decoder input gradient")
        _assert_gradient_close(encoder.grad, reference_encoder.grad, name="encoder input gradient")
        for (actual_name, actual_parameter), (expected_name, expected_parameter) in zip(
            packed_layer.named_parameters(), reference_layer.named_parameters(), strict=True
        ):
            assert actual_name == expected_name
            assert actual_parameter.grad is not None
            assert expected_parameter.grad is not None
            _assert_gradient_close(
                actual_parameter.grad,
                expected_parameter.grad,
                name=f"cross-attention gradient {actual_name}",
            )

    def test_cuda_requires_prebuilt_block_mask(self) -> None:
        config = CrossAttentionLayerConfig(64, 48, 4, 2, 16)
        layer = CrossAttentionLayer(config).cuda()
        decoder = t.randn(1, 4, 64, device="cuda")
        encoder = t.randn(1, 1, 3, 48, device="cuda")
        mask = t.ones(1, 1, 3, dtype=t.bool, device="cuda")
        seg_map = t.zeros(1, 4, dtype=t.long, device="cuda")
        packed = pack_dense_encoder_output_reference(encoder, mask)
        with pytest.raises(RuntimeError, match="precomputed packed_block_mask"):
            layer(decoder, packed, seg_map)

    def test_cuda_rejects_dense_encoder_output(self) -> None:
        config = CrossAttentionLayerConfig(64, 48, 4, 2, 16)
        layer = CrossAttentionLayer(config).cuda()
        decoder = t.randn(1, 4, 64, device="cuda")
        encoder = t.randn(1, 1, 3, 48, device="cuda")
        mask = t.ones(1, 1, 3, dtype=t.bool, device="cuda")
        seg_map = t.zeros(1, 4, dtype=t.long, device="cuda")
        block_mask = build_packed_cross_attention_block_mask(
            seg_map,
            pack_encoder_layout(mask),
        )
        with pytest.raises(TypeError, match="requires PackedEncoderOutput"):
            production_cross_attention_forward(
                layer,
                decoder,
                encoder,  # ty: ignore[invalid-argument-type]
                seg_map,
                packed_block_mask=block_mask,
            )


@requires_cuda
def test_pack_encoder_inputs_zeroes_tail_values() -> None:
    ids = t.arange(12, device="cuda").view(1, 3, 4)
    magnitudes = ids + 100
    mask = t.tensor([[[True, False, True, False]] * 3], device="cuda")
    packed = pack_encoder_inputs(ids, magnitudes, mask)
    real_count = int(mask.sum().item())
    assert packed.feature_ids[0, :real_count].tolist() == [0, 2, 4, 6, 8, 10]
    assert not packed.feature_ids[0, real_count:].any()
    assert not packed.magnitudes[0, real_count:].any()
