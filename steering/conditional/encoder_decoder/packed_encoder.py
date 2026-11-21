"""Globally packed feature-encoder tensor layout.

CUDA represents every real feature in a batch on one packed token axis.  The
``(batch, segment)`` owner tensors preserve isolation without retaining the
dense ``B x S x F`` allocation.  A single short tail pad keeps every FP8
linear's row count aligned; padding is never repeated per segment or batch row.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Self

import torch as t
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from steering.conditional.encoder_decoder import FP8_ALIGNMENT

PACKED_ENCODER_ALIGNMENT = 2 * FP8_ALIGNMENT
PACKED_ENCODER_BLOCK_SIZE = (128, 128)


@dataclass(frozen=True)
class PackedEncoderLayout:
    """Owner metadata for one globally packed encoder-token axis."""

    flat_indices: t.Tensor
    valid_mask: t.Tensor
    batch_indices: t.Tensor
    segment_indices: t.Tensor
    batch_size: int
    n_segments: int
    feature_count: int

    @property
    def packed_count(self) -> int:
        return self.flat_indices.numel()


@dataclass(frozen=True)
class PackedEncoderInputs:
    """Feature ids and magnitudes on one aligned packed axis."""

    feature_ids: t.Tensor
    magnitudes: t.Tensor
    layout: PackedEncoderLayout
    block_mask: BlockMask


@dataclass(frozen=True)
class PackedEncoderOutput:
    """Encoded feature values with their original batch/segment ownership."""

    values: t.Tensor
    layout: PackedEncoderLayout

    @property
    def is_cuda(self) -> bool:
        return self.values.is_cuda

    def detach(self) -> Self:
        """Detach encoded values while preserving their packed ownership metadata."""
        return detach_packed_encoder_output(self)

    def to(self, device: t.device | str) -> Self:
        """Move encoded values and every layout tensor to one device."""
        return move_packed_encoder_output(self, device)


def _validate_dense_mask(mask: t.Tensor) -> tuple[int, int, int]:
    if mask.ndim != 3:
        raise ValueError(f"encoder mask must have shape (B, S, F), got {tuple(mask.shape)}")
    if mask.dtype != t.bool:
        raise TypeError(f"encoder mask must have dtype bool, got {mask.dtype}")
    batch_size, n_segments, feature_count = mask.shape
    if batch_size < 1 or n_segments < 1 or feature_count < 1:
        raise ValueError(f"encoder mask dimensions must all be positive, got {tuple(mask.shape)}")
    return batch_size, n_segments, feature_count


@t.compiler.disable
def pack_encoder_layout(mask: t.Tensor) -> PackedEncoderLayout:
    batch_size, n_segments, feature_count = _validate_dense_mask(mask)
    real_indices = mask.reshape(-1).nonzero().flatten()
    real_count = real_indices.numel()
    packed_count = max(
        PACKED_ENCODER_ALIGNMENT,
        ((real_count + PACKED_ENCODER_ALIGNMENT - 1) // PACKED_ENCODER_ALIGNMENT)
        * PACKED_ENCODER_ALIGNMENT,
    )
    flat_indices = F.pad(real_indices, (0, packed_count - real_count))
    valid_mask = t.arange(packed_count, device=mask.device) < real_count
    slots_per_batch = n_segments * feature_count
    batch_indices = flat_indices // slots_per_batch
    segment_indices = (flat_indices % slots_per_batch) // feature_count
    return PackedEncoderLayout(
        flat_indices=flat_indices,
        valid_mask=valid_mask,
        batch_indices=batch_indices,
        segment_indices=segment_indices,
        batch_size=batch_size,
        n_segments=n_segments,
        feature_count=feature_count,
    )


def _pack_tensor(tensor: t.Tensor, layout: PackedEncoderLayout) -> t.Tensor:
    if tuple(tensor.shape[:3]) != (
        layout.batch_size,
        layout.n_segments,
        layout.feature_count,
    ):
        raise ValueError(
            "tensor leading dimensions disagree with packed encoder layout: "
            f"{tuple(tensor.shape[:3])} != "
            f"{(layout.batch_size, layout.n_segments, layout.feature_count)}"
        )
    trailing_shape = tensor.shape[3:]
    flat = tensor.reshape(-1, *trailing_shape)
    gathered = flat[layout.flat_indices]
    valid_shape = (layout.packed_count,) + (1,) * len(trailing_shape)
    return t.where(
        layout.valid_mask.reshape(valid_shape),
        gathered,
        t.zeros_like(gathered),
    ).unsqueeze(0)


def build_block_mask_from_dense_routing(
    routing: t.Tensor,
    *,
    mask_mod: Callable[..., t.Tensor],
) -> BlockMask:
    """Convert one exact token-routing matrix to FlexAttention sparse metadata.

    ``create_block_mask`` evaluates scalar ``mask_mod`` through nested vmaps.
    Packed routing already has vectorized owner tensors, so materializing the
    same boolean matrix directly avoids that Python/transform overhead while
    retaining the public ``BlockMask.from_kv_blocks`` construction contract.
    """
    if routing.ndim != 2:
        raise ValueError(f"routing must have shape (Q, KV), got {tuple(routing.shape)}")
    if routing.dtype != t.bool:
        raise TypeError(f"routing must have dtype bool, got {routing.dtype}")
    query_length, kv_length = routing.shape
    if query_length < 1 or kv_length < 1:
        raise ValueError(f"routing dimensions must be positive, got {tuple(routing.shape)}")

    query_block_size, kv_block_size = PACKED_ENCODER_BLOCK_SIZE
    padded = F.pad(
        routing,
        (
            0,
            (-kv_length) % kv_block_size,
            0,
            (-query_length) % query_block_size,
        ),
    )
    tiles = padded.view(
        1,
        1,
        padded.size(0) // query_block_size,
        query_block_size,
        padded.size(1) // kv_block_size,
        kv_block_size,
    ).permute(0, 1, 2, 4, 3, 5)
    tile_sums = tiles.sum(dim=(-2, -1), dtype=t.int32)
    full_tiles = tile_sums == query_block_size * kv_block_size
    partial_tiles = (tile_sums > 0) & ~full_tiles

    def ordered(dense: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        dense = dense.to(t.int32)
        counts = dense.sum(dim=-1).to(t.int32, memory_format=t.contiguous_format)
        indices = t.argsort(dense, dim=-1, descending=True, stable=True)
        return counts, indices.to(t.int32, memory_format=t.contiguous_format)

    partial_counts, partial_indices = ordered(partial_tiles)
    full_counts, full_indices = ordered(full_tiles)
    return BlockMask.from_kv_blocks(
        partial_counts,
        partial_indices,
        full_counts,
        full_indices,
        BLOCK_SIZE=PACKED_ENCODER_BLOCK_SIZE,
        mask_mod=mask_mod,
        seq_lengths=(query_length, kv_length),
    )


def build_packed_encoder_block_mask(layout: PackedEncoderLayout) -> BlockMask:
    """Build the segment-diagonal self-attention mask shared by every encoder layer."""
    # Capture only tensor state so the dynamic compile region does not inherit
    # the layout's Python integer fields as symbolic scalar guards.
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
        real_pair = (
            valid_mask[query_index]
            & valid_mask[kv_index]
            & (batch_indices[query_index] == batch_indices[kv_index])
            & (segment_indices[query_index] == segment_indices[kv_index])
        )
        # Tail queries attend only to themselves. Their values are exactly zero,
        # but a safe row avoids an all-masked softmax in either Flex backend.
        tail_diagonal = ~valid_mask[query_index] & (query_index == kv_index)
        return real_pair | tail_diagonal

    query_indices = t.arange(layout.packed_count, device=valid_mask.device)
    routing = (
        valid_mask[:, None]
        & valid_mask[None, :]
        & (batch_indices[:, None] == batch_indices[None, :])
        & (segment_indices[:, None] == segment_indices[None, :])
    ) | (~valid_mask[:, None] & (query_indices[:, None] == query_indices[None, :]))
    block_mask = build_block_mask_from_dense_routing(
        routing,
        mask_mod=mask_mod,
    )
    mark_packed_block_mask_dynamic(block_mask)
    return block_mask


def mark_packed_block_mask_dynamic(block_mask: BlockMask) -> None:
    # BlockMask's sparse row and column extents change when K crosses a
    # 128-token tile. Mark every non-singleton sparse extent so the first
    # production-sized trace is reusable across those crossings.
    for name in BlockMask._TENSOR_ATTRS:
        tensor = getattr(block_mask, name)
        if tensor is not None:
            dimensions = tuple(index for index, size in enumerate(tensor.shape) if size > 1)
            if dimensions:
                t._dynamo.mark_dynamic(tensor, dimensions)


def _mark_packed_inputs_dynamic(packed: PackedEncoderInputs) -> None:
    t._dynamo.mark_dynamic(packed.feature_ids, 1)
    t._dynamo.mark_dynamic(packed.magnitudes, 1)
    for tensor in (
        packed.layout.flat_indices,
        packed.layout.valid_mask,
        packed.layout.batch_indices,
        packed.layout.segment_indices,
    ):
        t._dynamo.mark_dynamic(tensor, 0)


@t.compiler.disable
def mark_packed_encoder_output_dynamic(packed: PackedEncoderOutput) -> None:
    t._dynamo.mark_dynamic(packed.values, 1)
    for tensor in (
        packed.layout.flat_indices,
        packed.layout.valid_mask,
        packed.layout.batch_indices,
        packed.layout.segment_indices,
    ):
        t._dynamo.mark_dynamic(tensor, 0)


def _validate_packed_output_shape(packed: PackedEncoderOutput, *, owner: str) -> None:
    layout = packed.layout
    if packed.values.ndim != 3 or packed.values.size(0) != 1:
        raise ValueError(
            f"{owner} values must have shape (1, K, C), got {tuple(packed.values.shape)}"
        )
    if packed.values.size(1) != layout.packed_count:
        raise ValueError(
            f"{owner} values and layout disagree on K: "
            f"{packed.values.size(1)} != {layout.packed_count}"
        )
    if packed.values.size(2) < 1:
        raise ValueError(f"{owner} encoder width must be positive")
    if layout.packed_count < PACKED_ENCODER_ALIGNMENT or (
        layout.packed_count % PACKED_ENCODER_ALIGNMENT
    ):
        raise ValueError(
            f"{owner} packed count must be a positive multiple of "
            f"{PACKED_ENCODER_ALIGNMENT}, got {layout.packed_count}"
        )
    if layout.batch_size < 1 or layout.n_segments < 1 or layout.feature_count < 1:
        raise ValueError(
            f"{owner} layout dimensions must all be positive, got "
            f"{(layout.batch_size, layout.n_segments, layout.feature_count)}"
        )
    metadata = {
        "flat_indices": (layout.flat_indices, t.int64),
        "valid_mask": (layout.valid_mask, t.bool),
        "batch_indices": (layout.batch_indices, t.int64),
        "segment_indices": (layout.segment_indices, t.int64),
    }
    for name, (tensor, dtype) in metadata.items():
        if tensor.ndim != 1 or tensor.numel() != layout.packed_count:
            raise ValueError(
                f"{owner} {name} must have shape ({layout.packed_count},), "
                f"got {tuple(tensor.shape)}"
            )
        if tensor.dtype != dtype:
            raise TypeError(f"{owner} {name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.device != packed.values.device:
            raise ValueError(
                f"{owner} {name} is on {tensor.device}, but values are on "
                f"{packed.values.device}"
            )


def _build_packed_encoder_output(
    *,
    real_values: t.Tensor,
    real_flat_indices: t.Tensor,
    real_batch_indices: t.Tensor,
    real_segment_indices: t.Tensor,
    batch_size: int,
    n_segments: int,
    feature_count: int,
) -> PackedEncoderOutput:
    """Build one canonical prefix-valid packed output with a single safe tail."""
    if real_values.ndim != 2:
        raise ValueError(
            f"real packed encoder values must have shape (R, C), got {tuple(real_values.shape)}"
        )
    real_count = real_values.size(0)
    for name, tensor in (
        ("flat_indices", real_flat_indices),
        ("batch_indices", real_batch_indices),
        ("segment_indices", real_segment_indices),
    ):
        if tensor.ndim != 1 or tensor.numel() != real_count:
            raise ValueError(
                f"real {name} must have shape ({real_count},), got {tuple(tensor.shape)}"
            )
        if tensor.dtype != t.int64:
            raise TypeError(f"real {name} must have dtype int64, got {tensor.dtype}")
        if tensor.device != real_values.device:
            raise ValueError(
                f"real {name} is on {tensor.device}, but values are on {real_values.device}"
            )

    packed_count = max(
        PACKED_ENCODER_ALIGNMENT,
        ((real_count + PACKED_ENCODER_ALIGNMENT - 1) // PACKED_ENCODER_ALIGNMENT)
        * PACKED_ENCODER_ALIGNMENT,
    )
    tail_count = packed_count - real_count
    layout = PackedEncoderLayout(
        flat_indices=F.pad(real_flat_indices, (0, tail_count)),
        valid_mask=t.arange(packed_count, device=real_values.device) < real_count,
        batch_indices=F.pad(real_batch_indices, (0, tail_count)),
        segment_indices=F.pad(real_segment_indices, (0, tail_count)),
        batch_size=batch_size,
        n_segments=n_segments,
        feature_count=feature_count,
    )
    packed = PackedEncoderOutput(
        values=F.pad(real_values, (0, 0, 0, tail_count)).unsqueeze(0),
        layout=layout,
    )
    mark_packed_encoder_output_dynamic(packed)
    return packed


@t.compiler.disable
def detach_packed_encoder_output(packed: PackedEncoderOutput) -> PackedEncoderOutput:
    """Return a detached packed state without changing any ownership metadata."""
    _validate_packed_output_shape(packed, owner="packed encoder output")
    detached = PackedEncoderOutput(values=packed.values.detach(), layout=packed.layout)
    mark_packed_encoder_output_dynamic(detached)
    return detached


@t.compiler.disable
def move_packed_encoder_output(
    packed: PackedEncoderOutput,
    device: t.device | str,
) -> PackedEncoderOutput:
    """Move a packed state atomically so values and owners cannot diverge."""
    _validate_packed_output_shape(packed, owner="packed encoder output")
    resolved_device = t.device(device)
    if packed.values.device == resolved_device:
        return packed
    layout = packed.layout
    moved = PackedEncoderOutput(
        values=packed.values.to(device=resolved_device),
        layout=PackedEncoderLayout(
            flat_indices=layout.flat_indices.to(device=resolved_device),
            valid_mask=layout.valid_mask.to(device=resolved_device),
            batch_indices=layout.batch_indices.to(device=resolved_device),
            segment_indices=layout.segment_indices.to(device=resolved_device),
            batch_size=layout.batch_size,
            n_segments=layout.n_segments,
            feature_count=layout.feature_count,
        ),
    )
    mark_packed_encoder_output_dynamic(moved)
    return moved


@t.compiler.disable
def append_packed_encoder_segment(
    existing: PackedEncoderOutput,
    new_segment: PackedEncoderOutput,
) -> PackedEncoderOutput:
    """Append one non-empty batch-one segment and rebuild canonical owner metadata."""
    _validate_packed_output_shape(existing, owner="existing packed encoder output")
    _validate_packed_output_shape(new_segment, owner="new packed encoder segment")
    old_layout = existing.layout
    new_layout = new_segment.layout
    if old_layout.batch_size != 1 or new_layout.batch_size != 1:
        raise ValueError(
            "packed encoder segment append requires batch size 1 for both states; "
            f"got {old_layout.batch_size} and {new_layout.batch_size}"
        )
    if new_layout.n_segments != 1:
        raise ValueError(
            "packed encoder segment append requires exactly one new segment; "
            f"got {new_layout.n_segments}"
        )
    if existing.values.device != new_segment.values.device:
        raise ValueError(
            "packed encoder segment append requires one device; got "
            f"{existing.values.device} and {new_segment.values.device}"
        )
    if existing.values.dtype != new_segment.values.dtype:
        raise TypeError(
            "packed encoder segment append requires one values dtype; got "
            f"{existing.values.dtype} and {new_segment.values.dtype}"
        )
    if existing.values.size(-1) != new_segment.values.size(-1):
        raise ValueError(
            "packed encoder segment append requires one encoder width; got "
            f"{existing.values.size(-1)} and {new_segment.values.size(-1)}"
        )

    old_valid = old_layout.valid_mask
    new_valid = new_layout.valid_mask
    if not bool(new_valid.any()):
        raise ValueError("cannot append an all-masked packed encoder segment")
    merged_feature_count = max(old_layout.feature_count, new_layout.feature_count)
    merged_n_segments = old_layout.n_segments + 1
    old_feature_indices = old_layout.flat_indices[old_valid] % old_layout.feature_count
    new_feature_indices = new_layout.flat_indices[new_valid] % new_layout.feature_count
    old_batch_indices = old_layout.batch_indices[old_valid]
    new_batch_indices = new_layout.batch_indices[new_valid]
    old_segment_indices = old_layout.segment_indices[old_valid]
    new_segment_indices = new_layout.segment_indices[new_valid] + old_layout.n_segments
    old_flat_indices = (
        (old_batch_indices * merged_n_segments + old_segment_indices)
        * merged_feature_count
        + old_feature_indices
    )
    new_flat_indices = (
        (new_batch_indices * merged_n_segments + new_segment_indices)
        * merged_feature_count
        + new_feature_indices
    )
    return _build_packed_encoder_output(
        real_values=t.cat(
            (existing.values[0, old_valid], new_segment.values[0, new_valid]),
            dim=0,
        ),
        real_flat_indices=t.cat((old_flat_indices, new_flat_indices)),
        real_batch_indices=t.cat((old_batch_indices, new_batch_indices)),
        real_segment_indices=t.cat((old_segment_indices, new_segment_indices)),
        batch_size=1,
        n_segments=merged_n_segments,
        feature_count=merged_feature_count,
    )


@t.compiler.disable
def repeat_packed_encoder_output(
    packed: PackedEncoderOutput,
    batch_size: int,
) -> PackedEncoderOutput:
    """Materialize one batch-one packed state for independent destination rows."""
    _validate_packed_output_shape(packed, owner="packed encoder output")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}")
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    layout = packed.layout
    if layout.batch_size != 1:
        raise ValueError(
            "packed encoder repetition requires a batch-one source; "
            f"got batch_size={layout.batch_size}"
        )
    if batch_size == 1:
        return packed

    valid = layout.valid_mask
    real_count = int(valid.sum().item())
    batch_indices = t.arange(
        batch_size,
        dtype=t.int64,
        device=packed.values.device,
    ).repeat_interleave(real_count)
    segment_indices = layout.segment_indices[valid].repeat(batch_size)
    feature_indices = (layout.flat_indices[valid] % layout.feature_count).repeat(batch_size)
    flat_indices = (
        (batch_indices * layout.n_segments + segment_indices) * layout.feature_count
        + feature_indices
    )
    return _build_packed_encoder_output(
        real_values=packed.values[0, valid].repeat(batch_size, 1),
        real_flat_indices=flat_indices,
        real_batch_indices=batch_indices,
        real_segment_indices=segment_indices,
        batch_size=batch_size,
        n_segments=layout.n_segments,
        feature_count=layout.feature_count,
    )


@t.compiler.disable
def stack_packed_encoder_outputs(
    rows: tuple[PackedEncoderOutput, ...],
) -> PackedEncoderOutput:
    """Stack independent batch-one states without padding their feature axes.

    Every row must describe the same number of logical segments and encoder
    width. Feature-axis capacities may differ because independently predicted
    self-conditioning segments can retain different numbers of features; the
    result uses their maximum capacity and remaps each real feature owner into
    that shared layout.
    """
    if not rows:
        raise ValueError("packed encoder stacking requires at least one row")
    for index, row in enumerate(rows):
        _validate_packed_output_shape(row, owner=f"packed encoder row {index}")

    first = rows[0]
    first_layout = first.layout
    expected_segments = first_layout.n_segments
    expected_width = first.values.size(-1)
    expected_device = first.values.device
    expected_dtype = first.values.dtype
    for index, row in enumerate(rows):
        layout = row.layout
        if layout.batch_size != 1:
            raise ValueError(
                "packed encoder stacking requires batch-one rows; "
                f"row {index} has batch_size={layout.batch_size}"
            )
        if layout.n_segments != expected_segments:
            raise ValueError(
                "packed encoder stacking requires one segment count; "
                f"row 0 has {expected_segments}, row {index} has {layout.n_segments}"
            )
        if row.values.size(-1) != expected_width:
            raise ValueError(
                "packed encoder stacking requires one encoder width; "
                f"row 0 has {expected_width}, row {index} has {row.values.size(-1)}"
            )
        if row.values.device != expected_device:
            raise ValueError(
                "packed encoder stacking requires one device; "
                f"row 0 is on {expected_device}, row {index} is on {row.values.device}"
            )
        if row.values.dtype != expected_dtype:
            raise TypeError(
                "packed encoder stacking requires one values dtype; "
                f"row 0 has {expected_dtype}, row {index} has {row.values.dtype}"
            )

    feature_count = max(row.layout.feature_count for row in rows)
    real_values: list[t.Tensor] = []
    real_flat_indices: list[t.Tensor] = []
    real_batch_indices: list[t.Tensor] = []
    real_segment_indices: list[t.Tensor] = []
    for batch_index, row in enumerate(rows):
        layout = row.layout
        valid = layout.valid_mask
        segment_indices = layout.segment_indices[valid]
        feature_indices = layout.flat_indices[valid] % layout.feature_count
        batch_indices = t.full_like(segment_indices, batch_index)
        real_values.append(row.values[0, valid])
        real_batch_indices.append(batch_indices)
        real_segment_indices.append(segment_indices)
        real_flat_indices.append(
            (batch_indices * expected_segments + segment_indices) * feature_count + feature_indices
        )

    return _build_packed_encoder_output(
        real_values=t.cat(real_values, dim=0),
        real_flat_indices=t.cat(real_flat_indices),
        real_batch_indices=t.cat(real_batch_indices),
        real_segment_indices=t.cat(real_segment_indices),
        batch_size=len(rows),
        n_segments=expected_segments,
        feature_count=feature_count,
    )


@t.compiler.disable
def pack_encoder_inputs(
    feature_ids: t.Tensor,
    magnitudes: t.Tensor,
    mask: t.Tensor,
) -> PackedEncoderInputs:
    """Pack dense encoder inputs outside Dynamo's graph and build one BlockMask."""
    if feature_ids.shape != magnitudes.shape or feature_ids.shape != mask.shape:
        raise ValueError(
            "feature_ids, magnitudes, and mask must have identical shapes, got "
            f"{tuple(feature_ids.shape)}, {tuple(magnitudes.shape)}, {tuple(mask.shape)}"
        )
    layout = pack_encoder_layout(mask)
    packed = PackedEncoderInputs(
        feature_ids=_pack_tensor(feature_ids, layout),
        magnitudes=_pack_tensor(magnitudes, layout),
        layout=layout,
        block_mask=build_packed_encoder_block_mask(layout),
    )
    _mark_packed_inputs_dynamic(packed)
    return packed
