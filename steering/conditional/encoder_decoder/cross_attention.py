"""
CrossAttentionLayer — decoder queries attend to globally packed encoder keys/values.

Operates at segment granularity while storing CUDA encoder outputs on one global
packed axis. Each decoder position attends only to features with the same
``(batch, segment)`` owner.

The CUDA path flattens decoder queries to ``(1, B*T)`` and consumes exactly the
globally packed encoder features. Padded SDPA lives only in the test suite as a
differential oracle; production cross-attention is packed and CUDA-only.

Uses QK normalization, GQA, and no bias in linear layers, following the same conventions
as CausalSelfAttention in nanochat/gpt.py.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from nanochat.gpt import Linear
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderLayout,
    PackedEncoderOutput,
    build_block_mask_from_dense_routing,
    mark_packed_block_mask_dynamic,
)

__all__ = [
    "CrossAttentionLayer",
    "CrossAttentionLayerConfig",
    "build_packed_cross_attention_block_mask",
    "norm",
    "validate_packed_cross_attention_inputs",
]

# FA4/CuTe backward cannot currently lower a genuinely dynamic BlockMask on
# Hopper (PyTorch 2.13: IR verification failure). Decoder self-attention retains
# FA4 independently; packed data-dependent attention uses fused Triton Flex.
_PACKED_FLEX_KERNEL_OPTIONS: FlexKernelOptions = {"BACKEND": "TRITON"}


@torch.compiler.disable(reason="BlockMask metadata must be marked dynamic outside Dynamo tracing")
def build_packed_cross_attention_block_mask(
    seg_map: torch.Tensor,
    layout: PackedEncoderLayout,
) -> BlockMask:
    """Build one data-dependent routing mask shared by all cross-attention layers.

    BlockMask construction reads batch data and materializes sparse metadata with
    data-dependent extents. This boundary is required because Dynamo forbids
    tracing ``mark_dynamic``; the declarations must happen before the mask enters
    the isolated dynamic attention graph.
    """
    batch_size, sequence_length = seg_map.shape
    if batch_size != layout.batch_size:
        raise ValueError(
            "seg_map and packed encoder batch sizes must match: "
            f"{batch_size} != {layout.batch_size}"
        )
    flat_seg_map = seg_map.reshape(-1)
    # Materialize query ownership before defining mask_mod. Deriving it inside
    # the closure as ``query_index // sequence_length`` captures a dynamic
    # Python SymInt, which FA4/CuTe cannot inline into its template.
    flat_batch_indices = (
        torch.arange(batch_size, device=seg_map.device)[:, None].expand_as(seg_map).reshape(-1)
    )
    valid_mask = layout.valid_mask
    encoder_batch_indices = layout.batch_indices
    encoder_segment_indices = layout.segment_indices

    def mask_mod(
        batch_index: torch.Tensor,
        head_index: torch.Tensor,
        query_index: torch.Tensor,
        kv_index: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        del head_index, kwargs
        return (
            valid_mask[kv_index]
            & (encoder_batch_indices[kv_index] == flat_batch_indices[query_index])
            & (encoder_segment_indices[kv_index] == flat_seg_map[query_index])
        )

    query_owners = flat_batch_indices * layout.n_segments + flat_seg_map
    encoder_owners = (
        encoder_batch_indices * layout.n_segments + encoder_segment_indices
    )
    routing = valid_mask[None, :] & (query_owners[:, None] == encoder_owners[None, :])
    block_mask = build_block_mask_from_dense_routing(
        routing,
        mask_mod=mask_mod,
    )
    mark_packed_block_mask_dynamic(block_mask)
    return block_mask


@dataclass
class CrossAttentionLayerConfig:
    """Configuration for a single CrossAttentionLayer.

    Note: This is distinct from CrossAttentionConfig in config.py, which holds
    high-level settings like frequency. This config holds the layer-level
    architectural parameters.
    """

    dec_n_embd: int  # decoder embedding dim (query source)
    enc_n_embd: int  # encoder embedding dim (KV source — encoder's native dim)
    n_head: int  # number of query heads
    n_kv_head: int  # number of KV heads (for GQA)
    head_dim: int  # dimension per head


def norm(x: torch.Tensor) -> torch.Tensor:
    """Purely functional RMS norm with no learnable params."""
    return F.rms_norm(x, (x.size(-1),))


def validate_packed_cross_attention_inputs(
    decoder_hidden: torch.Tensor,
    encoder_kv: PackedEncoderOutput,
    seg_map: torch.Tensor,
    static_max_seg_size: int | None = None,
) -> None:
    """Validate packed routing state once before a decoder-layer loop.

    ``seg_map`` and the packed encoder layout are shared by every
    cross-attention layer in one model forward. Keeping their data-dependent
    guards here lets the model validate once while direct layer callers retain
    the same fail-loud contract through :meth:`CrossAttentionLayer.forward`.
    """
    if not isinstance(encoder_kv, PackedEncoderOutput):
        raise TypeError(
            "cross-attention requires PackedEncoderOutput; "
            f"got {type(encoder_kv).__name__}"
        )
    if not decoder_hidden.is_cuda or not encoder_kv.is_cuda:
        raise RuntimeError("CrossAttentionLayer.forward requires CUDA packed state")
    if decoder_hidden.device != encoder_kv.values.device:
        raise ValueError(
            "decoder hidden state and packed encoder state must share a device: "
            f"{decoder_hidden.device} != {encoder_kv.values.device}"
        )
    batch_size, sequence_length, _ = decoder_hidden.size()
    n_segments = encoder_kv.layout.n_segments

    # --- Defense-in-depth contract assertions (eager-only) ---
    # These guard the upstream invariants documented in
    # ``docs/reference/architecture.md`` (seg_map contract section). They are
    # gated by ``torch.compiler.is_compiling()`` so they neither break compile
    # traces nor pay runtime cost in the compiled graph.
    if not torch.compiler.is_compiling():
        # P2-2a: seg_map non-negative range. Run this first so negative
        # entries surface with a focused message rather than tripping the
        # monotonicity guard below (which would describe the same row in
        # less actionable terms).
        assert seg_map.min() >= 0, (
            f"seg_map values must be >= 0; got seg_map.min()={int(seg_map.min())}"
        )
        # P2-2b: seg_map value-range bound. Out-of-range values cannot own
        # any packed encoder entry, which would silently remove
        # conditioning for those decoder positions.
        assert seg_map.max() < n_segments, (
            "seg_map values must be < n_seg "
            f"({n_segments}); got seg_map.max()={int(seg_map.max())}"
        )
        # P2-1 (manifest §4.18): seg_map monotonicity, scoped to non-zero
        # entries. Some callers reserve slot 0 as an empty/padding slot,
        # so a row can contain trailing or explicitly empty zero runs after
        # real feature segments. A non-zero value that decreases later in
        # the row violates the ordered, contiguous segment contract, so
        # reject it before attention routing.
        if sequence_length > 1:
            non_zero_seg_max = (
                torch.where(seg_map > 0, seg_map, torch.zeros_like(seg_map))
                .cummax(dim=1)
                .values
            )
            assert ((seg_map == 0) | (seg_map >= non_zero_seg_max)).all(), (
                "seg_map non-zero values must be monotonically "
                "non-decreasing within each row; a non-zero value "
                "followed by a smaller non-zero value violates the "
                "ordered-segment contract. "
                "(seg=0 is reserved for empty/padding slots and may repeat.) "
                "See docs/reference/architecture.md (seg_map contract)."
            )

    # --- Empty-slot routing convention (used by axbench-bpb) ---
    # Callers that need to route a token range to an empty encoder slot
    # while seg=0 is otherwise occupied by a real prefix segment SHOULD
    # prepend an empty conditioning slot at encoder index 0 (i.e., the
    # encoder's first segment has all-masked features) and route the
    # empty-slot token range to slot 0. Packed attention contributes zero
    # to decoder positions reading from a fully-masked slot, preserving
    # the monotonic-up-from-zero contract above. See
    # ``steering.eval.axbench_bpb.teacher_force._build_encdec_seg_map_and_encoder``
    # for the canonical implementation.

    # Packed-row manifests persist the raw largest observed
    # ``seg_token_end - seg_token_start``. Validate that semantic bound
    # directly so a stale manifest fails loudly before packed dispatch.
    if static_max_seg_size is not None:
        # Detect segment boundaries and forward-fill start positions via
        # cummax. Packed attention does not otherwise need query offsets.
        pos = torch.arange(sequence_length, device=seg_map.device).unsqueeze(0).expand(
            batch_size, -1
        )
        is_seg_start = torch.zeros(
            batch_size,
            sequence_length,
            dtype=torch.bool,
            device=seg_map.device,
        )
        is_seg_start[:, 0] = True
        is_seg_start[:, 1:] = seg_map[:, 1:] != seg_map[:, :-1]
        seg_start_pos = torch.where(is_seg_start, pos, torch.zeros_like(pos))
        seg_start_pos = seg_start_pos.cummax(dim=1).values
        required_max_seg_size = (pos - seg_start_pos).amax() + 1
        assert static_max_seg_size > 0, (
            f"static_max_seg_size must be > 0; got {static_max_seg_size}"
        )
        if torch.compiler.is_compiling():
            # Sync-free and graph-safe, matching FeatureEncoder's compiled
            # fail-loud guard. A Python/.item() assertion here would either
            # graph-break or disappear under compilation, recreating the
            # same-bucket stale-manifest corruption this check prevents.
            torch._assert_async(
                required_max_seg_size <= static_max_seg_size,
                "static_max_seg_size is smaller than required "
                "(= pos_in_seg.amax() + 1); packed-row manifests store "
                "the raw observed segment maximum.",
            )
        else:
            # Compute the dynamic value strictly so we can compare bounds.
            # ``int(...)`` is fine here because we are eager-only.
            required = int(required_max_seg_size.item())
            assert required <= static_max_seg_size, (
                f"static_max_seg_size={static_max_seg_size} is smaller than "
                f"required={required} (= pos_in_seg.amax() + 1); "
                "the batch violates its packed-row manifest bound."
            )


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer where decoder queries attend to encoder keys/values.

    Input shapes:
        decoder_hidden: (B, T, dec_n_embd) — decoder hidden states (pre-normed by caller)
        encoder_kv:     PackedEncoderOutput
        seg_map:        (B, T) int64 — maps each position to segment index

    Output: (B, T, dec_n_embd)

    The fused c_kv projection maps directly from enc_n_embd to KV heads, eliminating
    the need for a separate encoder output projection to dec_n_embd.

    Design: CUDA keeps queries packed as (B, T), flattens encoder K/V, and routes
    attention with a data-dependent FlexAttention block mask. Production has no
    dense or CPU fallback.
    """

    def __init__(self, config: CrossAttentionLayerConfig):
        super().__init__()
        self.config = config
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.dec_n_embd = config.dec_n_embd
        self.enc_n_embd = config.enc_n_embd

        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0, (
            f"GQA invariant violated: n_kv_head must be <= n_head and divide n_head; "
            f"got n_head={self.n_head}, n_kv_head={self.n_kv_head}"
        )

        self.kv_dim = self.n_kv_head * self.head_dim

        # Query projection from decoder hidden states
        self.c_q = Linear(self.dec_n_embd, self.n_head * self.head_dim, bias=False)
        # Fused key+value projection from encoder outputs (enc_n_embd → 2 * kv_dim)
        self.c_kv = Linear(self.enc_n_embd, 2 * self.kv_dim, bias=False)
        # Output projection back to decoder dim
        self.c_proj = Linear(self.n_head * self.head_dim, self.dec_n_embd, bias=False)

    @torch.no_grad()
    def init_weights(self):
        """Initialize weights following the same conventions as nanochat/gpt.py.

        c_q:    Uniform(+-sqrt(3/dec_n_embd))
        c_kv:   Uniform(+-sqrt(3/enc_n_embd))
        c_proj: Uniform(+-sqrt(3/(n_head * head_dim)))

        The product ``n_head * head_dim`` equals ``dec_n_embd`` in production
        because ``model.py:153`` derives ``xattn_head_dim = dec_n_embd // xattn_n_head``;
        the formula is therefore mathematically equivalent to
        ``Uniform(+-sqrt(3/dec_n_embd))`` (matching c_q's bound) but is written
        in fan-in form to mirror the per-projection convention.

        Note: c_proj was previously zero-initialized for residual-friendly startup,
        but all-zero weights cause FP8 tensorwise scaling to degenerate (amax=0 →
        extreme scale factors → inf gradients on the first backward pass).
        Using the same uniform init as the other projections avoids this while
        keeping the initialization magnitude appropriately scaled.
        """
        s_q = (3.0 / self.dec_n_embd) ** 0.5
        s_kv = (3.0 / self.enc_n_embd) ** 0.5
        s_proj = (3.0 / (self.n_head * self.head_dim)) ** 0.5
        torch.nn.init.uniform_(self.c_q.weight, -s_q, s_q)
        torch.nn.init.uniform_(self.c_kv.weight, -s_kv, s_kv)
        torch.nn.init.uniform_(self.c_proj.weight, -s_proj, s_proj)

    def forward(
        self,
        decoder_hidden: torch.Tensor,
        encoder_kv: PackedEncoderOutput,
        seg_map: torch.Tensor,
        static_max_seg_size: int | None = None,
        packed_block_mask: BlockMask | None = None,
    ) -> torch.Tensor:
        """
        Args:
            decoder_hidden:    (B, T, dec_n_embd) — pre-normed by caller
            encoder_kv:        PackedEncoderOutput
            seg_map:           (B, T) int64 — maps each position to segment index
            static_max_seg_size: raw semantic upper bound for
                                 pos_in_seg.amax() + 1. If provided, validate it
                                 against the batch.
            packed_block_mask:  CUDA FlexAttention mask shared across all
                                cross-attention layers in one model forward.

        Returns:
            (B, T, dec_n_embd)
        """
        validate_packed_cross_attention_inputs(
            decoder_hidden,
            encoder_kv,
            seg_map,
            static_max_seg_size=static_max_seg_size,
        )
        return self._forward_prevalidated(
            decoder_hidden,
            encoder_kv,
            packed_block_mask,
        )

    def _forward_prevalidated(
        self,
        decoder_hidden: torch.Tensor,
        encoder_kv: PackedEncoderOutput,
        packed_block_mask: BlockMask | None,
    ) -> torch.Tensor:
        """Run packed attention after shared routing state was validated."""
        if packed_block_mask is None:
            raise RuntimeError(
                "CUDA cross-attention requires a precomputed packed_block_mask; "
                "build it once before the decoder-layer loop"
            )
        return _dispatch_packed_cross_attention_dynamic(
            self,
            decoder_hidden,
            encoder_kv,
            packed_block_mask,
        )

    def _packed_cross_attention_dynamic(
        self,
        decoder_hidden: torch.Tensor,
        encoder_kv: PackedEncoderOutput,
        packed_block_mask: BlockMask,
    ) -> torch.Tensor:
        """Project packed Q/K/V, attend, and project the packed output."""
        batch_size, sequence_length, width = decoder_hidden.shape
        packed_count = encoder_kv.layout.packed_count

        q = self.c_q(decoder_hidden.reshape(1, batch_size * sequence_length, width)).view(
            1,
            batch_size * sequence_length,
            self.n_head,
            self.head_dim,
        )
        kv = self.c_kv(encoder_kv.values)
        k, v = kv.split([self.kv_dim, self.kv_dim], dim=-1)
        k = k.view(
            1,
            packed_count,
            self.n_kv_head,
            self.head_dim,
        )
        v = v.view(
            1,
            packed_count,
            self.n_kv_head,
            self.head_dim,
        )

        q = norm(q).transpose(1, 2).nan_to_num()
        k = norm(k).transpose(1, 2).nan_to_num()
        v = v.transpose(1, 2)
        enable_gqa = self.n_head != self.n_kv_head
        y = flex_attention(
            q,
            k,
            v,
            block_mask=packed_block_mask,
            enable_gqa=enable_gqa,
            kernel_options=_PACKED_FLEX_KERNEL_OPTIONS,
        )
        assert isinstance(y, torch.Tensor)
        y = y.nan_to_num()
        y = y.transpose(1, 2).reshape(batch_size, sequence_length, width)
        return self.c_proj(y)


# Compile the complete projection+attention implementation directly. The K/V
# token extent is dynamic; the projection weights and channel dimensions are
# static. Q/O do not inherently need dynamic shapes, but remain in this single
# inner call until a repeated accelerator benchmark establishes that splitting
# them is faster. The non-recursive disable wrapper graph-breaks the outer static
# model without disabling this inner compiled call.
_compiled_packed_cross_attention_dynamic = torch.compile(
    CrossAttentionLayer._packed_cross_attention_dynamic,
    dynamic=True,
)


@torch.compiler.disable(
    recursive=False,
    reason="dispatch dynamic packed attention outside the static model graph",
)
def _dispatch_packed_cross_attention_dynamic(
    layer: CrossAttentionLayer,
    decoder_hidden: torch.Tensor,
    encoder_kv: PackedEncoderOutput,
    packed_block_mask: BlockMask,
) -> torch.Tensor:
    """Graph-break the static caller, then enter the compiled dynamic method."""
    return _compiled_packed_cross_attention_dynamic(
        layer,
        decoder_hidden,
        encoder_kv,
        packed_block_mask,
    )
