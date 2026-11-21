"""
FeatureEncoder — bidirectional transformer that encodes SAE feature IDs
with additive magnitude embeddings for cross-attention conditioning.

Features within a segment are an unordered set (bag-of-features), so no
positional embeddings are used. Attention is bidirectional (no causal mask,
no RoPE). Output stays at encoder's native embedding dimension; the
cross-attention layer handles the dimension change to decoder space.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    BlockMask,
    FlexKernelOptions,
    flex_attention,
)

from errors import FeatureVocabError, SelfCondTrainingError
from nanochat.common import COMPUTE_DTYPE
from nanochat.gpt import Linear, has_ve
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderInputs,
    PackedEncoderOutput,
    mark_packed_encoder_output_dynamic,
    pack_encoder_inputs,
)
from steering.conditional.encoder_decoder.typed_module_list import TypedModuleList
from steering.conditional.feature_vocab import FeatureVocab

# FA4/CuTe backward cannot currently lower the encoder's segment-diagonal
# data-dependent mask on Hopper (PyTorch 2.13: IR verification failure).  Use
# the fused Triton Flex kernel for this mask. Decoder self-attention retains
# FA4 independently; packed cross-attention also uses Triton for the same
# dynamic-mask-backward constraint.
_PACKED_ENCODER_FLEX_KERNEL_OPTIONS: FlexKernelOptions = {"BACKEND": "TRITON"}


def _average_row_norm(matrix: torch.Tensor) -> torch.Tensor:
    """Mean L2 row norm, in fp32.

    Deliberately duplicates ``sae_embedding_init.average_row_norm`` rather than importing
    it: that module pulls in ``sae_lens``, and the encoder must not drag a heavy optional
    dependency into every import of the model. The two expressions are pinned bit-for-bit
    identical by ``test_sae_embedding_init_vocab.py::TestNormHelperAgreesWithSaeInit``.
    """
    return matrix.detach().float().norm(dim=1).mean()


def norm(x):
    """Purely functional RMSNorm with no learnable params (same as nanochat/gpt.py).

    NOTE: This duplicates ``nanochat.gpt.norm``. The duplication is intentional —
    ``nanochat/`` is vendored read-only (per CODEOWNERS / project memory), so we
    cannot add a public surface point that exports this helper. Audit
    ``2026-05-05`` consolidation manifest §2.1 adjudicated this as
    "DOCUMENTED INTENTIONAL DRIFT". If the RMSNorm semantics ever change, the
    change must be made in both places.
    """
    return F.rms_norm(x, (x.size(-1),))


@dataclass
class FeatureEncoderConfig:
    n_sae_features: int  # vocabulary size for feature IDs
    num_buckets: int  # number of magnitude discretization levels
    n_embd: int  # encoder embedding dimension
    n_layer: int  # number of self-attention layers (0 = embed + norm only)
    n_head: int
    n_kv_head: int
    head_dim: int
    ve_gate_channels: int = 32


class SteForm(StrEnum):
    """Feature-score carrier form used by self-conditioning's STE edge."""

    DENSE = "dense"
    SELECTED = "selected"


@dataclass(frozen=True)
class SteInputs:
    """Differentiable scores and per-column signs for a value-zero STE edge.

    Shapes use the encoder's input conventions: ``B`` batch rows, ``S``
    segments (slots), ``F`` dense conditioning columns per slot, and ``V`` is
    the feature embedding table's row count: ``config.n_sae_features`` for an
    uncompacted encoder, or ``feature_vocab.size`` for a compacted one (scores
    are then emitted in compact space). CUDA packing gathers the real ``F``
    positions and applies the same aligned tail to ``sign`` internally.
    """

    scores: torch.Tensor
    """float32[B, S, V] — sigmoided per-slot prediction scores, aligned so row
    ``[:, j]`` holds the SOURCE segment ``j-1``'s scores that produced slot
    ``j``'s conditioning. Must carry ``requires_grad`` (the whole point is the
    gradient edge); compacted scores are indexed by compact row, and rows for
    non-predicted slots are zeros and unused."""

    sign: torch.Tensor
    """float32[B, S, F] with values in {-1.0, 0.0, +1.0} — per-column STE
    polarity: ``+1`` positive conditioning columns (tied to ``s``), ``-1``
    negative columns (tied to ``1 - s``, i.e. the gradient is negated), ``0``
    real/doc-start slots, padding columns, and padding slots (no STE edge)."""

    form: SteForm
    """DENSE routes ``∂L/∂scores`` to EVERY embedding-table row via a soft
    mixture; SELECTED routes only to each column's own selected compact/full
    row after the encoder's original-ID remap."""


class EncoderSelfAttention(nn.Module):
    """
    Bidirectional multi-head self-attention (no causal mask, no RoPE).

    Supports GQA (n_kv_head <= n_head) and padding masks.
    QK normalization follows the same pattern as nanochat/gpt.py.
    """

    def __init__(self, config: FeatureEncoderConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = config.head_dim
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0

        self.q_dim = self.n_head * self.head_dim
        self.kv_dim = self.n_kv_head * self.head_dim
        self.c_qkv = Linear(self.n_embd, self.q_dim + 2 * self.kv_dim, bias=False)
        self.c_proj = Linear(self.q_dim, self.n_embd, bias=False)
        self.ve_gate_channels = config.ve_gate_channels
        self.ve_gate = (
            Linear(config.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve=None, mask=None, packed_block_mask: BlockMask | None = None):
        """
        Args:
            x: (B, Nf, C) input tensor where Nf is the number of features
            ve: (B, Nf, kv_dim) value embedding tensor, or None
            mask: (B, Nf) bool tensor, True=real, False=padding
        Returns:
            (B, Nf, C) output tensor
        """
        B, Nf, C = x.size()

        # Fused QKV projection + split
        qkv = self.c_qkv(x)
        q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, Nf, self.n_head, self.head_dim)
        k = k.view(B, Nf, self.n_kv_head, self.head_dim)
        v = v.view(B, Nf, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer-style): mix in value embedding with input-dependent gate
        if ve is not None:
            ve = ve.to(COMPUTE_DTYPE).view(B, Nf, self.n_kv_head, self.head_dim)
            ve_gate = cast(Linear, self.ve_gate)
            gate = 2 * torch.sigmoid(ve_gate(x[..., : self.ve_gate_channels]))  # (B, Nf, n_kv_head)
            v = v + gate.unsqueeze(-1) * ve

        q = q.transpose(1, 2)  # (B, n_head, Nf, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_head, Nf, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_head, Nf, head_dim)

        # QK normalization
        q, k = norm(q), norm(k)

        if packed_block_mask is not None:
            enable_gqa = self.n_kv_head != self.n_head
            y = flex_attention(
                q,
                k,
                v,
                block_mask=packed_block_mask,
                enable_gqa=enable_gqa,
                kernel_options=_PACKED_ENCODER_FLEX_KERNEL_OPTIONS,
            )
            assert isinstance(y, torch.Tensor)
            y = y.nan_to_num()
            y = y.transpose(1, 2).contiguous().view(B, Nf, -1)
            return self.c_proj(y)

        # Padded SDPA reference: repeat each KV head for its query heads.
        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)

        # Build attention mask from padding mask
        attn_mask = None
        if mask is not None:
            # mask: (B, Nf) bool, True=real -> convert to SDPA format
            # SDPA expects: (B, 1, 1, Nf_key) — broadcast over heads and query positions
            attn_mask = mask[:, None, None, :]  # (B, 1, 1, Nf)

        # Bidirectional attention (is_causal=False)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=False
        )  # (B, n_head, Nf, head_dim)
        # All-masked segments (padding only) produce NaN from softmax(all -inf); zero them out
        y = y.nan_to_num()

        # Re-assemble heads and project
        y = y.transpose(1, 2).contiguous().view(B, Nf, -1)  # (B, Nf, n_head * head_dim)
        y = self.c_proj(y)
        return y


class EncoderMLP(nn.Module):
    """Feed-forward network with relu^2 activation (same pattern as nanochat/gpt.py MLP)."""

    def __init__(self, config: FeatureEncoderConfig):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class EncoderBlock(nn.Module):
    """Pre-norm residual block: x = x + attn(norm(x), mask), x = x + mlp(norm(x))."""

    def __init__(self, config: FeatureEncoderConfig, layer_idx: int):
        super().__init__()
        self.attn = EncoderSelfAttention(config, layer_idx)
        self.mlp = EncoderMLP(config)

    def forward(self, x, ve=None, mask=None, packed_block_mask: BlockMask | None = None):
        x = x + self.attn(
            norm(x),
            ve=ve,
            mask=mask,
            packed_block_mask=packed_block_mask,
        )
        x = x + self.mlp(norm(x))
        x = x.to(
            COMPUTE_DTYPE
        )  # Float8Linear may output float32; keep activations in compute dtype
        return x


class FeatureEncoder(nn.Module):
    """
    Bidirectional transformer encoder for SAE features.

    Embeds feature IDs and magnitude buckets (additively), processes through
    self-attention layers, and outputs at the encoder's native embedding dimension.
    """

    def __init__(
        self,
        config: FeatureEncoderConfig,
        feature_vocab: FeatureVocab | None = None,
    ):
        super().__init__()
        if config.n_layer < 1:
            raise ValueError(
                f"FeatureEncoder requires at least 1 layer (got n_layer={config.n_layer}). "
                "Self-attention blocks handle dtype management (COMPUTE_DTYPE casting) "
                "required for cross-attention compatibility."
            )
        self.config = config

        # The compacted feature dictionary (issue #695). ``None`` ⇒ no compaction:
        # the tables are full-width and ``_to_compact`` is a literal early-return, so
        # the whole module is byte-identical to the pre-#695 code (invariant I5).
        #
        # torch.compile: this is set at CONSTRUCTION and never mutated, so Dynamo
        # specialises the ``is None`` branch once at trace time. Contrast
        # ``EncoderDecoderGPT.feature_mask``, which is set post-init and therefore
        # carries the H10 set-before-first-forward contract.
        # An IDENTITY vocab is not a compaction. The guard lives HERE, on the class that sizes
        # the tables and registers the vocab buffers, rather than on ``EncoderDecoderGPT``:
        # that only protected callers who came in through the model, while anything building an
        # encoder directly still produced a checkpoint NO load path accepts
        # (``feature_vocab_from_state_dict`` RAISES on identity buffers, by design — no writer
        # is supposed to be able to emit them).
        #
        # The load-bearing invariant is that a non-``None`` FeatureVocab ALWAYS means GENUINELY
        # COMPACTED (V < n_sae_features); every consumer reads ``feature_vocab is not None`` as
        # "this model was compacted". Both the mask-only and cluster-aware production
        # constructors normalize an identity to ``None``, so no config path reaches this — but
        # the constructor is public, and "unreachable today" is exactly how a trap stays armed.
        if feature_vocab is not None and feature_vocab.is_identity:
            raise FeatureVocabError(
                f"feature_vocab is the IDENTITY vocabulary (V == n_sae_features == "
                f"{config.n_sae_features}). That is not a compaction.\n"
                "The load-bearing invariant is that a non-None FeatureVocab ALWAYS means "
                "GENUINELY COMPACTED (V < n_sae_features): every consumer reads "
                "`feature_vocab is not None` as 'this model was compacted'. An identity would be "
                "simultaneously 'compacted' and full-width, and would poison all of them.\n"
                "Building it would also register the two vocab buffers, producing a "
                "checkpoint that NOTHING can load: feature_vocab_from_state_dict refuses "
                "identity buffers, because no writer is supposed to be able to emit them.\n"
                "Fix: pass feature_vocab=None for a full-width encoder. The mask-only and "
                "cluster-aware production constructors already normalize an all-allowing "
                "effective universe to None, so this vocab was constructed directly rather "
                "than resolved from a config."
            )

        self.feature_vocab = feature_vocab
        vocab_size = config.n_sae_features if feature_vocab is None else feature_vocab.size

        # Captured by init_weights(); the SAE-init rescale target. NOT model state — it
        # is init-time only (SAE init runs at cold start, never on resume), so it is
        # deliberately a plain attribute rather than a buffer: it must not enter the
        # state_dict, where it would change the checkpoint format for no reason.
        self._feature_embed_full_avg_row_norm: torch.Tensor | None = None

        self.feature_embed = nn.Embedding(vocab_size, config.n_embd)
        self.magnitude_embed = nn.Embedding(config.num_buckets, config.n_embd)
        self.blocks: TypedModuleList[EncoderBlock] = TypedModuleList(
            [EncoderBlock(config, i) for i in range(config.n_layer)],
            module_type=EncoderBlock,
            item_label="FeatureEncoder block",
        )

        # Value embeddings (ResFormer-style, feature-ID-indexed)
        kv_dim = config.n_kv_head * config.head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )

        if feature_vocab is not None:
            # THE durable record, and the ONLY vocab state the encoder keeps. Both are
            # persistent, so load_state_dict restores them; a loaded model is self-describing
            # (n_sae_features is NOT recoverable from a compacted table's row count). The
            # compact-id map is NOT stored — ``_to_compact`` derives it per-forward from
            # ``feature_vocab_ids`` via ``searchsorted``. That is deliberate: a derived tensor
            # stored SEPARATELY and kept coherent across to_empty / device-moves / loads was the
            # single root cause of this feature's entire fix-to-fix history, so there is nothing
            # to keep coherent and nothing to desync.
            self.register_buffer(
                "feature_vocab_ids",
                torch.zeros(feature_vocab.size, dtype=torch.int64),
                persistent=True,
            )
            self.register_buffer(
                "feature_vocab_n_sae",
                torch.zeros((), dtype=torch.int64),
                persistent=True,
            )
            # Precomputed at construction so torch._assert_async can use it as a
            # trace-time constant. It cannot name the offending ids (that would need
            # a host sync on every eager CUDA forward), so it names the shape of the
            # problem and the remedy instead. CUDA's async assertion runtime rejects
            # messages of 255 bytes or more even when the condition is true.
            self._vocab_oov_msg = (
                "encoder feature id outside the compacted vocabulary "
                f"(V={feature_vocab.size}, n_sae={config.n_sae_features}). Configure "
                "the checkpoint's feature_mask/feature_clusters; inspect via "
                "scripts.analysis.dump_feature_mask."
            )
            assert len(self._vocab_oov_msg.encode("utf-8")) < 255, (
                "torch._assert_async CUDA messages must be shorter than 255 bytes"
            )

    @torch.no_grad()
    def init_vocab_buffers(self):
        """Materialize the two persistent vocab buffers from the Python ``FeatureVocab``.

        Called from ``init_weights()`` and ``EncoderDecoderGPT.init_buffers()``, mirroring how
        both already re-derive ``cos``/``sin``. Required because construction happens on the
        ``meta`` device, so the buffers created in ``__init__`` carry no storage until
        ``to_empty()``.

        This is a FRESH-PATH materializer. It writes what the configured vocab says, so on the
        LOAD path it MUST run BEFORE ``load_state_dict`` — which then overwrites these buffers
        with the checkpoint's (``assign=True``). Production always orders it that way
        (``checkpoint_manager.build_encdec_model``: ``to_empty`` → ``init_buffers`` → ``load``).
        Do NOT call it AFTER a load: it would overwrite the checkpoint's vocabulary with the
        configured one, and ``assert_vocab_consistent()`` would then have nothing left to catch
        the swap with. There is no derived map to re-derive — ``_to_compact`` computes the
        remap on the fly from ``feature_vocab_ids`` — so this is the whole of it.
        """
        if self.feature_vocab is None:
            return
        device = self.feature_embed.weight.device
        self.feature_vocab_ids = self.feature_vocab.new_to_old_tensor(device=device)
        self.feature_vocab_n_sae = torch.tensor(
            self.feature_vocab.n_sae_features, dtype=torch.int64, device=device
        )

    @torch.no_grad()
    def _init_feature_indexed_(self, weight, init_fn):
        """Initialise a feature-id-indexed table, preserving the reference RNG stream.

        SLICE-INIT (spec §1.4): when compacted, draw the **full** ``(n_sae, ...)``
        table — consuming the IDENTICAL number of randoms as the uncompacted run —
        then keep only the vocab's rows. ``feature_embed`` is drawn FIRST in
        ``init_weights``, so restoring its draw count keeps ``magnitude_embed``, the
        blocks, ``value_embeds``, the VE gates and cross-attention all **bitwise
        identical** to the reference.

        Without this, a compacted run diverges from the reference at step 0 and
        "byte-exact vs the reference" would only hold on the no-mask path.

        PRECONDITION — easy to violate silently:
            ``nn.Embedding.reset_parameters()`` draws ``numel()`` randoms at
            CONSTRUCTION. On a REAL device a compact table therefore draws fewer than a
            full one, and the RNG stream diverges *before* ``init_weights()`` is even
            reached — destroying byte-exactness no matter how correct this function is.

            Production is safe two ways: models are built on the ``meta`` device (which
            draws nothing) and the training script reseeds immediately before the
            cold-start ``init_weights()`` call. Pinned by
            ``tests/steering/test_encoder_feature_vocab.py::TestMetaConstructionContract``.

        RETURNS the FULL ``(n_sae, d)`` table. ``init_weights`` derives the SAE-init rescale
        target from it — a V-row subset's mean row-norm is only a *sample estimate* of the
        full table's (measured 2.1e-04 relative at V=35,275 of 65,536), so rescaling to the
        compact table's own norm would make compaction silently perturb the init.
        Compaction is purely representational; it must not.

        On a compacted model the full table exists ONLY as the transient below, so this is
        the single place it is observable. The caller must not take its norm until AFTER
        the COMPUTE_DTYPE cast — see the capture at the end of ``init_weights``.
        """
        if self.feature_vocab is None:
            init_fn(weight)
            return weight  # the table IS the full table
        full = torch.empty(
            self.config.n_sae_features,
            weight.shape[1],
            device=weight.device,
            dtype=weight.dtype,
        )
        init_fn(full)
        weight.copy_(self.feature_vocab.select_rows(full))
        return full

    @property
    def feature_embed_full_avg_row_norm(self) -> torch.Tensor:
        """The FULL ``(n_sae, d)`` feature table's average row norm.

        The rescale target for SAE-based encoder init. On a compacted model the full
        table is only ever a transient inside ``init_weights()``, so the number is
        captured there and read back here.

        Raises rather than falling back to the compact table's norm. That fallback is
        precisely the defect this exists to remove: it is *plausible* (the compact table
        is right there) and *wrong* (it is a sample estimate, off by ~2.1e-04 at the real
        shape), so it would silently make compaction perturb the init instead of being
        purely representational. Fail loud.
        """
        if self._feature_embed_full_avg_row_norm is None:
            raise FeatureVocabError(
                "the full feature_embed table's average row norm was never captured, so "
                "SAE-based encoder init cannot reproduce the uncompacted reference's "
                "rescale target. init_weights() has not run on this encoder. Build the "
                "model on `meta`, call to_empty(device=...), seed, then init_weights() "
                "BEFORE initialize_encoder_feature_embeddings_from_sae — the order "
                "production uses (conditional_encdec_train.py)."
            )
        return self._feature_embed_full_avg_row_norm

    def _to_compact(self, ids_flat, mask_flat):
        """Original SAE feature ids → compact embedding rows.

        The remap is computed here, per-forward, from the persistent sorted
        ``feature_vocab_ids`` buffer via ``searchsorted`` — there is NO stored lookup table.
        ``feature_vocab_ids`` is sorted and unique by construction (``FeatureVocab``
        canonicalizes, and ``feature_vocab_from_state_dict`` rejects any checkpoint whose
        buffer is not), so a binary search recovers the compact row, and the ``ids[pos] == id``
        check recovers membership.

        Mask-aware by necessity: every producer pads ``enc_feature_ids`` with literal ``0`` (a
        real SAE id a whitelist may exclude), and ``forward`` re-injects more ``0``s via its own
        FP8 ``F.pad``. Padding slots are PROVEN dead — masked as keys in cross-attention, so
        exactly-zero output effect and exactly-zero gradient — so they map to compact row 0. A
        **real** out-of-vocab id in a real slot is caught by the assert.

        The guard is ``torch._assert_async``: sync-free, zero graph breaks, and it fires in
        BOTH eager and compiled. It is load-bearing, not belt-and-braces — under
        ``torch.compile`` ``nn.Embedding(-1)`` SILENTLY RETURNS THE LAST ROW, so an OOV id that
        slipped through would corrupt the forward with no error. This form never emits ``-1``:
        the output is ``where(mask & hit, pos, 0)``, so a non-member always resolves to row 0
        and the *only* thing standing between an OOV id and a wrong answer is the assert — which
        is exactly why the assert stays and is checked before the return.

        ``in_range`` is NOT mask-gated, while the membership term is. So an id that is negative
        or ``>= n_sae`` raises even in a PADDING slot, whereas an in-range non-member in a
        padding slot is waved through. That asymmetry is the point: padding is contractually id
        ``0``, so a padding slot holding ``-5`` means the producer is broken and no masking
        makes that safe — whereas an in-range id in padding is exactly what compaction tolerates.
        """
        if self.feature_vocab is None:
            return ids_flat  # I5: zero added ops ⇒ byte-identical

        n = self.config.n_sae_features
        ids = self.feature_vocab_ids
        v = ids.numel()
        in_range = (ids_flat >= 0) & (ids_flat < n)
        # clamp is MEMORY-SAFETY ONLY (an out-of-range gather is UB under inductor); the
        # membership check and the assert guarantee the clamped index is never relied upon.
        pos = torch.searchsorted(ids, ids_flat.clamp(0, n - 1)).clamp(0, v - 1)
        hit = ids[pos] == ids_flat
        ok = in_range & ~(mask_flat & ~hit)
        torch._assert_async(ok.all(), self._vocab_oov_msg)
        return torch.where(mask_flat & hit, pos, torch.zeros_like(pos))

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize all weights explicitly.

        feature_embed:     Normal(0, 1)
        magnitude_embed:   Normal(0, 1)
        block attn c_qkv:  Uniform(+-sqrt(3/n_embd))
        block attn c_proj: Zeros  (intentional asymmetry vs cross-attention; see note)
        block mlp c_fc:    Uniform(+-sqrt(3/n_embd))
        block mlp c_proj:  Zeros  (intentional asymmetry vs cross-attention; see note)
        ve_gate:           Zeros  (so 2*sigmoid(0) = 1, neutral gate at init)

        NOTE: The encoder ``c_proj`` matrices and ``ve_gate`` are zero-init
        (residual-friendly identity at init), but cross-attention's ``c_proj``
        was switched to ``Uniform(±sqrt(3/...))`` because all-zero weights
        cause FP8 tensorwise scaling to degenerate (amax=0 → extreme scale
        factors → inf gradients on the first backward pass — see
        ``cross_attention.py::CrossAttentionLayer.init_weights`` docstring).

        The encoder side keeps zero-init intentionally: this is documented in
        audit ``2026-05-05`` map.md §6 B1 and consolidation manifest §7.
        Slice 8 audit pinned the contract via
        ``tests/audit_2026_05/slice_08_feature_encoder/test_init_weights_contract.py``.

        If FP8 cold-start instability surfaces in production, coordinate with
        slice 9 owner before changing — both schemas should move together.
        """
        # Embeddings.
        # feature_embed is feature-ID-indexed ⇒ SLICE-INIT (draws n_sae*n_embd randoms
        # even when compacted, so the RNG stream — and therefore every init below —
        # stays bitwise identical to the uncompacted reference). See
        # _init_feature_indexed_.
        #
        # Returns the FULL table — on a compacted model this is the only place it exists.
        # The SAE-init rescale target is derived from it AFTER the COMPUTE_DTYPE cast below;
        # see the capture at the end of this method for why the order is load-bearing.
        full_feature_table = self._init_feature_indexed_(
            self.feature_embed.weight,
            lambda w: nn.init.normal_(w, mean=0.0, std=1.0),
        )
        nn.init.normal_(self.magnitude_embed.weight, mean=0.0, std=1.0)

        # Transformer blocks
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5  # sqrt(3/n_embd) — same std as Normal
        for block in self.blocks:
            nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
            # Intentional zero-init (see init_weights docstring); diverges from
            # cross_attention.CrossAttentionLayer.init_weights by design.
            nn.init.zeros_(block.attn.c_proj.weight)
            nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            nn.init.zeros_(block.mlp.c_proj.weight)

        # Value embedding weights — also feature-ID-indexed ⇒ SLICE-INIT.
        for ve in self.value_embeds.values():
            self._init_feature_indexed_(ve.weight, lambda w: nn.init.uniform_(w, -s, s))

        # Vocab buffers (no-op when not compacted).
        self.init_vocab_buffers()

        # VE gate weights: zero-init so sigmoid(0)*2 = 1.0 (neutral)
        for block in self.blocks:
            if block.attn.ve_gate is not None:
                nn.init.zeros_(block.attn.ve_gate.weight)

        # Cast embeddings to COMPUTE_DTYPE on CUDA (optimizer can tolerate it, saves memory)
        if COMPUTE_DTYPE != torch.float32 and self.feature_embed.weight.device.type == "cuda":
            self.feature_embed.to(dtype=COMPUTE_DTYPE)
            self.magnitude_embed.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

        # The SAE-init rescale target, captured AFTER the cast and in the LIVE dtype.
        #
        # The order is load-bearing and the bug it prevents is invisible on CPU.
        # sae_embedding_init compares this number against what the UNCOMPACTED reference
        # computes from ITS table — and the reference reads its table live, i.e. bf16 on
        # CUDA. Capturing before the cast would stash the norm of a discarded fp32 table:
        # bf16 rounding moves it (measured: 5 of 384 elements of the resulting embedding
        # differ), so `random` SAE-init would silently stop being bitwise — on exactly the
        # hardware the queued run uses, and on no CPU test.
        #
        # `full_feature_table` is the compacted model's transient (a plain fp32 tensor, so
        # the module cast above did not touch it) or, when uncompacted, the weight
        # Parameter itself (which the cast DID convert in place). `.to(live dtype)` makes
        # both describe the same thing. Casting is elementwise, so it commutes with the row
        # subset: the compacted and reference runs land on the identical number.
        self._feature_embed_full_avg_row_norm = _average_row_norm(
            full_feature_table.to(dtype=self.feature_embed.weight.dtype)
        )

    def forward(
        self,
        feature_ids,
        magnitudes,
        mask,
        ste: SteInputs | None = None,
    ) -> PackedEncoderOutput:
        """Encode CUDA features and preserve their globally packed representation."""
        if not feature_ids.is_cuda:
            raise RuntimeError("FeatureEncoder.forward requires CUDA inputs")
        packed = self.forward_packed(feature_ids, magnitudes, mask, ste=ste)
        # Direct callers retain this state outside the encoder's dynamic graph.
        # Mark its packed extent here without changing the ordinary model path,
        # which calls ``forward_packed`` directly.
        mark_packed_encoder_output_dynamic(packed)
        return packed

    @torch.compiler.disable
    def forward_packed(
        self,
        feature_ids: torch.Tensor,
        magnitudes: torch.Tensor,
        mask: torch.Tensor,
        ste: SteInputs | None = None,
    ) -> PackedEncoderOutput:
        """Encode only real CUDA features on one globally packed token axis."""
        if not feature_ids.is_cuda:
            raise RuntimeError("forward_packed requires CUDA encoder inputs")
        packed = pack_encoder_inputs(feature_ids, magnitudes, mask)
        # Keep the variable packed length behind its own dynamic graph.  The
        # surrounding model can remain static, which is required by the FA4
        # decoder kernels in PyTorch 2.13, while this graph is reused across
        # different real-feature counts.
        return _compiled_encode_packed_dynamic(self, packed, ste)

    def _encode_packed(
        self,
        packed: PackedEncoderInputs,
        ste: SteInputs | None,
    ) -> PackedEncoderOutput:
        layout = packed.layout
        ids = self._to_compact(
            packed.feature_ids,
            layout.valid_mask.unsqueeze(0),
        )
        magnitudes = packed.magnitudes
        x = self.feature_embed(ids) + self.magnitude_embed(magnitudes)

        sign: torch.Tensor
        scores_flat: torch.Tensor
        packed_rows = layout.batch_indices * layout.n_segments + layout.segment_indices
        if ste is not None:
            # A compacted encoder is supported: its embedding table has
            # ``feature_vocab.size`` rows, so STE scores are emitted in compact
            # space. Shape accessors come from ``layout`` (the packed-state
            # refactor removed the standalone ``B``/``S`` locals).
            expected_scores_shape = (
                layout.batch_size,
                layout.n_segments,
                self.config.n_sae_features
                if self.feature_vocab is None
                else self.feature_vocab.size,
            )
            if tuple(ste.scores.shape) != expected_scores_shape:
                if self.feature_vocab is not None:
                    raise SelfCondTrainingError(
                        "ste.scores shape for a compacted encoder must be "
                        f"{expected_scores_shape}, got {tuple(ste.scores.shape)}"
                    )
                raise SelfCondTrainingError(
                    f"ste.scores shape must be {expected_scores_shape}, got "
                    f"{tuple(ste.scores.shape)}"
                )
            expected_sign_shape = (
                layout.batch_size,
                layout.n_segments,
                layout.feature_count,
            )
            if tuple(ste.sign.shape) != expected_sign_shape:
                raise SelfCondTrainingError(
                    f"ste.sign shape must be {expected_sign_shape}, got {tuple(ste.sign.shape)}"
                )

            sign = ste.sign.reshape(-1)[layout.flat_indices]
            sign = torch.where(layout.valid_mask, sign, torch.zeros_like(sign)).unsqueeze(0)
            scores_flat = ste.scores.reshape(
                layout.batch_size * layout.n_segments,
                -1,
            )
            if ste.form == SteForm.DENSE:
                mix = scores_flat.to(x.dtype) @ self.feature_embed.weight.detach()
                packed_mix = mix[packed_rows].unsqueeze(0)
                x = x + sign.to(x.dtype).unsqueeze(-1) * (packed_mix - packed_mix.detach())
            else:
                packed_scores = scores_flat[packed_rows].unsqueeze(0)
                col_scores = packed_scores.gather(2, ids.unsqueeze(-1)).squeeze(-1)
                rows = self.feature_embed.weight.detach()[ids]
                term = col_scores.to(x.dtype).unsqueeze(-1) * rows
                x = x + sign.to(x.dtype).unsqueeze(-1) * (term - term.detach())

        for i, block in enumerate(self.blocks):
            value_embed = (
                cast(nn.Embedding, self.value_embeds[str(i)])
                if str(i) in self.value_embeds
                else None
            )
            ve = value_embed(ids) if value_embed is not None else None
            if ste is not None and ve is not None:
                assert value_embed is not None
                value_weight = value_embed.weight.detach()
                if ste.form == SteForm.DENSE:
                    mix = scores_flat.to(ve.dtype) @ value_weight
                    packed_mix = mix[packed_rows].unsqueeze(0)
                    ve = ve + sign.to(ve.dtype).unsqueeze(-1) * (packed_mix - packed_mix.detach())
                else:
                    packed_scores = scores_flat[packed_rows].unsqueeze(0)
                    col_scores = packed_scores.gather(2, ids.unsqueeze(-1)).squeeze(-1)
                    rows = value_weight[ids]
                    term = col_scores.to(ve.dtype).unsqueeze(-1) * rows
                    ve = ve + sign.to(ve.dtype).unsqueeze(-1) * (term - term.detach())
            x = block(x, ve=ve, packed_block_mask=packed.block_mask)

        x = norm(x).nan_to_num()
        x = torch.where(
            layout.valid_mask[None, :, None],
            x,
            torch.zeros_like(x),
        )
        return PackedEncoderOutput(values=x, layout=layout)

# Compile the implementation method directly: it already owns the complete
# data-dependent encoder computation, so an extra one-call wrapper adds no
# compiler boundary. ``forward_packed`` remains disabled because packing and
# BlockMask construction must happen outside the surrounding static model graph.
_compiled_encode_packed_dynamic = torch.compile(
    FeatureEncoder._encode_packed,
    dynamic=True,
)
