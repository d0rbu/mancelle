"""
EncoderDecoderGPT — conditional pretraining model that composes a FeatureEncoder,
GPT decoder blocks, and CrossAttentionLayers.

SAE features condition the decoder through cross-attention: the encoder produces
per-segment feature representations which are injected into the decoder at regular
intervals via segment-level cross-attention (no gather expansion needed).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import NamedTuple, cast

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask

from errors import FeatureVocabError
from nanochat import flash_attention as _flash_attention
from nanochat.common import COMPUTE_DTYPE, get_dist_info, print0
from nanochat.gpt import Block, GPTConfig, Linear, OptimizerKind, has_ve, norm
from nanochat.optim import DistMuonAdamW, MuonAdamW
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    resolve_encoder_depth,
    resolve_encoder_dim,
    resolve_encoder_heads,
)
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
    build_packed_cross_attention_block_mask,
    validate_packed_cross_attention_inputs,
)
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
)
from steering.conditional.encoder_decoder.feature_prediction import (
    compute_feature_pred_loss,
    resolve_layer_index,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderLayout,
    PackedEncoderOutput,
)
from steering.conditional.encoder_decoder.packed_optimizer import (
    DistPackedMuonAdamW,
    PackedMuonAdamW,
)
from steering.conditional.encoder_decoder.sm12_fa4 import (
    admit_project_attention,
    uses_native_sm12_fa4,
)
from steering.conditional.encoder_decoder.typed_module_list import TypedModuleList
from steering.conditional.feature_mask import FeatureMask
from steering.conditional.feature_vocab import FeatureVocab
from steering.refined_types import (
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

LOGIT_SOFTCAP = 15
ROTARY_SEQ_MULTIPLIER = 10  # 10x to support extended context during generation
BASE_MODEL_DIM = 768  # Reference dimension for muP-style LR scaling


class ForwardResult(NamedTuple):
    loss: t.Tensor
    lm_loss: t.Tensor
    feature_pred_loss: t.Tensor


def _coalesce_compatible_muon_groups(param_groups: list[dict]) -> list[dict]:
    """Merge shape-identical Muon groups only when all update semantics match."""
    result: list[dict] = []
    positions: dict[tuple[object, ...], int] = {}
    for group in param_groups:
        if group["kind"] != OptimizerKind.MUON:
            result.append(group)
            continue
        params = list(group["params"])
        if not params:
            raise ValueError("Muon parameter groups must not be empty")
        shapes = {parameter.shape for parameter in params}
        dtypes = {parameter.dtype for parameter in params}
        devices = {parameter.device for parameter in params}
        if len(shapes) != 1 or len(dtypes) != 1 or len(devices) != 1:
            raise ValueError("Muon groups must have uniform shape, dtype, and device")
        key = (
            next(iter(shapes)),
            next(iter(dtypes)),
            next(iter(devices)),
            group["lr"],
            group["momentum"],
            group["ns_steps"],
            group["beta2"],
            group["weight_decay"],
        )
        position = positions.get(key)
        if position is None:
            positions[key] = len(result)
            result.append({**group, "params": params})
        else:
            result[position]["params"].extend(params)
    return result


def fp_head_unavailable_error(
    model: "EncoderDecoderGPT", requested: str, uncompacted: str
) -> RuntimeError:
    """Return the common error for a model built without an FP head.

    Compact and full-width models can both own a feature-prediction head.  Its
    absence therefore has one meaning: this particular model/checkpoint was
    constructed without ``feature_pred_config``.
    """
    del model, requested
    return RuntimeError(uncompacted)


class EncoderDecoderGPT(nn.Module):
    """Conditional pretraining model: encoder produces feature representations,
    decoder generates tokens conditioned on those features via cross-attention."""

    def __init__(
        self,
        dec_config: GPTConfig,
        enc_config: EncoderConfig,
        xattn_config: CrossAttentionConfig,
        n_sae_features: int,
        num_buckets: int,
        pad_vocab_size_to: int = 64,
        feature_pred_config: FeaturePredictionConfig | None = None,
        feature_vocab: FeatureVocab | None = None,
    ):
        super().__init__()
        admit_project_attention(device=t.get_default_device())
        self.dec_config = dec_config
        self.enc_config = enc_config
        self.xattn_config = xattn_config
        # Eager evaluation-only seam. ``None`` is deliberately an ordinary
        # Python attribute (not a Module/parameter/buffer), so the default path
        # adds no tensor work and no state-dict key. Scoped installers live in
        # ``eval_residual.py`` and restore this slot in ``finally``.
        self._post_cross_attention_eval_callback = None
        self._single_token_mask_cache_enabled = False
        self._single_token_mask_cache: tuple[PackedEncoderLayout, int, BlockMask] | None = None

        # Encoder feature-dictionary compaction (issue #695). ``None`` ⇒ full-width
        # tables and a byte-identical encoder (invariant I5).
        if feature_vocab is not None and feature_vocab.n_sae_features != n_sae_features:
            raise FeatureVocabError(
                f"feature_vocab.n_sae_features={feature_vocab.n_sae_features} disagrees "
                f"with the model's n_sae_features={n_sae_features}."
            )
        # An identity vocab is refused by ``FeatureEncoder.__init__`` — the class that sizes the
        # tables and registers the vocab buffers, and therefore owns the invariant. Duplicating
        # the check here would only protect callers who come in through the model, which is
        # exactly the hole that put it in the wrong place to begin with.
        # Resolve encoder depth, dim, and head counts from config
        # (may be ratios of decoder — follows same scaling laws)
        depth_spec = (
            NonNegativeInt.parse(enc_config.depth)
            if isinstance(enc_config.depth, int)
            else NonNegativeFloat.parse(enc_config.depth)
        )
        dim_spec = (
            PositiveInt.parse(enc_config.n_embd)
            if isinstance(enc_config.n_embd, int)
            else PositiveFloat.parse(enc_config.n_embd)
        )
        enc_depth = resolve_encoder_depth(
            depth_spec,
            PositiveInt.parse(dec_config.n_layer),
        )
        enc_dim = resolve_encoder_dim(
            dim_spec,
            PositiveInt.parse(dec_config.n_embd),
        )
        enc_n_head, enc_n_kv_head = resolve_encoder_heads(
            (
                PositiveInt.parse(enc_config.n_head)
                if enc_config.n_head is not None
                else None
            ),
            (
                PositiveInt.parse(enc_config.n_kv_head)
                if enc_config.n_kv_head is not None
                else None
            ),
            enc_dim,
            PositiveInt.parse(enc_config.head_dim),
        )

        # Build FeatureEncoder (output stays at enc_dim; cross-attention handles dim change)
        self.enc_dim = enc_dim
        enc_fe_config = FeatureEncoderConfig(
            n_sae_features=n_sae_features,
            num_buckets=num_buckets,
            n_embd=enc_dim,
            n_layer=enc_depth,
            n_head=enc_n_head,
            n_kv_head=enc_n_kv_head,
            head_dim=enc_config.head_dim,
            ve_gate_channels=enc_config.ve_gate_channels,
        )
        self.encoder = FeatureEncoder(enc_fe_config, feature_vocab=feature_vocab)

        # Build decoder components (same pattern as GPT)
        self.window_sizes = self._compute_window_sizes(dec_config)

        padded_vocab_size = (
            (dec_config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != dec_config.vocab_size:
            print0(
                f"Padding vocab_size from {dec_config.vocab_size} to {padded_vocab_size} for efficiency"
            )

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, dec_config.n_embd),
                "h": TypedModuleList(
                    [
                        Block(dec_config, layer_idx)
                        for layer_idx in range(dec_config.n_layer)
                    ],
                    module_type=Block,
                    item_label="EncoderDecoderGPT decoder block",
                ),
            }
        )
        self.lm_head = Linear(dec_config.n_embd, padded_vocab_size, bias=False)

        # Per-layer learnable scalars
        self.resid_lambdas = nn.Parameter(t.ones(dec_config.n_layer))
        self.x0_lambdas = nn.Parameter(t.zeros(dec_config.n_layer))

        # Value embeddings (ResFormer-style)
        head_dim = dec_config.n_embd // dec_config.n_head
        kv_dim = dec_config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(dec_config.n_layer)
                if has_ve(i, dec_config.n_layer)
            }
        )

        # Rotary embeddings
        self.rotary_seq_len = dec_config.sequence_len * ROTARY_SEQ_MULTIPLIER
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        # Cross-attention layers at every frequency-th decoder layer
        freq = xattn_config.frequency
        # Resolve cross-attention head counts (None → match decoder head structure)
        xattn_n_head = xattn_config.n_head if xattn_config.n_head is not None else dec_config.n_head
        xattn_n_kv_head = (
            xattn_config.n_kv_head if xattn_config.n_kv_head is not None else xattn_n_head
        )
        xattn_head_dim = dec_config.n_embd // xattn_n_head
        xattn_layer_config = CrossAttentionLayerConfig(
            dec_n_embd=dec_config.n_embd,
            enc_n_embd=enc_dim,
            n_head=xattn_n_head,
            n_kv_head=xattn_n_kv_head,
            head_dim=xattn_head_dim,
        )
        self.cross_attn_layers = nn.ModuleDict(
            {
                str(i): CrossAttentionLayer(xattn_layer_config)
                for i in range(dec_config.n_layer)
                if (i + 1) % freq == 0
            }
        )

        # Feature prediction head (optional auxiliary loss)
        self.feature_pred_config = feature_pred_config
        if feature_pred_config is not None:
            self._feature_pred_layer_idx = resolve_layer_index(
                feature_pred_config.layer,
                dec_config.n_layer,
            )
            fp_width = n_sae_features if feature_vocab is None else feature_vocab.size
            self.feature_pred_head = Linear(dec_config.n_embd, fp_width, bias=True)
            if feature_pred_config.head_layers:
                self.feature_pred_head = ResidualFeaturePredictionHead(
                    self.feature_pred_head,
                    feature_pred_config.head_layers,
                    feature_pred_config.block,
                    decoder_config=dec_config,
                )
            self._n_sae_features = n_sae_features
        else:
            self._feature_pred_layer_idx = -1  # unused sentinel
            self.feature_pred_head = None
            self._n_sae_features = n_sae_features

        # Optional feature blacklist/whitelist for FP-loss masking.
        # Set by the training script after both model and feature_pipeline exist.
        # Defaults to None (byte-identical no-op, invariant I5).
        #
        # t.compile contract (H10):
        #   Dynamo guards on this attribute's identity at trace time.  It MUST be
        #   set on orig_model (the unwrapped module) before the first compiled
        #   forward — training scripts do so before their "while True:" loops.
        #   Changing it after the first compiled forward triggers a dynamo
        #   recompile (correctness-safe but slow).  The H10 fail-fast assertions
        #   in the training scripts enforce the set-before-first-forward ordering.
        self.feature_mask: FeatureMask | None = None

    @property
    def _decoder_embedding(self) -> nn.Embedding:
        """Return the validated token embedding stored in ``transformer``."""
        embedding = self.transformer["wte"]
        if not isinstance(embedding, nn.Embedding):
            raise TypeError(
                "EncoderDecoderGPT transformer['wte'] must be nn.Embedding, "
                f"got {type(embedding).__name__}"
            )
        return embedding

    @property
    def _decoder_blocks(self) -> TypedModuleList[Block]:
        """Return the validated decoder block collection."""
        blocks = self.transformer["h"]
        if (
            not isinstance(blocks, TypedModuleList)
            or blocks.module_type is not Block
        ):
            raise TypeError(
                "EncoderDecoderGPT transformer['h'] must be "
                "TypedModuleList[Block], "
                f"got {type(blocks).__name__}"
            )
        # String-form cast: ``cast(TypedModuleList[Block], ...)`` evaluated
        # the generic subscript at runtime, which Dynamo cannot trace
        # ("SourcelessBuilder ... typing._GenericAlias") — the resulting hard
        # graph break split each Block.forward into its own frame whose
        # window_size ints became SymInts, and the flex FLASH backend rejects
        # the symbolic mask capture.  The string form never constructs the
        # alias object, so the property traces cleanly while ty still
        # narrows the return type.
        return cast("TypedModuleList[Block]", blocks)

    def _value_embedding(self, layer_key: str) -> nn.Embedding | None:
        """Return one validated sparse value embedding, if present."""
        if layer_key not in self.value_embeds:
            return None
        embedding = self.value_embeds[layer_key]
        if not isinstance(embedding, nn.Embedding):
            raise TypeError(
                f"EncoderDecoderGPT value_embeds[{layer_key!r}] must be "
                f"nn.Embedding, got {type(embedding).__name__}"
            )
        return embedding

    def _value_embedding_values(self) -> tuple[nn.Embedding, ...]:
        """Return every sparse value embedding with concrete types preserved."""
        embeddings: list[nn.Embedding] = []
        for key in self.value_embeds:
            embedding = self._value_embedding(key)
            assert embedding is not None
            embeddings.append(embedding)
        return tuple(embeddings)

    def _cross_attention_layer(
        self, layer_key: str
    ) -> CrossAttentionLayer | None:
        """Return one validated cross-attention layer, if present."""
        if layer_key not in self.cross_attn_layers:
            return None
        layer = self.cross_attn_layers[layer_key]
        if not isinstance(layer, CrossAttentionLayer):
            raise TypeError(
                f"EncoderDecoderGPT cross_attn_layers[{layer_key!r}] must be "
                f"CrossAttentionLayer, got {type(layer).__name__}"
            )
        return layer

    def _cross_attention_layer_values(
        self,
    ) -> tuple[CrossAttentionLayer, ...]:
        """Return every cross-attention layer with concrete types preserved."""
        layers: list[CrossAttentionLayer] = []
        for key in self.cross_attn_layers:
            layer = self._cross_attention_layer(key)
            assert layer is not None
            layers.append(layer)
        return tuple(layers)

    def _compute_window_sizes(self, config):
        """Compute per-layer window sizes for sliding window attention (same as GPT)."""
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        long_window = config.sequence_len
        short_window = long_window // 2
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def _precompute_rotary_embeddings(
        self,
        seq_len: int,
        head_dim: int,
        base: float = 10000,
        device: t.device | str | None = None,
    ) -> tuple[t.Tensor, t.Tensor]:
        """Precompute rotary position embeddings (same as GPT)."""
        if device is None:
            device = self._decoder_embedding.weight.device
        channel_range = t.arange(
            0, head_dim, 2, dtype=t.float32, device=device
        )
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        positions = t.arange(seq_len, dtype=t.float32, device=device)
        freqs = t.outer(positions, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :]
        return cos, sin

    @t.no_grad()
    def init_weights(self):
        """Initialize all weights: decoder (same as GPT), encoder, cross-attention."""
        # --- Decoder init (mirrors GPT.init_weights) ---
        t.nn.init.normal_(
            self._decoder_embedding.weight, mean=0.0, std=1.0
        )
        t.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        n_embd = self.dec_config.n_embd
        s = 3**0.5 * n_embd**-0.5
        for block in self._decoder_blocks:
            t.nn.init.uniform_(block.attn.c_qkv.weight, -s, s)
            t.nn.init.zeros_(block.attn.c_proj.weight)
            t.nn.init.uniform_(block.mlp.c_fc.weight, -s, s)
            t.nn.init.zeros_(block.mlp.c_proj.weight)

        self.resid_lambdas.fill_(1.0)
        self.x0_lambdas.fill_(0.1)

        for ve in self._value_embedding_values():
            t.nn.init.uniform_(ve.weight, -s, s)

        for block in self._decoder_blocks:
            if block.attn.ve_gate is not None:
                t.nn.init.zeros_(block.attn.ve_gate.weight)

        # Rotary embeddings
        head_dim = self.dec_config.n_embd // self.dec_config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE on CUDA
        if (
            COMPUTE_DTYPE != t.float32
            and self._decoder_embedding.weight.device.type == "cuda"
        ):
            self._decoder_embedding.to(dtype=COMPUTE_DTYPE)
            for ve in self._value_embedding_values():
                ve.to(dtype=COMPUTE_DTYPE)

        # --- Encoder init ---
        self.encoder.init_weights()

        # --- Cross-attention init ---
        for layer in self._cross_attention_layer_values():
            layer.init_weights()

        # --- Feature prediction head init ---
        if self.feature_pred_head is not None:
            if self.feature_vocab is None:
                # Literal legacy branch: preserve shape, initializer call, RNG
                # stream, and storage writes byte-for-byte.
                t.nn.init.normal_(self.feature_pred_head.weight, mean=0.0, std=0.001)
            else:
                # Draw the full reference table once, then retain only effective
                # representative rows.  The FP head is initialized last, so this
                # preserves both selected rows and the final global RNG state.
                full_weight = t.empty(
                    self._n_sae_features,
                    self.feature_pred_head.in_features,
                    dtype=self.feature_pred_head.weight.dtype,
                    device=self.feature_pred_head.weight.device,
                )
                t.nn.init.normal_(full_weight, mean=0.0, std=0.001)
                self.feature_pred_head.weight.copy_(
                    self.feature_vocab.select_rows(full_weight)
                )
                del full_weight
            t.nn.init.zeros_(self.feature_pred_head.bias)
            if isinstance(self.feature_pred_head, ResidualFeaturePredictionHead):
                self.feature_pred_head.init_residual_weights()

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Check the materialized device after CPU/meta construction, including
        # ordinary .to(cuda) and the checkpoint loader's .to_empty(device=...).
        admit_project_attention(device=self.get_device())
        return result

    # NOTE: there is deliberately NO ``to_empty`` override here, and none on ``FeatureEncoder``.
    # An earlier version had both, to poison the encoder's derived vocab map after ``to_empty()``
    # hands every buffer fresh uninitialized storage. They were redundant with
    # ``FeatureEncoder._apply`` — which poisons on ANY storage-replacing move — and they were
    # WRONG: this one called ``poison_vocab_map()`` unconditionally while forwarding ``recurse``,
    # so ``to_empty(device=…, recurse=False)`` — which by contract materialises only THIS
    # module's tensors and leaves children alone — destroyed a live, correct map that
    # ``super()`` had never touched, and the next forward raised on an IN-VOCAB feature.
    #
    # ``_apply`` cannot make that mistake: it IS the recursion, so it fires exactly when the
    # encoder's own storage is replaced and never when it isn't.
    #
    # Inserting a method here also silently STOLE ``@t.no_grad()`` from ``init_buffers``
    # below — the decorator binds to whatever function follows it, and a line diff shows
    # nothing. Pinned by ``test_every_init_method_still_carries_no_grad``.

    @t.no_grad()
    def init_buffers(self):
        """Materialize buffers after ``to_empty()`` without touching parameters.

        The decoder's rotary ``cos``/``sin`` (non-persistent, recomputed), plus the encoder's
        two PERSISTENT vocab buffers (``feature_vocab_ids`` / ``feature_vocab_n_sae``). Both
        must be written after ``to_empty()``, which leaves ``meta``-created buffers holding
        uninitialised storage.

        On the LOAD path this runs BEFORE ``load_state_dict``, which then overwrites the vocab
        buffers with the checkpoint's — the ONLY correct order (see
        ``FeatureEncoder.init_vocab_buffers``). It must never be called to "repair" a model
        after a load: it would overwrite the checkpoint's vocabulary with the configured one.
        """
        head_dim = self.dec_config.n_embd // self.dec_config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin
        self.encoder.init_vocab_buffers()

    # -- feature vocabulary (issue #695) -----------------------------------

    @property
    def feature_vocab(self) -> FeatureVocab | None:
        """The encoder's compacted feature dictionary, or ``None`` when full-width.

        Lives on ``FeatureEncoder`` (next to the tables it sizes); exposed here so the
        FP-head follow-up can reach it without moving anything.
        """
        return self.encoder.feature_vocab

    @property
    def feature_vocab_size(self) -> int:
        """V — the encoder's FEATURE-embedding table row count.

        Equals ``n_sae_features`` when not compacted. It is neither of the two numbers
        it could be mistaken for:

        * NOT ``n_sae_features``, which stays the ORIGINAL public width.
        * NOT ``vocab_size``. That name means the TOKEN vocabulary everywhere in this
          repo (``GPTConfig.vocab_size == 32768``; ``forward`` slices its logits with
          ``logits[..., : self.dec_config.vocab_size]``), so a model-level
          ``vocab_size`` returning the feature-table height is a trap: any generic
          consumer reaching for the near-universal ``model.vocab_size`` idiom — a
          sampler, a logit slicer, an HF-style adapter — would silently get V and slice
          against the wrong width, with no error to catch. Pinned by
          ``test_encoder_feature_vocab.py::test_the_FEATURE_table_height_is_NOT_called_vocab_size``.
        """
        return self.encoder.feature_embed.weight.shape[0]

    @property
    def feature_prediction_width(self) -> int | None:
        """FP-head output width, or ``None`` when the model has no FP head."""
        if self.feature_pred_head is None:
            return None
        return self.feature_pred_head.out_features

    def assert_vocab_consistent(self) -> None:
        """Verify the persistent vocab buffer still agrees with the Python ``FeatureVocab``.

        **Required after every ``load_state_dict``.** ``assign=True`` REPLACES
        ``feature_vocab_ids`` with the checkpoint's tensor. If that tensor describes a
        different vocabulary than the model was configured with, every embedding row is
        silently reinterpreted — this is the guard that turns that into a raise.

        There is no derived map to check: ``_to_compact`` computes the compaction on the fly
        from ``feature_vocab_ids``, so ``ids`` being correct is the whole invariant. Runs on
        every rank of every train and eval process, against the buffers the model already holds.
        """
        enc = self.encoder
        vocab = enc.feature_vocab
        if vocab is None:
            if hasattr(enc, "feature_vocab_ids"):
                raise FeatureVocabError(
                    "encoder has vocab buffers but no FeatureVocab — inconsistent state"
                )
            return

        # (1) The persistent ids buffer vs the configured Python vocab. This is what makes the
        # checkpoint's own record answerable to the configured vocabulary. dtype-strict, because
        # t.equal compares an int32 and an int64 buffer of the same values as EQUAL.
        ids = enc.feature_vocab_ids
        expected_ids = vocab.new_to_old_tensor(device=ids.device)
        if ids.dtype != expected_ids.dtype or not t.equal(ids, expected_ids):
            raise FeatureVocabError(
                "encoder.feature_vocab_ids disagrees with the model's FeatureVocab. "
                "A checkpoint was loaded whose vocabulary differs from the configured "
                "one; every embedding row would be silently reinterpreted."
            )

        # (2) The original dictionary width.
        if int(enc.feature_vocab_n_sae.item()) != vocab.n_sae_features:
            raise FeatureVocabError(
                f"encoder.feature_vocab_n_sae="
                f"{int(enc.feature_vocab_n_sae.item())} disagrees with the model's "
                f"n_sae_features={vocab.n_sae_features}."
            )

    def get_device(self) -> t.device:
        return self._decoder_embedding.weight.device

    @property
    def num_cache_layers(self) -> int:
        """Decoder slots followed by transformer feature-head slots."""
        fp = self.feature_pred_config
        depth = fp.head_layers if fp is not None and fp.block == FeaturePredictionBlock.TRANSFORMER else 0
        return self.dec_config.n_layer + depth

    def apply_feature_prediction_head(
        self,
        hidden: t.Tensor,
        start_pos: int = 0,
        *,
        kv_cache=None,
        project: bool = True,
    ) -> t.Tensor:
        """Apply the head to normalized captured states using their decoder context.

        Keeping this outside the trunk also lets head-only RLVR backpropagate
        through every head block while collecting trunk states under no_grad.
        """
        head = self.feature_pred_head
        if head is None:
            raise ValueError("model has no feature-prediction head")
        if isinstance(head, ResidualFeaturePredictionHead):
            end_pos = start_pos + hidden.shape[1]
            if start_pos < 0 or end_pos > self.rotary_seq_len:
                raise ValueError("feature head positions exceed decoder RoPE capacity")
            return head(
                hidden,
                cos_sin=(self.cos[:, start_pos:end_pos], self.sin[:, start_pos:end_pos]),
                window_size=self.window_sizes[self._feature_pred_layer_idx],
                kv_cache=kv_cache,
                project=project,
            )
        return head(hidden) if project else hidden

    def _validate_kv_cache(self, cache, batch: int, length: int, *, needs_features: bool) -> None:
        """Admit a complete cache or an LM-only decoder cache before any writes."""
        if cache is None:
            return
        decoder_layers = self.dec_config.n_layer
        required = self.num_cache_layers
        if cache.n_layers not in (decoder_layers, required) or (
            needs_features and cache.n_layers != required
        ):
            raise ValueError(
                f"feature prediction requires {required} cache layers; decoder-only LM "
                f"inference permits {decoder_layers}, got {cache.n_layers}. "
                "Allocate the complete cache before prefill."
            )
        if (
            cache.batch_size != batch
            or cache.n_heads != self.dec_config.n_kv_head
            or cache.head_dim != self.dec_config.n_embd // self.dec_config.n_head
        ):
            raise ValueError("KV cache shape does not match decoder batch/head geometry")
        positions = cache.cache_seqlens
        if length < 1 or bool((positions < 0).any()) or bool((positions != positions[0]).any()):
            raise ValueError("KV cache requires nonnegative aligned positions and a nonempty suffix")
        if bool((positions + length > min(cache.max_seq_len, self.rotary_seq_len)).any()):
            raise ValueError("KV cache capacity or RoPE capacity exceeded")

    def prewarm_decoder_block_masks(self, batch_size: int, seq_len: int) -> None:
        """Eagerly build every decoder self-attention BlockMask that a
        compiled forward will need for ``(batch_size, seq_len)`` inputs.

        nanochat's flash_attention caches BlockMasks by ``(device, B, H_q,
        H_kv, S, window_left, block sizes)``.  A cache MISS inside a compiled
        forward rebuilds the mask in-graph — and there the per-layer window
        int is a SymInt: the deliberate packed cross-attention graph breaks
        split each decoder ``Block.forward`` into its own Dynamo frame, whose
        shared code object sees both the S- and L-layer window values, so
        automatic dynamic promotes the int on the second value.  torch>=2.13
        lowers decoder flex attention through the FLASH (CuteDSL) backend on
        Hopper+, which refuses symbolic mask_mod captures ("NYI: score_mod or
        mask_mod captures a dynamic scalar").  A cache HIT instead returns a
        mask whose mask_mod closes over a concrete Python int, which the
        FLASH template inlines.  nanochat/ is owner-gated, so the warm-up
        lives here; call it before ``t.compile``-ing the model.

        No-op when the flex path is inactive (sdpa / fa3 override or no flex
        support) — those paths never consult the BlockMask cache.
        """
        device = self.get_device()
        if uses_native_sm12_fa4(device) or not _flash_attention._use_flex():
            return
        for window_left in sorted({window[0] for window in self.window_sizes}):
            _flash_attention._get_block_mask(
                batch_size,
                self.dec_config.n_head,
                self.dec_config.n_kv_head,
                seq_len,
                window_left,
                device,
            )

    @contextmanager
    def reuse_single_token_packed_masks(self) -> Iterator[None]:
        """Reuse routing only while one eager, batch-one generation owns the model."""
        if self.training:
            raise RuntimeError("packed-mask reuse requires evaluation mode")
        if self._single_token_mask_cache_enabled:
            raise RuntimeError("packed-mask reuse cannot be nested")
        self._single_token_mask_cache_enabled = True
        self._single_token_mask_cache = None
        try:
            yield
        finally:
            self._single_token_mask_cache = None
            self._single_token_mask_cache_enabled = False

    def forward(
        self,
        idx,
        targets,
        enc_feature_ids=None,
        enc_magnitudes=None,
        enc_mask: t.Tensor | None = None,
        seg_map=None,
        loss_reduction="mean",
        kv_cache=None,
        encoder_out: PackedEncoderOutput | None = None,
        encoder_strict=False,
        seg_doc_id=None,
        static_max_seg_size: int | None = None,
        fp_targets=None,
        return_fp_logits: bool = False,
        fp_logits_only: bool = False,
        logits_to_keep: int = 0,
        fp_hidden_only: bool = False,
    ):
        """
        Args:
            idx:             (B, T) int tensor of token indices
            targets:         (B, T) int tensor of target token indices, or None for inference
            enc_feature_ids: (B, n_seg, F) int tensor of SAE feature indices (None if encoder_out provided)
            enc_magnitudes:  (B, n_seg, F) int tensor of discretized magnitude bucket indices (None if encoder_out provided)
            enc_mask:        (B, n_seg, F) bool tensor for raw encoder inputs;
                             must be omitted with precomputed packed state
            seg_map:         (B, T) int tensor mapping each decoder position to a segment index
            loss_reduction:  'mean' or 'none' for cross-entropy loss
            kv_cache:        optional KVCache for autoregressive generation
            encoder_out:     optional precomputed PackedEncoderOutput
            encoder_strict:  if True, assert that either (enc_feature_ids, enc_magnitudes) XOR encoder_out is provided
            return_fp_logits: inference-only flag. When True (and ``targets is None``),
                returns ``(logits, fp_logits)`` where ``fp_logits`` has shape
                ``(B, T, feature_prediction_width)``. This is the effective
                feature-vocabulary width for a compact checkpoint and the original
                SAE width otherwise. Used by self-conditioned generation
                to read the feature-prediction head without a separate forward
                pass. Requires the model to have been built with
                ``feature_pred_config``. Incompatible with ``targets`` (training
                path keeps its existing return type via ``ForwardResult``).
            logits_to_keep: number of final token positions to project through
                the LM head; zero preserves all positions. Positive values
                require targets=None. Decoder states and feature logits retain
                every position, including when ``return_fp_logits=True``.
            fp_hidden_only: returns only the normalized feature-head input
                (B, T, H), without applying either output head. Requires
                ``targets=None`` and excludes the other feature-output flags.

        Returns:
            loss (scalar) if targets is not None, else logits (B, T, vocab_size).
            A positive ``logits_to_keep`` limits the LM-logit time dimension to
            ``min(T, logits_to_keep)``.
            If ``return_fp_logits=True`` and ``targets is None``, returns
            ``(logits, fp_logits)`` instead.
        """
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("logits_to_keep must be a nonnegative integer")
        if logits_to_keep and targets is not None:
            raise ValueError("logits_to_keep requires targets=None")
        if fp_logits_only or fp_hidden_only:
            requested = "fp_hidden_only=True" if fp_hidden_only else "fp_logits_only=True"
            if return_fp_logits or (fp_hidden_only and fp_logits_only):
                raise ValueError(
                    f"{requested} and other feature-output flags are mutually exclusive"
                )
            if targets is not None:
                raise ValueError(
                    f"{requested} is incompatible with targets: the LM "
                    "logits/loss are deliberately skipped in this mode."
                )
            if self.feature_pred_head is None:
                raise fp_head_unavailable_error(
                    self,
                    requested=requested,
                    uncompacted=(
                        f"{requested} requires a feature-prediction head, "
                        "but this model was constructed without feature_pred_config."
                    ),
                )
        if return_fp_logits and targets is not None:
            raise ValueError(
                "return_fp_logits=True is incompatible with targets being "
                "provided. Use the standard training-path return type "
                "(ForwardResult) for joint LM + fp loss computation; the "
                "fp_logits flag is reserved for inference."
            )
        if return_fp_logits and self.feature_pred_head is None:
            raise fp_head_unavailable_error(
                self,
                requested="return_fp_logits=True",
                uncompacted=(
                    "return_fp_logits=True requires a feature-prediction head, but "
                    "this model was constructed without feature_pred_config. "
                    "Either rebuild with feature_pred_config or omit the flag."
                ),
            )
        B, T = idx.size()
        needs_fp = fp_logits_only or return_fp_logits or (
            targets is not None and fp_targets is not None and self.feature_pred_head is not None
        )
        self._validate_kv_cache(kv_cache, B, T, needs_features=needs_fp)

        has_raw = enc_feature_ids is not None and enc_magnitudes is not None
        has_precomputed = encoder_out is not None
        if encoder_strict:
            assert has_raw ^ has_precomputed, (
                "Exactly one of (enc_feature_ids, enc_magnitudes) or encoder_out must be provided"
            )

        # --- Encoder forward (skip if precomputed) ---
        if has_precomputed:
            if not isinstance(encoder_out, PackedEncoderOutput):
                raise TypeError(
                    "precomputed encoder_out must be PackedEncoderOutput; "
                    "dense encoder state is no longer a production contract"
                )
            if enc_mask is not None:
                raise ValueError(
                    "enc_mask must be omitted when encoder_out is precomputed; "
                    "packed ownership metadata is authoritative"
                )
            encoder_state = encoder_out
        else:
            assert enc_feature_ids is not None and enc_magnitudes is not None
            assert enc_mask is not None, "enc_mask is required with raw encoder inputs"
            encoder_state = self.encoder.forward_packed(
                enc_feature_ids,
                enc_magnitudes,
                enc_mask,
            )

        if encoder_state.layout.batch_size != B:
            raise ValueError(
                "packed encoder batch size must match decoder batch size: "
                f"{encoder_state.layout.batch_size} != {B}"
            )
        if encoder_state.values.device != idx.device:
            raise ValueError(
                "packed encoder state and decoder tokens must share a device: "
                f"{encoder_state.values.device} != {idx.device}"
            )

        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        try:
            # --- Decoder forward ---
            if not t.compiler.is_compiling():
                # Cached suffixes start at a non-zero offset. Checking only ``T``
                # permits a truncated/empty RoPE slice when T fits by itself but
                # ``T0 + T`` does not.
                assert T0 + T <= self.cos.size(1), (
                    f"RoPE cache too small for T0={T0}, T={T}, "
                    f"cache={self.cos.size(1)}"
                )
                assert self.cos.dtype == COMPUTE_DTYPE, (
                    f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
                )
            cos_sin = self.cos[:, T0 : T0 + T], self.sin[:, T0 : T0 + T]

            x = self._decoder_embedding(idx)
            x = x.to(COMPUTE_DTYPE)
            x = norm(x)
            x0 = x

            packed_xattn_mask = None
            if self.cross_attn_layers:
                assert seg_map is not None, "seg_map is required for cross-attention"
                if x.is_cuda:
                    validate_packed_cross_attention_inputs(
                        x,
                        encoder_state,
                        seg_map,
                        static_max_seg_size=static_max_seg_size,
                    )
                if self._single_token_mask_cache_enabled and (B, T) == (1, 1):
                    if t.compiler.is_compiling():
                        raise RuntimeError("packed-mask reuse requires eager evaluation")
                    # A self-conditioning refresh returns a new packed layout.
                    # Retain the layout itself so identity cannot be recycled.
                    slot = int(seg_map.item())
                    cached = self._single_token_mask_cache
                    if cached is not None and cached[0] is encoder_state.layout and cached[1] == slot:
                        packed_xattn_mask = cached[2]
                    else:
                        packed_xattn_mask = build_packed_cross_attention_block_mask(
                            seg_map,
                            encoder_state.layout,
                        )
                        self._single_token_mask_cache = (
                            encoder_state.layout,
                            slot,
                            packed_xattn_mask,
                        )
                else:
                    packed_xattn_mask = build_packed_cross_attention_block_mask(
                        seg_map,
                        encoder_state.layout,
                    )

            feature_pred_hidden = None
            post_cross_attention_callback = self._post_cross_attention_eval_callback
            if post_cross_attention_callback is not None:
                if self.training:
                    raise RuntimeError(
                        "post-cross-attention callbacks are evaluation-only; call model.eval()"
                    )
                if t.compiler.is_compiling():
                    raise RuntimeError(
                        "active post-cross-attention callbacks require eager evaluation"
                    )
            for i, block in enumerate(self._decoder_blocks):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                x = x.to(COMPUTE_DTYPE)  # resid_lambdas are fp32 → may promote x; cast back
                value_embedding = self._value_embedding(str(i))
                ve = (
                    value_embedding(idx).to(x.dtype)
                    if value_embedding is not None
                    else None
                )
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache=kv_cache)

                # Cross-attention injection (seg_map-based: supports variable-sized segments)
                cross_attention = self._cross_attention_layer(str(i))
                if cross_attention is not None:
                    normalized_x = norm(x)
                    if normalized_x.is_cuda:
                        xattn_out = cross_attention._forward_prevalidated(
                            normalized_x,
                            encoder_state,
                            packed_xattn_mask,
                        )
                    else:
                        # Production cross-attention is CUDA-only. Keeping the
                        # checked public call here preserves the test-owned CPU
                        # reference seam without widening the production contract.
                        xattn_out = cross_attention(
                            normalized_x,
                            encoder_state,
                            seg_map,
                            static_max_seg_size=static_max_seg_size,
                            packed_block_mask=packed_xattn_mask,
                        )
                    x = x + xattn_out
                # This is intentionally after the same-layer cross-attention
                # residual injection and before feature-prediction state capture.
                # It therefore affects ordinary logits, fp logits, prefill, and
                # cached decode through the one shared forward path.
                if post_cross_attention_callback is not None:
                    x = post_cross_attention_callback(i, x)
                # Capture hidden state for feature prediction
                if self.feature_pred_head is not None and i == self._feature_pred_layer_idx:
                    feature_pred_hidden = x

            if not (fp_logits_only or fp_hidden_only):
                x = norm(x)
                logits = self.lm_head(x[:, -logits_to_keep:] if logits_to_keep else x)
                logits = logits[..., : self.dec_config.vocab_size].float()
                logits = LOGIT_SOFTCAP * t.tanh(logits / LOGIT_SOFTCAP)
                if targets is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                        ignore_index=-1,
                        reduction=loss_reduction,
                    )

            # A complete transformer-head cache must track every committed token,
            # including LM-only calls after speculative feature prediction/rewind.
            cache_head = (
                kv_cache is not None
                and self.num_cache_layers > self.dec_config.n_layer
                and kv_cache.n_layers == self.num_cache_layers
            )
            if needs_fp or cache_head or fp_hidden_only:
                hidden = norm(feature_pred_hidden)
            if needs_fp or cache_head:
                fp_logits = self.apply_feature_prediction_head(
                    hidden, T0, kv_cache=kv_cache, project=needs_fp
                )
            if fp_hidden_only:
                return hidden
            if fp_logits_only:
                return fp_logits

            if targets is not None:
                # Feature prediction auxiliary loss
                if (
                    self.feature_pred_head is not None
                    and self.feature_pred_config is not None
                    and fp_targets is not None
                ):
                    # fp_targets spans all pre-selection extracted features per segment.
                    # When feature_mask is configured, both the live and fused dataloader
                    # paths zero masked-feature columns at source (defense-in-depth); the
                    # FP loss additionally gates masked dims via compute_feature_pred_loss.
                    if seg_doc_id is None:
                        raise ValueError(
                            "EncoderDecoderGPT.forward: seg_doc_id is required "
                            "when fp_targets is provided. Defaulting to all-zeros "
                            "would silently merge unrelated documents in the "
                            "feature-prediction loss (every segment treated as "
                            "doc 0), violating the cross-document filter contract "
                            "in feature_prediction.py. Pass seg_doc_id from the "
                            "dataloader (live + precomputed dataloaders both "
                            "expose it; padding slots use -1). See manifest "
                            "§4.16 / slice-10 map P1-B."
                        )
                    if seg_map is None:
                        raise ValueError(
                            "EncoderDecoderGPT.forward: seg_map is required when "
                            "fp_targets is provided"
                        )
                    t._assert(
                        fp_targets.ndim == 3,
                        "fp_targets must have shape [B, n_seg, feature_prediction_width]",
                    )
                    t._assert(
                        fp_targets.shape[0] == fp_logits.shape[0],
                        "fp_targets batch dimension must equal fp_logits batch dimension",
                    )
                    t._assert(
                        fp_targets.shape[1] == seg_doc_id.shape[1],
                        "fp_targets segment dimension must equal seg_doc_id",
                    )
                    t._assert(
                        fp_targets.shape[2] == self.feature_pred_head.out_features,
                        "fp_targets final dimension must equal feature_pred_head.out_features",
                    )
                    t._assert_async(
                        t.isfinite(fp_targets).all(),
                        "fp_targets must contain only finite values",
                    )
                    t._assert_async(
                        ((fp_targets >= 0.0) & (fp_targets <= 1.0)).all(),
                        "fp_targets values must lie in [0, 1]",
                    )
                    t._assert_async(
                        ((seg_map >= 0) & (seg_map < fp_targets.shape[1])).all(),
                        "seg_map references a segment outside fp_targets",
                    )
                    fp_loss_logits = fp_logits
                    if (
                        self.feature_vocab is not None
                        and self.feature_pred_config.loss_fn == FeaturePredLossFn.MSE
                    ):
                        # The head follows the model compute dtype (normally bf16),
                        # while packed percentile targets are deliberately float32.
                        # CPU MSE accepts that mixed pair in forward but its backward
                        # raises a Float/BFloat16 mismatch. Compact FP is new, so make
                        # its MSE boundary explicitly float32 without perturbing the
                        # literal legacy full-width loss path or public logits dtype.
                        fp_loss_logits = fp_logits.float()
                    fp_loss = compute_feature_pred_loss(
                        fp_loss_logits,
                        seg_map,
                        fp_targets,
                        seg_doc_id,
                        self.feature_pred_config.loss_mode,
                        self.feature_pred_config.loss_fn,
                        feature_mask=(
                            None if self.feature_vocab is not None else self.feature_mask
                        ),
                    )
                    combined = loss + self.feature_pred_config.weight * fp_loss
                    return ForwardResult(loss=combined, lm_loss=loss, feature_pred_loss=fp_loss)
                return loss
            if return_fp_logits:
                return logits, fp_logits
            return logits
        except Exception:
            # K/V written beyond the old prefix are uncommitted and will be
            # overwritten on retry. Restore all counters even if the last
            # attention block already advanced before a later operation failed.
            if kv_cache is not None:
                advanced = kv_cache.get_pos() - T0
                if advanced:
                    kv_cache.rewind(advanced)
            raise

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        adam_betas=(0.8, 0.95),
        scalar_lr=0.5,
        encoder_embedding_lr=0.2,
        encoder_matrix_lr=0.02,
        xattn_matrix_lr=0.02,
        trainable_only=False,
        fp_head_lr=None,
        fp_head_matrix_lr=0.02,
        muon_parameter_packing=False,
    ):
        """Set up optimizer groups while preserving component-specific hyperparameters.

        ``trainable_only`` is an opt-in seam for objectives whose frozen
        parameters must be absent from ``DistMuonAdamW``. Its default preserves
        the historical eligibility behavior.

        ``fp_head_lr`` is the feature-prediction head projection's AdamW
        learning rate. ``None`` means "share ``unembedding_lr``". Linear heads
        therefore retain their historical grouping byte-for-byte. Pass a float
        only when the FP head and the LM head are driven by different objectives,
        as structured-SFT GRPO does once its auxiliary next-token-prediction term
        makes ``lm_head`` trainable.

        ``fp_head_matrix_lr`` is the Muon learning rate for 2-D matrices owned
        by residual feature-prediction blocks. Any non-matrix parameters those
        blocks may acquire remain in the projection's AdamW group.

        ``muon_parameter_packing`` uses persistent contiguous Muon storage.
        Construct only after loading and final device/dtype conversion, before
        the first forward/backward. Reconstruct after replacing parameters or
        changing their storage. Group order, learning rates, and checkpoint
        state remain compatible with the upstream optimizer at the same
        distributed topology.
        """
        if type(muon_parameter_packing) is not bool:
            raise ValueError("muon_parameter_packing must be a boolean")
        model_dim = self.dec_config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # --- Decoder groups (same pattern as GPT) ---
        dec_matrix_params = list(self._decoder_blocks.parameters())
        dec_value_embeds_params = list(self.value_embeds.parameters())
        dec_embedding_params = list(self._decoder_embedding.parameters())
        dec_lm_head_params = list(self.lm_head.parameters())
        all_fp_head_params = (
            list(self.feature_pred_head.parameters()) if self.feature_pred_head is not None else []
        )
        fp_head_matrix_params: list[nn.Parameter] = []
        fp_head_adamw_params = all_fp_head_params
        if isinstance(self.feature_pred_head, ResidualFeaturePredictionHead):
            residual_param_ids = {
                id(parameter) for parameter in self.feature_pred_head.residual_blocks.parameters()
            }
            fp_head_matrix_params = [
                parameter
                for parameter in all_fp_head_params
                if id(parameter) in residual_param_ids and parameter.ndim == 2
            ]
            fp_head_matrix_ids = {id(parameter) for parameter in fp_head_matrix_params}
            fp_head_adamw_params = [
                parameter
                for parameter in all_fp_head_params
                if id(parameter) not in fp_head_matrix_ids
            ]
        dec_resid_params = [self.resid_lambdas]
        dec_x0_params = [self.x0_lambdas]

        # --- Encoder groups ---
        enc_embed_params = (
            list(self.encoder.feature_embed.parameters())
            + list(self.encoder.magnitude_embed.parameters())
            + list(self.encoder.value_embeds.parameters())
        )
        enc_matrix_params = list(self.encoder.blocks.parameters())

        # --- Cross-attention groups ---
        xattn_matrix_params = list(self.cross_attn_layers.parameters())

        if trainable_only:
            dec_matrix_params = [p for p in dec_matrix_params if p.requires_grad]
            dec_value_embeds_params = [p for p in dec_value_embeds_params if p.requires_grad]
            dec_embedding_params = [p for p in dec_embedding_params if p.requires_grad]
            dec_lm_head_params = [p for p in dec_lm_head_params if p.requires_grad]
            fp_head_adamw_params = [p for p in fp_head_adamw_params if p.requires_grad]
            fp_head_matrix_params = [p for p in fp_head_matrix_params if p.requires_grad]
            dec_resid_params = [p for p in dec_resid_params if p.requires_grad]
            dec_x0_params = [p for p in dec_x0_params if p.requires_grad]
            enc_embed_params = [p for p in enc_embed_params if p.requires_grad]
            enc_matrix_params = [p for p in enc_matrix_params if p.requires_grad]
            xattn_matrix_params = [p for p in xattn_matrix_params if p.requires_grad]

        eligible_fp_head_params = (
            [parameter for parameter in all_fp_head_params if parameter.requires_grad]
            if trainable_only
            else all_fp_head_params
        )
        grouped_fp_head_params = fp_head_adamw_params + fp_head_matrix_params
        grouped_fp_head_ids = [id(parameter) for parameter in grouped_fp_head_params]
        assert len(grouped_fp_head_ids) == len(set(grouped_fp_head_ids)) and set(
            grouped_fp_head_ids
        ) == {id(parameter) for parameter in eligible_fp_head_params}, (
            "Every eligible feature-prediction head parameter must belong to exactly one "
            "optimizer group"
        )

        # Verify all params are accounted for
        all_grouped = (
            dec_matrix_params
            + dec_value_embeds_params
            + dec_embedding_params
            + dec_lm_head_params
            + fp_head_adamw_params
            + fp_head_matrix_params
            + dec_resid_params
            + dec_x0_params
            + enc_embed_params
            + enc_matrix_params
            + xattn_matrix_params
        )
        expected_parameters = (
            [parameter for parameter in self.parameters() if parameter.requires_grad]
            if trainable_only
            else list(self.parameters())
        )
        assert len(expected_parameters) == len(all_grouped), (
            f"Parameter count mismatch: model has {len(expected_parameters)} eligible "
            f"parameters but grouped {len(all_grouped)}"
        )

        # Scale LR for AdamW parameters
        dmodel_lr_scale = (model_dim / BASE_MODEL_DIM) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters: 1/sqrt({model_dim}/{BASE_MODEL_DIM}) = {dmodel_lr_scale:.6f}"
        )
        if grouped_fp_head_params:
            fp_adamw_lr = (
                unembedding_lr if fp_head_lr is None else fp_head_lr
            ) * dmodel_lr_scale
            fp_optimizer_groups = []
            if fp_head_adamw_params:
                fp_optimizer_groups.append(
                    f"{OptimizerKind.ADAMW.value} lr={fp_adamw_lr:g} "
                    f"params={len(fp_head_adamw_params)}"
                )
            if fp_head_matrix_params:
                fp_optimizer_groups.append(
                    f"{OptimizerKind.MUON.value} lr={fp_head_matrix_lr:g} "
                    f"matrices={len(fp_head_matrix_params)}"
                )
            print0(
                "Feature-prediction head optimizer groups: " + ", ".join(fp_optimizer_groups)
            )

        param_groups = [
            # Decoder AdamW groups
            {
                "kind": OptimizerKind.ADAMW,
                "params": dec_lm_head_params,
                "lr": unembedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": OptimizerKind.ADAMW,
                "params": dec_embedding_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": OptimizerKind.ADAMW,
                "params": dec_value_embeds_params,
                "lr": embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": OptimizerKind.ADAMW,
                "params": dec_resid_params,
                "lr": scalar_lr * 0.01,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            {
                "kind": OptimizerKind.ADAMW,
                "params": dec_x0_params,
                "lr": scalar_lr,
                "betas": (0.96, 0.95),
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
            # Encoder AdamW groups (embeddings + projection)
            {
                "kind": OptimizerKind.ADAMW,
                "params": enc_embed_params,
                "lr": encoder_embedding_lr * dmodel_lr_scale,
                "betas": adam_betas,
                "eps": 1e-10,
                "weight_decay": 0.0,
            },
        ]
        if trainable_only:
            param_groups = [group for group in param_groups if group["params"]]

        if fp_head_adamw_params:
            param_groups.append(
                {
                    "kind": OptimizerKind.ADAMW,
                    "params": fp_head_adamw_params,
                    "lr": (unembedding_lr if fp_head_lr is None else fp_head_lr) * dmodel_lr_scale,
                    "betas": adam_betas,
                    "eps": 1e-10,
                    "weight_decay": 0.0,
                }
            )

        # Feature-prediction residual-block Muon groups (grouped by shape)
        for shape in sorted({p.shape for p in fp_head_matrix_params}):
            group_params = [p for p in fp_head_matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": OptimizerKind.MUON,
                    "params": group_params,
                    "lr": fp_head_matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )

        # Decoder Muon groups (grouped by shape)
        for shape in sorted({p.shape for p in dec_matrix_params}):
            group_params = [p for p in dec_matrix_params if p.shape == shape]
            param_groups.append(
                {
                    "kind": OptimizerKind.MUON,
                    "params": group_params,
                    "lr": matrix_lr,
                    "momentum": 0.95,
                    "ns_steps": 5,
                    "beta2": 0.95,
                    "weight_decay": weight_decay,
                }
            )

        # Encoder Muon groups (grouped by shape)
        if enc_matrix_params:
            for shape in sorted({p.shape for p in enc_matrix_params}):
                group_params = [p for p in enc_matrix_params if p.shape == shape]
                param_groups.append(
                    {
                        "kind": OptimizerKind.MUON,
                        "params": group_params,
                        "lr": encoder_matrix_lr,
                        "momentum": 0.95,
                        "ns_steps": 5,
                        "beta2": 0.95,
                        "weight_decay": weight_decay,
                    }
                )

        # Cross-attention Muon groups (grouped by shape)
        if xattn_matrix_params:
            for shape in sorted({p.shape for p in xattn_matrix_params}):
                group_params = [p for p in xattn_matrix_params if p.shape == shape]
                param_groups.append(
                    {
                        "kind": OptimizerKind.MUON,
                        "params": group_params,
                        "lr": xattn_matrix_lr,
                        "momentum": 0.95,
                        "ns_steps": 5,
                        "beta2": 0.95,
                        "weight_decay": weight_decay,
                    }
                )

        param_groups = _coalesce_compatible_muon_groups(param_groups)

        if muon_parameter_packing:
            PackedFactory = DistPackedMuonAdamW if ddp else PackedMuonAdamW
            optimizer = PackedFactory(param_groups, owner=self)
        else:
            Factory = DistMuonAdamW if ddp else MuonAdamW
            optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def num_scaling_params(self):
        """Return detailed parameter counts for scaling law analysis."""
        wte = sum(p.numel() for p in self._decoder_embedding.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(
            p.numel() for p in self._decoder_blocks.parameters()
        )
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel()
        encoder = sum(p.numel() for p in self.encoder.parameters())
        cross_attn = sum(p.numel() for p in self.cross_attn_layers.parameters())
        feature_pred_head = (
            sum(p.numel() for p in self.feature_pred_head.parameters())
            if self.feature_pred_head is not None
            else 0
        )
        total = (
            wte
            + value_embeds
            + lm_head
            + transformer_matrices
            + scalars
            + encoder
            + cross_attn
            + feature_pred_head
        )
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "encoder": encoder,
            "cross_attn": cross_attn,
            "feature_pred_head": feature_pred_head,
            "total": total,
        }

    def estimate_flops(
        self,
        avg_features_per_segment: float,
        segment_size: int,
    ) -> float:
        """Return estimated FLOPs per token for the model (forward + backward).

        Convention: 'per token' means per decoder token. Encoder and cross-attention
        FLOPs are amortized across decoder tokens using the ratio
        R = (n_segments * avg_features_per_segment) / sequence_len.

        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +)
        in the forward pass, and 2X that in backward => 2+4=6.
        Cleanest explanation: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4

        On top of the weight-based FLOPs, activation-to-activation matmuls in attention
        (Q@K^T and attn@V) are counted separately: 12 * n_head * head_dim * seq_len
        per layer. This applies to decoder self-attention, cross-attention, and
        encoder self-attention.
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).

        Args:
            avg_features_per_segment: Average number of active encoder features per
                segment. Depends on the thresholding config (e.g., topk=2048).
            segment_size: Decoder segment stride in tokens (e.g., 128). Used to
                compute the number of segments per sequence.
        """
        T = self.dec_config.sequence_len
        Nf = avg_features_per_segment
        n_seg = T / segment_size  # segments per sequence
        R = (n_seg * Nf) / T  # encoder tokens per decoder token

        # --- Decoder-side matmul params (applied per decoder token) ---
        dec_block_params = sum(
            p.numel() for p in self._decoder_blocks.parameters()
        )
        lm_head_params = self.lm_head.weight.numel()
        xattn_dec_params = sum(
            layer.c_q.weight.numel() + layer.c_proj.weight.numel()
            for layer in self._cross_attention_layer_values()
        )

        # --- Encoder-side matmul params (applied per encoder token) ---
        enc_block_params = sum(p.numel() for p in self.encoder.blocks.parameters())
        xattn_enc_params = sum(
            layer.c_kv.weight.numel()
            for layer in self._cross_attention_layer_values()
        )

        # Weight-based FLOPs (6 per param: 2 forward + 4 backward)
        dec_weight_flops = 6 * (dec_block_params + lm_head_params + xattn_dec_params)
        enc_weight_flops = 6 * (enc_block_params + xattn_enc_params) * R

        # --- Decoder self-attention activation FLOPs (per decoder token) ---
        # 12 * n_head * head_dim * effective_seq per layer (forward + backward)
        h = self.dec_config.n_head
        d = self.dec_config.n_embd // self.dec_config.n_head
        dec_attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]
            effective_seq = T if window < 0 else min(window, T)
            dec_attn_flops += 12 * h * d * effective_seq

        # --- Cross-attention activation FLOPs (per decoder token) ---
        # Each decoder token attends to Nf encoder features per cross-attn layer.
        # Q@K^T + attn@V: forward = 4 * n_head * head_dim * Nf, x3 for backward = 12
        n_xattn_layers = len(self.cross_attn_layers)
        if n_xattn_layers > 0:
            xattn_layer = self._cross_attention_layer_values()[0]
            xattn_h = xattn_layer.n_head
            xattn_d = xattn_layer.head_dim
            xattn_attn_flops = (
                12 * xattn_h * xattn_d * Nf * n_xattn_layers
            )
        else:
            xattn_attn_flops = 0

        # --- Encoder self-attention activation FLOPs (per encoder token, amortized) ---
        # Bidirectional: each encoder token attends to all Nf tokens.
        # Per encoder token per layer: 12 * enc_n_head * enc_head_dim * Nf
        # Total per decoder token: above * enc_n_layer * R
        enc_cfg = self.encoder.config
        if enc_cfg.n_layer > 0:
            enc_attn_flops = 12 * enc_cfg.n_head * enc_cfg.head_dim * Nf * enc_cfg.n_layer * R
        else:
            enc_attn_flops = 0

        # Feature prediction head (per decoder token, same as any weight matmul)
        if self.feature_pred_head is not None:
            fp_head_params = sum(
                parameter.numel()
                for parameter in self.feature_pred_head.parameters()
                if parameter.ndim == 2
            )
            fp_head_flops = 6 * fp_head_params
            if self.num_cache_layers > self.dec_config.n_layer:
                window = self.window_sizes[self._feature_pred_layer_idx][0]
                effective_seq = T if window < 0 else min(window, T)
                fp_head_flops += 12 * h * d * effective_seq * (
                    self.num_cache_layers - self.dec_config.n_layer
                )
        else:
            fp_head_flops = 0

        return (
            dec_weight_flops
            + enc_weight_flops
            + dec_attn_flops
            + xattn_attn_flops
            + enc_attn_flops
            + fp_head_flops
        )
