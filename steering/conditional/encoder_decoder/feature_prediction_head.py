"""Residual feature-prediction capacity with the legacy projection state keys."""

import torch as t
import torch.nn as nn
import torch.nn.functional as F

from nanochat.gpt import MLP, Block, GPTConfig, Linear, norm
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    validate_feature_prediction_head_layers,
)
from steering.conditional.encoder_decoder.typed_module_list import TypedModuleList


class ResidualFeaturePredictionHead(nn.Module):
    """Independent pre-norm residual blocks and the retained linear projection.

    Callers already normalize the captured decoder hidden states. This module
    adds only each residual branch's normalization, with no final normalization.
    Projection parameters remain at the root, preserving legacy checkpoint keys
    and parameter identity when upgrading a strictly loaded linear head.

    This deliberately does not inherit ``nn.Linear``: conversion passes must not
    replace the entire nonlinear module as though it were a single projection.
    """

    def __init__(
        self,
        projection: Linear,
        head_layers: int,
        block: FeaturePredictionBlock = FeaturePredictionBlock.MLP,
        *,
        decoder_config: GPTConfig | None = None,
    ):
        super().__init__()
        depth = validate_feature_prediction_head_layers(head_layers)
        if depth == 0:
            raise ValueError("ResidualFeaturePredictionHead requires positive head_layers")
        self.block = FeaturePredictionBlock(block)
        if self.block == FeaturePredictionBlock.TRANSFORMER and (
            decoder_config is None or decoder_config.n_embd != projection.in_features
        ):
            raise ValueError("transformer feature head requires the matching decoder configuration")
        self.in_features = projection.in_features
        self.out_features = projection.out_features
        self.weight = projection.weight
        self.bias = projection.bias
        device = self.weight.device
        devices = [device] if device.type == "cuda" else []
        # Linear constructors consume random numbers on real devices. Include
        # allocation and initialization in the fork so fresh SFT upgrades do not
        # perturb subsequent dropout/selection/training RNG draws.
        with t.random.fork_rng(devices=devices), t.device(device):
            self.residual_blocks: TypedModuleList[MLP] | TypedModuleList[Block]
            if self.block == FeaturePredictionBlock.MLP:
                self.residual_blocks = TypedModuleList(
                    [MLP(GPTConfig(n_embd=self.in_features)) for _ in range(depth)],
                    module_type=MLP,
                    item_label="feature prediction residual block",
                ).to(dtype=self.weight.dtype)
            else:
                assert decoder_config is not None
                blocks = [Block(decoder_config, decoder_config.n_layer + i) for i in range(depth)]
                for layer in blocks:
                    layer.attn.ve_gate = None
                self.residual_blocks = TypedModuleList(
                    blocks, module_type=Block, item_label="feature prediction transformer block"
                ).to(dtype=self.weight.dtype)
            self.init_residual_weights()
        self.train(projection.training)

    @t.no_grad()
    def init_residual_weights(self) -> None:
        """Initialize only new branches; never alter the retained projection."""
        device = self.weight.device
        devices = [device] if device.type == "cuda" else []
        with t.random.fork_rng(devices=devices):
            scale = (3 / self.in_features) ** 0.5
            for block in self.residual_blocks:
                if isinstance(block, Block):
                    nn.init.uniform_(block.attn.c_qkv.weight, -scale, scale)
                    nn.init.zeros_(block.attn.c_proj.weight)
                    mlp = block.mlp
                else:
                    mlp = block
                nn.init.uniform_(mlp.c_fc.weight, -scale, scale)
                nn.init.zeros_(mlp.c_proj.weight)

    def forward(
        self,
        x: t.Tensor,
        *,
        cos_sin: tuple[t.Tensor, t.Tensor] | None = None,
        window_size: tuple[int, int] | None = None,
        kv_cache=None,
        project: bool = True,
    ) -> t.Tensor:
        for block in self.residual_blocks:
            if isinstance(block, Block):
                if cos_sin is None or window_size is None:
                    raise ValueError("transformer feature head requires decoder RoPE and window context")
                x = block(x, None, cos_sin, window_size, kv_cache)
            else:
                x = x + block(norm(x))
        if not project:
            return x
        return F.linear(
            x,
            self.weight.to(dtype=x.dtype),
            self.bias.to(dtype=x.dtype) if self.bias is not None else None,
        )
