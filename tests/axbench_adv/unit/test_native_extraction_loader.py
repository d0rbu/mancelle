"""Fail-closed native extraction admission before model loading."""

from __future__ import annotations

import pytest

from steering.eval.dense_feature_targets import load_weight_free_gemma3_extraction_stack
from steering.sae_resolver import ResolvedSAE


@pytest.mark.parametrize(
    ("sae_model", "hook_name", "device", "sae_layer"),
    [
        ("gemma-2-2b", "blocks.22.hook_resid_post", "cuda:0", 22),
        ("gemma-3-4b-pt", "blocks.22.hook_resid_post", "cpu", 22),
        ("gemma-3-4b-pt", "blocks.22.hook_resid_post", "cuda:0", None),
        ("gemma-3-4b-pt", "blocks.21.hook_resid_post", "cuda:0", 22),
        ("gemma-3-4b-pt", "blocks.22.hook_mlp_out", "cuda:0", 22),
    ],
)
def test_native_loader_rejects_unqualified_source(
    sae_model: str, hook_name: str, device: str, sae_layer: int | None
) -> None:
    resolved = ResolvedSAE("unused-release", "unused-id", hook_name)
    with pytest.raises(ValueError, match="Gemma-3|matching residual-post SAE"):
        load_weight_free_gemma3_extraction_stack(
            sae_model=sae_model,
            resolved_sae=resolved,
            device=device,
            native=True,
            sae_layer=sae_layer,
        )
