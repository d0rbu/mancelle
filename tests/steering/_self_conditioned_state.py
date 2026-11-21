"""Test-only seed state for direct self-conditioning provider tests."""

from __future__ import annotations

import torch as t

from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
    SelfCondState,
)


def build_test_self_cond_state(
    provider: SelfConditionedFeatureProvider,
    features: dict[int, int] | None = None,
) -> SelfCondState:
    """Encode one seed slot so tests can exercise ``step`` in isolation.

    Production obtains its initial state from real-feature prefill. Direct
    provider tests need a minimal state without recreating that full pipeline,
    so the fixture intentionally uses the provider's private encoding helpers.
    """
    if features is None:
        candidate_ids = t.arange(provider.cfg.n_sae_features, dtype=t.int64)
        vocab = provider._model.feature_vocab
        if vocab is not None:
            candidate_ids = vocab.new_to_old_tensor()
        feature_mask = provider.cfg.feature_mask
        if feature_mask is not None:
            candidate_ids = candidate_ids[feature_mask.keep_mask(candidate_ids)]
        if candidate_ids.numel() == 0:
            raise ValueError("test provider has no valid feature for its real seed slot")
        features = {int(candidate_ids[0].item()): 0}
    feature_ids, magnitudes, mask = provider._build_segment_inputs(features)
    encoder_out = provider._encode_one_segment(
        feature_ids,
        magnitudes,
        mask,
    )
    return SelfCondState(
        encoder_out=encoder_out,
        target_slot_idx=0,
    )
