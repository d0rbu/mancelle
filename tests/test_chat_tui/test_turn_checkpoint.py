"""Real checkpoint save/load through Chat's loader, then two turn epochs.

The tiny weights, metadata and checkpoint reconstruction are real. Only the
tokenizer asset and Gemma extraction are test doubles; raw-SHA lineage/producer
integration has its separate tests in test_turn_setup.
"""

from dataclasses import asdict
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import nanochat.checkpoint_manager as checkpoint_manager
import nanochat.flash_attention as flash_attention
from nanochat.checkpoint_manager import save_checkpoint
from nanochat.flash_attention import FlashAttnImpl
from scripts.chat_tui.setup import load_chat_target_model
from scripts.chat_tui.turn_session import TurnChatSession
from steering.conditional.checkpoint_feature_contract import CheckpointFeatureContract
from steering.segmentation import SegmentMode
from tests.test_chat_tui.test_turn_session import TurnExtraction, build_turn


@pytest.mark.parametrize("cadence", [1, 4])
def test_checkpoint_roundtrip_preserves_weights_and_two_turn_runtime(
    monkeypatch, tmp_path, cadence
):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    control, tokenizer, _ = build_turn(cadence=cadence)
    model = control._driver._model
    metadata = {
        "step": 7,
        "model_config": asdict(model.dec_config),
        "encoder_config": asdict(model.enc_config),
        "xattn_config": asdict(model.xattn_config),
        "feature_prediction_config": asdict(model.feature_pred_config),
        "num_buckets": 2,
        "sft_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value,
        "turn_self_cond_segment_size_rustbpe": cadence,
        "user_config": {
            "negative_selection": {
                **asdict(control._driver._setup.output_negative_selection),
                "distribution": control._driver._setup.output_negative_selection.distribution.value,
            }
        },
    }
    save_checkpoint(str(tmp_path), 7, model.state_dict(), None, metadata, rank=0)
    setup: Any = SimpleNamespace(**vars(control._driver._setup))
    setup.meta = metadata
    setup.target_meta_path = tmp_path / "meta_000007.json"
    setup.feature_contract = CheckpointFeatureContract(feature_mask=None, resolved_clusters=None)
    monkeypatch.setattr(tokenizer, "get_vocab_size", lambda: model.dec_config.vocab_size)
    monkeypatch.setattr(checkpoint_manager, "get_tokenizer", lambda: tokenizer)
    loaded, loaded_tokenizer = load_chat_target_model(setup, "cpu")
    for key, value in model.state_dict().items():
        assert t.equal(value, loaded.state_dict()[key]), key
    session = TurnChatSession(
        setup=setup,
        model=loaded,
        tokenizer=loaded_tokenizer,
        device="cpu",
        config=control._driver._config,
        extraction=TurnExtraction(loaded_tokenizer),
    )
    for user in ("hi", "next"):
        control.send_user_turn(user, lambda _: None)
        session.send_user_turn(user, lambda _: None)
        assert session.conversation_tokens() == control.conversation_tokens()
        assert session.ledger() == control.ledger()
        assert session._driver._kv_cache is not None
        assert t.equal(session._driver._kv_cache.k_cache, control._driver._kv_cache.k_cache)
        assert t.equal(session._driver._kv_cache.v_cache, control._driver._kv_cache.v_cache)
