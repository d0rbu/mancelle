"""Live immutable HF source contracts; downloads configs/tokenizers only."""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
from hydra import compose, initialize_config_dir
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from steering.sft.hf_contract import HFDecoderFamily
from steering.sft.hf_decoder import load_hf_config_and_tokenizer
from steering.sft.token_bytes import compute_hf_token_bytes

_CONFIG_DIR = str(Path(__file__).resolve().parents[2] / "configs" / "training")


def _source(config_name: str):
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        cfg = compose(config_name=config_name)
    model_id = str(cfg.huggingface.model_id)
    revision = str(cfg.huggingface.revision)
    source_tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_id,
            revision=revision,
            trust_remote_code=False,
        ),
    )
    assert source_tokenizer.chat_template is None
    config, tokenizer, contract = load_hf_config_and_tokenizer(
        model_id=model_id,
        revision=revision,
    )
    assert config._commit_hash == revision
    return tokenizer, contract


@pytest.mark.api
def test_pinned_smollm3_config_and_tokenizer_contract() -> None:
    tokenizer, contract = _source("chat_sft_smollm3_stage1_step_40000")
    assert contract.family is HFDecoderFamily.SMOLLM3
    assert (
        tokenizer.tokenizer.bos_token_id,
        tokenizer.tokenizer.eos_token_id,
        tokenizer.tokenizer.pad_token_id,
    ) == (128000, 128012, 128004)
    assert tokenizer.get_vocab_size() == 128256
    emoji_ids = tokenizer.encode("😀")
    assert emoji_ids == [76460, 222]
    token_bytes = compute_hf_token_bytes(tokenizer)
    assert sum(token_bytes[token_id] for token_id in emoji_ids) == 4
    conversation = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "python", "text": "x()"},
                    {"type": "python_output", "text": "ok"},
                    {"type": "text", "text": "After"},
                ],
            },
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation)
    assert tokenizer.decode(ids) == (
        "<|begin_of_text|><|im_start|>system\nSystem<|im_end|>\n"
        "<|im_start|>user\nQuestion?<|im_end|>\n"
        "<|im_start|>assistant\nBefore<tool_call>x()</tool_call>"
        "<tool_response>ok</tool_response>After<|im_end|>\n"
    )
    assert (
        tokenizer.decode([token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled])
        == "Before<tool_call>x()</tool_call>After<|im_end|>\n"
    )
    unicode_ids, unicode_mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "Question?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Before😀"},
                        {"type": "python", "text": "x()"},
                        {"type": "python_output", "text": "4"},
                        {"type": "text", "text": " 😀"},
                    ],
                },
            ]
        }
    )
    assert (
        tokenizer.decode(
            [
                token_id
                for token_id, enabled in zip(unicode_ids, unicode_mask, strict=True)
                if enabled
            ]
        )
        == "Before😀<tool_call>x()</tool_call> 😀<|im_end|>\n"
    )


@pytest.mark.api
@pytest.mark.parametrize(
    "config_name",
    ["chat_sft_pythia_2_8b", "chat_sft_pythia_1_4b"],
)
def test_pinned_pythia_config_and_tokenizer_contract(config_name: str) -> None:
    tokenizer, contract = _source(config_name)
    assert contract.family is HFDecoderFamily.PYTHIA
    assert contract.model_vocab_size == 50304
    assert tokenizer.get_vocab_size() == 50277
    assert tokenizer.tokenizer.all_special_ids == [0, 1]
    assert (
        tokenizer.tokenizer.bos_token_id,
        tokenizer.tokenizer.eos_token_id,
        tokenizer.tokenizer.pad_token_id,
    ) == (0, 0, 1)
    emoji_ids = tokenizer.encode("😀")
    assert emoji_ids == [14931, 235, 211]
    token_bytes = compute_hf_token_bytes(tokenizer)
    assert sum(token_bytes[token_id] for token_id in emoji_ids) == 4
    conversation = {
        "messages": [
            {"role": "system", "content": "System"},
            {"role": "user", "content": "Question?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Before"},
                    {"type": "python", "text": "x()"},
                    {"type": "python_output", "text": "ok"},
                    {"type": "text", "text": "After"},
                ],
            },
        ]
    }
    ids, mask = tokenizer.render_conversation(conversation)
    assert tokenizer.decode(ids) == (
        "<|endoftext|>system:\nSystem<|endoftext|>\n"
        "user:\nQuestion?<|endoftext|>\n"
        "assistant:\nBefore<tool_call>x()</tool_call>"
        "\n<tool_response>ok</tool_response>\nAfter<|endoftext|>\n"
    )
    assert (
        tokenizer.decode([token_id for token_id, enabled in zip(ids, mask, strict=True) if enabled])
        == "Before<tool_call>x()</tool_call>\n\nAfter<|endoftext|>\n"
    )
    post_processor = json.loads(tokenizer.tokenizer.backend_tokenizer.post_processor.__getstate__())
    assert post_processor == {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": True,
    }

    leading_ids, leading_mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "Question?"},
                {"role": "assistant", "content": " answer"},
            ]
        }
    )
    assert tokenizer.encode(" answer") == [3662]
    assert (
        tokenizer.decode(
            [
                token_id
                for token_id, enabled in zip(leading_ids, leading_mask, strict=True)
                if enabled
            ]
        )
        == " answer<|endoftext|>\n"
    )

    gsm_ids, gsm_mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "What is two plus two?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "The result = "},
                        {"type": "python", "text": "2 + 2"},
                        {"type": "python_output", "text": "4"},
                        {"type": "text", "text": "So 4."},
                    ],
                },
            ]
        }
    )
    assert (
        tokenizer.decode(
            [token_id for token_id, enabled in zip(gsm_ids, gsm_mask, strict=True) if enabled]
        )
        == "The result = <tool_call>2 + 2</tool_call>\n\nSo 4.<|endoftext|>\n"
    )
    unicode_ids, unicode_mask = tokenizer.render_conversation(
        {
            "messages": [
                {"role": "user", "content": "Question?"},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Before"},
                        {"type": "python", "text": "x()"},
                        {"type": "python_output", "text": "4"},
                        {"type": "text", "text": " 😀"},
                    ],
                },
            ]
        }
    )
    assert (
        tokenizer.decode(
            [
                token_id
                for token_id, enabled in zip(unicode_ids, unicode_mask, strict=True)
                if enabled
            ]
        )
        == "Before<tool_call>x()</tool_call>\n\n 😀<|endoftext|>\n"
    )
