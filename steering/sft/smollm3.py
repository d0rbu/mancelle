"""Immutable SmolLM3 source and chat-format contract for standard SFT."""

from transformers import SmolLM3ForCausalLM

from steering.sft.hf_contract import HFDecoderContract, HFDecoderFamily

BEGIN_OF_TEXT_ID = 128000
SOURCE_END_OF_TEXT_ID = 128001
FINETUNE_PAD_ID = 128004
IM_START_ID = 128011
IM_END_ID = 128012
TOOL_RESPONSE_OPEN_ID = 128013
TOOL_RESPONSE_CLOSE_ID = 128014
TOOL_CALL_OPEN_ID = 128015
TOOL_CALL_CLOSE_ID = 128016

BEGIN_OF_TEXT = "<|begin_of_text|>"
SOURCE_END_OF_TEXT = "<|end_of_text|>"
FINETUNE_PAD = "<|finetune_right_pad_id|>"
IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
TOOL_RESPONSE_OPEN = "<tool_response>"
TOOL_RESPONSE_CLOSE = "</tool_response>"
TOOL_CALL_OPEN = "<tool_call>"
TOOL_CALL_CLOSE = "</tool_call>"

# The immutable base tokenizer has no template. Transformers
# apply_chat_template remains the renderer and mask engine; this template
# defines the repository's SmolLM3 standard-SFT wire format.
SMOLLM3_CHAT_TEMPLATE = """{{- bos_token -}}
{%- for message in messages -%}
{{- '<|im_start|>' + message['role'] + '\n' -}}
{%- if message['role'] == 'system' or message['role'] == 'user' -%}
{{- message['content'] + '<|im_end|>\n' -}}
{%- elif message['role'] == 'assistant' -%}
{%- if message['content'] is string -%}
{%- generation -%}{{- message['content'] + '<|im_end|>\n' -}}{%- endgeneration -%}
{%- else -%}
{%- set ns = namespace(pending='') -%}
{%- for part in message['content'] -%}
{%- if part['type'] == 'text' -%}
{%- set ns.pending = ns.pending + part['text'] -%}
{%- elif part['type'] == 'python' -%}
{%- set ns.pending = ns.pending + '<tool_call>' + part['text'] + '</tool_call>' -%}
{%- elif part['type'] == 'python_output' -%}
{%- generation -%}{{- ns.pending -}}{%- endgeneration -%}
{{- '<tool_response>' + part['text'] + '</tool_response>' -}}
{%- set ns.pending = '' -%}
{%- endif -%}
{%- endfor -%}
{%- generation -%}{{- ns.pending + '<|im_end|>\n' -}}{%- endgeneration -%}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}"""

SMOLLM3_CONTRACT = HFDecoderContract(
    family=HFDecoderFamily.SMOLLM3,
    model_type="smollm3",
    architecture="SmolLM3ForCausalLM",
    model_class=SmolLM3ForCausalLM,
    model_vocab_size=128256,
    tokenizer_vocab_size=128256,
    source_bos_token_id=BEGIN_OF_TEXT_ID,
    source_eos_token_id=SOURCE_END_OF_TEXT_ID,
    source_pad_token_id=None,
    generation_bos_token_id=BEGIN_OF_TEXT_ID,
    generation_eos_token_id=IM_END_ID,
    generation_pad_token_id=FINETUNE_PAD_ID,
    bos_token=BEGIN_OF_TEXT,
    eos_token=IM_END,
    pad_token=FINETUNE_PAD,
    chat_template=SMOLLM3_CHAT_TEMPLATE,
    required_token_ids={
        BEGIN_OF_TEXT: BEGIN_OF_TEXT_ID,
        SOURCE_END_OF_TEXT: SOURCE_END_OF_TEXT_ID,
        FINETUNE_PAD: FINETUNE_PAD_ID,
        IM_START: IM_START_ID,
        IM_END: IM_END_ID,
        TOOL_RESPONSE_OPEN: TOOL_RESPONSE_OPEN_ID,
        TOOL_RESPONSE_CLOSE: TOOL_RESPONSE_CLOSE_ID,
        TOOL_CALL_OPEN: TOOL_CALL_OPEN_ID,
        TOOL_CALL_CLOSE: TOOL_CALL_CLOSE_ID,
    },
    structural_token_ids={
        "begin_of_text": BEGIN_OF_TEXT_ID,
        "source_end_of_text": SOURCE_END_OF_TEXT_ID,
        "finetune_pad": FINETUNE_PAD_ID,
        "im_start": IM_START_ID,
        "im_end": IM_END_ID,
        "tool_response_open": TOOL_RESPONSE_OPEN_ID,
        "tool_response_close": TOOL_RESPONSE_CLOSE_ID,
        "tool_call_open": TOOL_CALL_OPEN_ID,
        "tool_call_close": TOOL_CALL_CLOSE_ID,
    },
    source_config_fields={
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "max_position_embeddings": 4096,
    },
    tied_embeddings=True,
)
