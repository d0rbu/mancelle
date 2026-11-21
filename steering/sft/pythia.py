"""Immutable Pythia source and chat-format contract for standard SFT."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from tokenizers import processors
from transformers import GPTNeoXForCausalLM, PreTrainedTokenizerBase

from steering.sft.hf_contract import HFDecoderContract, HFDecoderFamily

PYTHIA_END_OF_TEXT_ID = 0
PYTHIA_PAD_ID = 1
PYTHIA_END_OF_TEXT = "<|endoftext|>"
PYTHIA_PAD = "<|padding|>"


class PythiaModelId(StrEnum):
    """Pinned Hugging Face Pythia sizes supported by standard SFT."""

    PYTHIA_2_8B = "EleutherAI/pythia-2.8b"
    PYTHIA_1_4B = "EleutherAI/pythia-1.4b"


@dataclass(frozen=True)
class PythiaGeometry:
    """Source-config fields that vary across supported Pythia sizes."""

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    max_position_embeddings: int = 2048
    use_parallel_residual: bool = True

    def source_config_fields(self) -> MappingProxyType[str, Any]:
        return MappingProxyType(
            {
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size,
                "num_hidden_layers": self.num_hidden_layers,
                "num_attention_heads": self.num_attention_heads,
                "max_position_embeddings": self.max_position_embeddings,
                "use_parallel_residual": self.use_parallel_residual,
            }
        )


PYTHIA_GEOMETRIES = MappingProxyType(
    {
        PythiaModelId.PYTHIA_2_8B: PythiaGeometry(
            hidden_size=2560,
            intermediate_size=10240,
            num_hidden_layers=32,
            num_attention_heads=32,
        ),
        PythiaModelId.PYTHIA_1_4B: PythiaGeometry(
            hidden_size=2048,
            intermediate_size=8192,
            num_hidden_layers=24,
            num_attention_heads=16,
        ),
    }
)
_PYTHIA_SOURCE_CONFIG_FIELDS = MappingProxyType(
    {
        model_id.value: geometry.source_config_fields()
        for model_id, geometry in PYTHIA_GEOMETRIES.items()
    }
)

# The immutable base tokenizer has no template. Transformers
# apply_chat_template remains the renderer and mask engine; this template
# defines the repository's Pythia standard-SFT wire format.
PYTHIA_CHAT_TEMPLATE = """{{- bos_token -}}
{%- for message in messages -%}
{{- message['role'] + ':\n' -}}
{%- if message['role'] == 'system' or message['role'] == 'user' -%}
{{- message['content'] + eos_token + '\n' -}}
{%- elif message['role'] == 'assistant' -%}
{%- if message['content'] is string -%}
{%- generation -%}{{- message['content'] + eos_token + '\n' -}}{%- endgeneration -%}
{%- else -%}
{%- set ns = namespace(pending='') -%}
{%- for part in message['content'] -%}
{%- if part['type'] == 'text' -%}
{%- set ns.pending = ns.pending + part['text'] -%}
{%- elif part['type'] == 'python' -%}
{%- set ns.pending = ns.pending + '<tool_call>' + part['text'] + '</tool_call>' -%}
{%- elif part['type'] == 'python_output' -%}
{%- generation -%}{{- ns.pending + '\n' -}}{%- endgeneration -%}
{{- '<tool_response>' + part['text'] + '</tool_response>' -}}
{%- set ns.pending = '\n' -%}
{%- endif -%}
{%- endfor -%}
{%- generation -%}{{- ns.pending + eos_token + '\n' -}}{%- endgeneration -%}
{%- endif -%}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- 'assistant:\n' -}}{%- endif -%}"""


def prepare_pythia_tokenizer(tokenizer: PreTrainedTokenizerBase) -> None:
    """Keep ByteLevel offsets token-complete for generation-span masks."""
    backend = tokenizer.backend_tokenizer
    post_processor = backend.post_processor
    if not isinstance(post_processor, processors.ByteLevel):
        raise ValueError(
            "Pythia tokenizer must use a ByteLevel post-processor; "
            f"got {type(post_processor).__name__}"
        )
    state = json.loads(post_processor.__getstate__())
    if state.get("add_prefix_space") is not False or state.get("use_regex") is not True:
        raise ValueError(f"Pythia ByteLevel post-processor contract drifted: {state!r}")
    # The source's trim_offsets=True drops boundary whitespace from offset
    # mappings. Transformers then cannot map Jinja generation spans to tokens:
    # a leading-space answer may become all-zero, while a multipart span whose
    # final space merges with '<' can run through an unsupervised tool response.
    # Disabling offset trimming changes offsets only, not token IDs or text.
    backend.post_processor = processors.ByteLevel(
        add_prefix_space=False,
        trim_offsets=False,
        use_regex=True,
    )


PYTHIA_CONTRACT = HFDecoderContract(
    family=HFDecoderFamily.PYTHIA,
    model_type="gpt_neox",
    architecture="GPTNeoXForCausalLM",
    model_class=GPTNeoXForCausalLM,
    model_vocab_size=50304,
    tokenizer_vocab_size=50277,
    source_bos_token_id=PYTHIA_END_OF_TEXT_ID,
    source_eos_token_id=PYTHIA_END_OF_TEXT_ID,
    source_pad_token_id=None,
    generation_bos_token_id=PYTHIA_END_OF_TEXT_ID,
    generation_eos_token_id=PYTHIA_END_OF_TEXT_ID,
    generation_pad_token_id=PYTHIA_PAD_ID,
    bos_token=PYTHIA_END_OF_TEXT,
    eos_token=PYTHIA_END_OF_TEXT,
    pad_token=PYTHIA_PAD,
    chat_template=PYTHIA_CHAT_TEMPLATE,
    required_token_ids={
        PYTHIA_END_OF_TEXT: PYTHIA_END_OF_TEXT_ID,
        PYTHIA_PAD: PYTHIA_PAD_ID,
    },
    structural_token_ids={
        "end_of_text": PYTHIA_END_OF_TEXT_ID,
        "padding": PYTHIA_PAD_ID,
    },
    source_config_fields=_PYTHIA_SOURCE_CONFIG_FIELDS[PythiaModelId.PYTHIA_2_8B.value],
    tied_embeddings=False,
    prepare_tokenizer=prepare_pythia_tokenizer,
    source_config_fields_by_model_id=_PYTHIA_SOURCE_CONFIG_FIELDS,
)
