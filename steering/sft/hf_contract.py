"""Shared types for validated Hugging Face decoder SFT contracts."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from steering.refined_types import LowercaseSha256, NonEmptyStr, NonNegativeInt, PositiveInt

HF_SFT_FORMAT_VERSION = 2
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


class DecoderSFTBackend(StrEnum):
    """Supported decoder-only SFT model implementations."""

    NANOCHAT = "nanochat"
    HUGGINGFACE = "huggingface"


class HFDecoderFamily(StrEnum):
    """Hugging Face decoder contracts supported by the shared adapter."""

    SMOLLM3 = "smollm3"
    PYTHIA = "pythia"


class HFAttentionImplementation(StrEnum):
    """Attention backends admitted by the pinned decoder SFT path."""

    SDPA = "sdpa"
    FLASH_ATTENTION_4 = "flash_attention_4"


class HFDistributedImplementation(StrEnum):
    """Gradient-reduction/optimizer strategies admitted by HF decoder SFT."""

    DIST_MUON_ADAMW = "dist_muon_adamw"
    DDP_FUSED_ADAMW = "ddp_fused_adamw"


TokenizerPreparer = Callable[[PreTrainedTokenizerBase], None]


class HFConfigTokenIds(NamedTuple):
    """One complete BOS/EOS/PAD config identity, never a mix of stages."""

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int | None


@dataclass(frozen=True)
class HFDecoderContract:
    """Architecture-specific facts needed by otherwise generic HF SFT code."""

    family: HFDecoderFamily
    model_type: str
    architecture: str
    model_class: type[PreTrainedModel]
    model_vocab_size: int
    tokenizer_vocab_size: int
    source_bos_token_id: int
    source_eos_token_id: int
    source_pad_token_id: int | None
    generation_bos_token_id: int
    generation_eos_token_id: int
    generation_pad_token_id: int
    bos_token: str
    eos_token: str
    pad_token: str
    chat_template: str
    required_token_ids: Mapping[str, int]
    structural_token_ids: Mapping[str, int]
    source_config_fields: Mapping[str, Any]
    tied_embeddings: bool
    prepare_tokenizer: TokenizerPreparer | None = None
    source_config_fields_by_model_id: Mapping[str, Mapping[str, Any]] | None = None

    def source_config_fields_for_model_id(
        self,
        model_id: str | None,
    ) -> Mapping[str, Any]:
        """Resolve model-size geometry while retaining one family contract."""
        if self.source_config_fields_by_model_id is None or model_id is None:
            return self.source_config_fields
        try:
            return self.source_config_fields_by_model_id[model_id]
        except KeyError as exc:
            raise ValueError(
                f"unsupported {self.family.value} source model_id {model_id!r}; "
                f"supported={sorted(self.source_config_fields_by_model_id)}"
            ) from exc

    @property
    def accepted_source_token_ids(self) -> tuple[HFConfigTokenIds, HFConfigTokenIds]:
        """Admit the pristine source and our portable SFT export unchanged."""
        return (
            HFConfigTokenIds(
                self.source_bos_token_id, self.source_eos_token_id, self.source_pad_token_id
            ),
            HFConfigTokenIds(
                self.generation_bos_token_id,
                self.generation_eos_token_id,
                self.generation_pad_token_id,
            ),
        )

    @property
    def chat_template_sha256(self) -> str:
        return hashlib.sha256(self.chat_template.encode("utf-8")).hexdigest()


def validate_repository_hf_sft_metadata(payload: Any) -> dict[str, Any]:
    """Validate the common subset required from repository HF SFT exports."""

    if not isinstance(payload, dict):
        raise ValueError("sft_metadata must be a mapping")
    try:
        source = payload["source"]
        output = payload["output"]
        precision = payload["precision"]
        batch = payload["batch"]
        vocabulary = payload["vocabulary"]
        if not all(
            isinstance(block, Mapping) for block in (source, output, precision, batch, vocabulary)
        ):
            raise TypeError("nested metadata blocks must be mappings")
        for value in (
            payload["format"],
            payload["backend"],
            payload["family"],
            source["model_id"],
            source["revision"],
            source["architecture"],
            source["model_type"],
            output["architecture"],
            output["model_type"],
        ):
            NonEmptyStr.parse(value)
        PositiveInt.parse(payload["format_version"])
        NonNegativeInt.parse(payload["completed_optimizer_step"])
        PositiveInt.parse(batch["max_seq_len"])
        PositiveInt.parse(vocabulary["model"])
        PositiveInt.parse(vocabulary["tokenizer"])
        LowercaseSha256.parse(payload["chat_template_sha256"])
    except (KeyError, TypeError) as exc:
        raise ValueError(f"invalid repository HF SFT metadata: {exc}") from exc

    if payload["format_version"] != HF_SFT_FORMAT_VERSION:
        raise ValueError(
            f"sft_metadata.format_version must equal {HF_SFT_FORMAT_VERSION}, "
            f"got {payload['format_version']!r}"
        )
    if payload.get("resumable") is not False:
        raise ValueError("sft_metadata.resumable must be exactly false")
    revision = source.get("revision")
    if _COMMIT_RE.fullmatch(revision) is None:
        raise ValueError(
            "sft_metadata.source.revision must be a pinned 40-character lowercase commit SHA"
        )
    try:
        HFAttentionImplementation(payload.get("attention_implementation"))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "sft_metadata.attention_implementation must be 'sdpa' or "
            "'flash_attention_4'"
        ) from exc
    if precision.get("export") != "bfloat16":
        raise ValueError("sft_metadata.precision.export must be 'bfloat16'")
    return payload
