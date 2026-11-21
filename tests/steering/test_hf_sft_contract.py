"""Architecture-neutral repository HF SFT metadata contract tests."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from steering.sft.hf_contract import (
    HF_SFT_FORMAT_VERSION,
    validate_repository_hf_sft_metadata,
)


def _metadata() -> dict[str, Any]:
    return {
        "format": "huggingface_standard_sft",
        "format_version": HF_SFT_FORMAT_VERSION,
        "resumable": False,
        "completed_optimizer_step": 3,
        "backend": "huggingface",
        "family": "pythia",
        "source": {
            "model_id": "EleutherAI/pythia-70m",
            "revision": "a" * 40,
            "architecture": "GPTNeoXForCausalLM",
            "model_type": "gpt_neox",
        },
        "output": {
            "architecture": "GPTNeoXForCausalLM",
            "model_type": "gpt_neox",
        },
        "attention_implementation": "sdpa",
        "precision": {"export": "bfloat16"},
        "batch": {"max_seq_len": 32},
        "chat_template_sha256": "b" * 64,
        "vocabulary": {"model": 50304, "tokenizer": 50277},
        "backend_specific": {"identity_bearing": True},
    }


def _replace(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target: dict[str, Any] = payload
    for key in path[:-1]:
        child = target[key]
        assert isinstance(child, dict)
        target = child
    target[path[-1]] = value


def test_shared_hf_metadata_accepts_backend_specific_fields() -> None:
    metadata = _metadata()
    assert validate_repository_hf_sft_metadata(metadata) == metadata


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("format",), ""),
        (("format_version",), 1),
        (("format_version",), True),
        (("resumable",), True),
        (("completed_optimizer_step",), -1),
        (("completed_optimizer_step",), True),
        (("backend",), ""),
        (("family",), ""),
        (("source", "model_id"), ""),
        (("source", "revision"), "main"),
        (("source", "revision"), "A" * 40),
        (("source", "architecture"), ""),
        (("source", "model_type"), ""),
        (("output", "architecture"), ""),
        (("output", "model_type"), ""),
        (("attention_implementation",), "eager"),
        (("precision", "export"), "float32"),
        (("batch", "max_seq_len"), 0),
        (("chat_template_sha256",), "not-a-digest"),
        (("vocabulary", "model"), 0),
        (("vocabulary", "tokenizer"), True),
    ],
)
def test_shared_hf_metadata_rejects_contract_drift(
    path: tuple[str, ...],
    value: Any,
) -> None:
    metadata = copy.deepcopy(_metadata())
    _replace(metadata, path, value)
    with pytest.raises(ValueError):
        validate_repository_hf_sft_metadata(metadata)


@pytest.mark.parametrize(
    "missing",
    [
        "format",
        "format_version",
        "resumable",
        "completed_optimizer_step",
        "backend",
        "family",
        "source",
        "output",
        "attention_implementation",
        "precision",
        "batch",
        "chat_template_sha256",
        "vocabulary",
    ],
)
def test_shared_hf_metadata_rejects_missing_required_fields(missing: str) -> None:
    metadata = _metadata()
    del metadata[missing]
    with pytest.raises(ValueError):
        validate_repository_hf_sft_metadata(metadata)
