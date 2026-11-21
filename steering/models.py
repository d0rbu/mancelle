"""Shared model configuration for extraction and SFT pipelines.

Lightweight module — only depends on ``transformers`` (AutoTokenizer / AutoConfig)
and ``loguru``.  No torch/sae imports so that scripts can use it without pulling
in the full extraction stack.
"""

from __future__ import annotations

from dataclasses import dataclass

from loguru import logger
from transformers import AutoConfig, AutoTokenizer


@dataclass(frozen=True)
class ExtractionModelConfig:
    """Configuration for a TransformerLens extraction model.

    ``chat_template_source`` lets each model declare where to borrow its
    chat template from when the model's own tokenizer lacks one. If
    ``None``, :func:`ensure_chat_template` falls back to its legacy
    behaviour of appending ``-it`` to the model's ``name_or_path``. The
    legacy behaviour is correct for Gemma 2 and for Gemma 3 270m but is
    broken for Gemma 3 4B/12B/27B PT models (whose IT variants strip the
    ``-pt`` suffix rather than appending ``-it``), so those entries
    declare an explicit source.
    """

    hf_model_id: str
    chat_template_source: str | None = None


EXTRACTION_MODELS: dict[str, ExtractionModelConfig] = {
    "gemma-2-2b": ExtractionModelConfig(hf_model_id="google/gemma-2-2b"),
    "gemma-2-9b": ExtractionModelConfig(hf_model_id="google/gemma-2-9b"),
    "gemma-2-27b": ExtractionModelConfig(hf_model_id="google/gemma-2-27b"),
    # Gemma 3 — see docs/pipelines/gemma-scope-2.md.
    # Note: 270m PT has no -pt suffix in its HF id.
    "gemma-3-270m": ExtractionModelConfig(
        hf_model_id="google/gemma-3-270m",
        chat_template_source="google/gemma-3-270m-it",
    ),
    "gemma-3-270m-it": ExtractionModelConfig(
        hf_model_id="google/gemma-3-270m-it",
    ),
    "gemma-3-1b-pt": ExtractionModelConfig(
        hf_model_id="google/gemma-3-1b-pt",
        chat_template_source="google/gemma-3-1b-it",
    ),
    "gemma-3-1b-it": ExtractionModelConfig(
        hf_model_id="google/gemma-3-1b-it",
    ),
    "gemma-3-4b-pt": ExtractionModelConfig(
        hf_model_id="google/gemma-3-4b-pt",
        chat_template_source="google/gemma-3-4b-it",
    ),
    "gemma-3-4b-it": ExtractionModelConfig(
        hf_model_id="google/gemma-3-4b-it",
    ),
    "gemma-3-12b-pt": ExtractionModelConfig(
        hf_model_id="google/gemma-3-12b-pt",
        chat_template_source="google/gemma-3-12b-it",
    ),
    "gemma-3-12b-it": ExtractionModelConfig(
        hf_model_id="google/gemma-3-12b-it",
    ),
    "gemma-3-27b-pt": ExtractionModelConfig(
        hf_model_id="google/gemma-3-27b-pt",
        chat_template_source="google/gemma-3-27b-it",
    ),
    "gemma-3-27b-it": ExtractionModelConfig(
        hf_model_id="google/gemma-3-27b-it",
    ),
}


def ensure_chat_template(tokenizer) -> None:
    """Ensure the tokenizer has a chat_template set.

    Legacy contract: if no template is present, load from
    ``f"{name_or_path}-it"``. Correct for Gemma 2 / Gemma 3 270m; broken
    for Gemma 3 4B/12B/27B PT. New callers should prefer
    :func:`ensure_chat_template_from_source` and an explicit source.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        return
    model_name = getattr(tokenizer, "name_or_path", "")
    if not model_name:
        raise ValueError("Tokenizer has no chat_template and no name_or_path to resolve IT variant")
    ensure_chat_template_from_source(tokenizer, source=model_name + "-it")


def ensure_chat_template_from_source(tokenizer, *, source: str) -> None:
    """Copy ``source``'s chat template onto ``tokenizer``.

    Raises:
        ValueError: if ``source`` itself has no chat_template.
    """
    if getattr(tokenizer, "chat_template", None) is not None:
        return
    src_tok = AutoTokenizer.from_pretrained(source)
    source_template = getattr(src_tok, "chat_template", None)
    if not source_template:
        raise ValueError(f"Chat template source {source!r} also has no chat_template")
    tokenizer.chat_template = source_template
    logger.info(f"Loaded chat_template from {source}")


def max_position_embeddings(hf_config) -> int:
    """Return ``max_position_embeddings`` from an HF config, tolerating the
    gemma-3 multimodal layout.

    gemma-3-4b/12b/27b load as ``Gemma3ForConditionalGeneration`` whose
    ``Gemma3Config`` nests the text params under ``text_config`` (with no
    top-level alias); gemma-2 and gemma-3 1b/270m expose them flat. (The
    model loader — ``steering.gemma3_loading.load_gemma3_text_decoder`` —
    remaps the VLM text weights (``language_model.model.*``) into a
    ``Gemma3ForCausalLM``; a plain ``Gemma3ForCausalLM.from_pretrained``
    would silently random-init them — issue #549. The raw ``AutoConfig``
    keeps the text params under ``text_config``.)
    """
    text_config = getattr(hf_config, "text_config", None)
    src = text_config if text_config is not None else hf_config
    return int(src.max_position_embeddings)


def resolve_sae_max_context(model: str) -> int:
    """Extraction's truncation cap (max position embeddings) for ``model``, read
    weight-free from the HF config (gemma-3-aware via :func:`max_position_embeddings`).

    RAISES if the config cannot be read or parsed -- a supported model's config must
    be available. We deliberately do NOT silently fall back to "no cap": that would
    let a too-large context_window misalign extraction or validation, and it would
    push the fallback decision implicitly onto every caller. Extraction itself uses
    the loaded ``model.cfg.n_ctx``; this is the weight-free counterpart, and they
    match for the supported models.
    """
    hf_cfg = AutoConfig.from_pretrained(EXTRACTION_MODELS[model].hf_model_id)
    return max_position_embeddings(hf_cfg)


def resolve_tokenizer(model: str, *, load_chat_template: bool = True):
    """Load the HuggingFace tokenizer for an extraction model.

    Args:
        model: TransformerLens model name (e.g. ``gemma-2-2b``).
        load_chat_template: Borrow a missing chat template. Disable for raw-text
            consumers that do not format chat messages.

    Raises:
        ValueError: If model is not in ``EXTRACTION_MODELS``.
    """
    if model not in EXTRACTION_MODELS:
        raise ValueError(
            f"Model '{model}' not found in EXTRACTION_MODELS. "
            f"Please add it to indicate its HuggingFace ID and system message support."
        )
    cfg = EXTRACTION_MODELS[model]
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_id)
    if load_chat_template:
        if cfg.chat_template_source is not None:
            ensure_chat_template_from_source(tokenizer, source=cfg.chat_template_source)
        else:
            ensure_chat_template(tokenizer)
    return tokenizer
