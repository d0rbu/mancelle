"""Unit tests for steering/models.py — slice 01 Phase 1 audit."""

from types import SimpleNamespace
from unittest.mock import Mock, call

import pytest

from steering import models
from steering.models import EXTRACTION_MODELS, resolve_tokenizer

# ---------------------------------------------------------------------------
# Sanity: resolve_tokenizer for unknown model
# ---------------------------------------------------------------------------


class TestResolveTokenizerUnknownModel:
    def test_unknown_model_raises_clean_error(self):
        """An unknown model name must raise ValueError with a helpful message."""
        with pytest.raises(ValueError, match="not found in EXTRACTION_MODELS|EXTRACTION_MODELS"):
            resolve_tokenizer("totally-fake-model")

    def test_known_models_listed(self):
        """The registry must contain at least gemma-2-2b."""
        assert "gemma-2-2b" in EXTRACTION_MODELS


@pytest.mark.parametrize("model", ["gemma-2-2b", "gemma-3-4b-pt"])
@pytest.mark.parametrize("mode", ["default", "enabled", "disabled"])
def test_resolve_tokenizer_optional_chat_template(monkeypatch, model, mode):
    cfg = EXTRACTION_MODELS[model]
    tokenizer = SimpleNamespace(name_or_path=cfg.hf_model_id, chat_template=None)
    source = SimpleNamespace(chat_template="borrowed template")
    factory = Mock(side_effect=[tokenizer, source])
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", factory)
    kwargs = {} if mode == "default" else {"load_chat_template": mode == "enabled"}

    assert resolve_tokenizer(model, **kwargs) is tokenizer

    expected = [call(cfg.hf_model_id)]
    if mode != "disabled":
        expected.append(call(cfg.chat_template_source or cfg.hf_model_id + "-it"))
    assert factory.call_args_list == expected
    assert tokenizer.chat_template == (None if mode == "disabled" else source.chat_template)


@pytest.mark.parametrize("model", ["gemma-2-2b", "gemma-3-4b-pt"])
@pytest.mark.parametrize("template", [None, "", "existing template", {"default": "existing"}])
def test_raw_tokenizer_preserves_original_template(monkeypatch, model, template):
    tokenizer = SimpleNamespace(chat_template=template)
    factory = Mock(return_value=tokenizer)
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", factory)

    assert resolve_tokenizer(model, load_chat_template=False) is tokenizer

    factory.assert_called_once_with(EXTRACTION_MODELS[model].hf_model_id)
    assert tokenizer.chat_template is template


@pytest.mark.parametrize("load_chat_template", [True, False])
def test_resolve_tokenizer_errors_before_template_loading(monkeypatch, load_chat_template):
    error = OSError("base tokenizer unavailable")
    factory = Mock(side_effect=error)
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", factory)
    with pytest.raises(ValueError, match="not found in EXTRACTION_MODELS"):
        resolve_tokenizer("unknown-model", load_chat_template=load_chat_template)
    factory.assert_not_called()

    with pytest.raises(OSError) as caught:
        resolve_tokenizer("gemma-2-2b", load_chat_template=load_chat_template)
    assert caught.value is error
    factory.assert_called_once_with("google/gemma-2-2b")


@pytest.mark.parametrize("model", ["gemma-2-2b", "gemma-3-4b-pt"])
@pytest.mark.parametrize("mode", ["default", "enabled"])
def test_resolve_tokenizer_retains_missing_template_error(monkeypatch, model, mode):
    cfg = EXTRACTION_MODELS[model]
    tokenizer = SimpleNamespace(name_or_path=cfg.hf_model_id, chat_template=None)
    factory = Mock(side_effect=[tokenizer, SimpleNamespace(chat_template=None)])
    monkeypatch.setattr(models.AutoTokenizer, "from_pretrained", factory)
    kwargs = {} if mode == "default" else {"load_chat_template": True}

    with pytest.raises(ValueError, match="also has no chat_template"):
        resolve_tokenizer(model, **kwargs)

    assert factory.call_count == 2
