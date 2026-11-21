"""Contract tests for full-document feature extraction."""

import inspect

from steering.extract_features import extract_document_features


def test_document_relative_char_ranges_are_documented():
    doc = extract_document_features.__doc__ or ""
    assert "document-relative" in doc
    assert "assistant_char_offset" not in doc


def test_signature_has_no_assistant_char_offset():
    sig = inspect.signature(extract_document_features)
    assert "assistant_char_offset" not in sig.parameters
