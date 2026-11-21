"""Unit tests for the document iterator in
``scripts.preprocess.tok_train``.

The original ``tok_train.py`` defines its iterator at module level and
runs the full training pipeline as soon as the module is imported. We
extract the iterator into a pure helper ``build_text_iterator(cfg,
batches_iter)`` so tests can drive it deterministically without loading
real datasets.
"""

from __future__ import annotations

from omegaconf import DictConfig

from scripts.preprocess.tok_train import build_text_iterator


def test_iterator_yields_full_docs_when_below_doc_cap():
    cfg = DictConfig({"doc_cap": 100, "max_chars": 1_000})
    batches = iter([["aaaa", "bbb"], ["cc"]])
    out = list(build_text_iterator(cfg, batches))
    assert out == ["aaaa", "bbb", "cc"]


def test_iterator_truncates_long_docs_to_doc_cap():
    cfg = DictConfig({"doc_cap": 4, "max_chars": 1_000})
    batches = iter([["a" * 10, "bb"]])
    out = list(build_text_iterator(cfg, batches))
    # First doc truncated to 4 chars, second short doc unchanged.
    assert out == ["aaaa", "bb"]


def test_iterator_breaks_on_max_chars_within_batch():
    cfg = DictConfig({"doc_cap": 100, "max_chars": 5})
    batches = iter([["abc", "defg", "should_not_appear"]])
    out = list(build_text_iterator(cfg, batches))
    # Cumulative chars: 3 (abc) -> below; 7 (abc+defg) -> over 5; break.
    assert out == ["abc", "defg"]


def test_iterator_breaks_on_max_chars_across_batches():
    cfg = DictConfig({"doc_cap": 100, "max_chars": 6})
    batches = iter([["abc"], ["defgh", "skip_me"]])
    out = list(build_text_iterator(cfg, batches))
    assert out == ["abc", "defgh"]


def test_iterator_handles_empty_batches():
    cfg = DictConfig({"doc_cap": 10, "max_chars": 1000})
    batches = iter([[], ["a"], []])
    out = list(build_text_iterator(cfg, batches))
    assert out == ["a"]


def test_iterator_doc_cap_does_not_limit_total_count():
    """`doc_cap` truncates per-doc; it does not stop the iterator."""
    cfg = DictConfig({"doc_cap": 1, "max_chars": 100})
    batches = iter([["abcdef", "ghijkl"]])
    out = list(build_text_iterator(cfg, batches))
    assert out == ["a", "g"]
