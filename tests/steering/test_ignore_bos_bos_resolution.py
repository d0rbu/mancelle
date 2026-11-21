"""Pin: resolve_bos_token_id == tokenizer.bos_token_id for an HF-like
tokenizer so unifying BOS resolution cannot perturb context-independent
output."""

import pytest

from steering.segmentation import resolve_bos_token_id


def test_resolve_matches_attribute_for_hf_like():
    class HF:
        bos_token_id = 2

    assert resolve_bos_token_id(HF()) == 2


def test_resolve_none_when_undiscoverable():
    class NoBos:
        pass

    assert resolve_bos_token_id(NoBos()) is None


def test_resolve_via_get_bos_token_id_method():
    """Object with no bos_token_id attribute but get_bos_token_id() method →
    result equals what the method returns."""
    class MethodOnly:
        def get_bos_token_id(self) -> int:
            return 42

    assert resolve_bos_token_id(MethodOnly()) == 42


def test_resolve_get_bos_token_id_returning_none():
    """Object whose get_bos_token_id() returns None → result is None."""
    class MethodReturnsNone:
        def get_bos_token_id(self):
            return None

    assert resolve_bos_token_id(MethodReturnsNone()) is None


@pytest.mark.slow
def test_resolve_matches_real_gemma_tokenizer():
    pytest.importorskip("transformers")
    from steering.models import resolve_tokenizer

    tok = resolve_tokenizer("gemma-2-2b")
    assert resolve_bos_token_id(tok) == tok.bos_token_id
