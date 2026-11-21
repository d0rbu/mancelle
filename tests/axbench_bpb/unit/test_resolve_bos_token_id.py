"""Unit tests for steering.segmentation.resolve_bos_token_id.

Commit 32cc48db promoted ``_resolve_bos_token_id`` to public
``resolve_bos_token_id`` because axbench-bpb's ``assert_bos_lengths_match``
needs it. The helper probes the two BOS-discovery contracts in use:

- HF-style ``.bos_token_id`` attribute (e.g. ``transformers.GemmaTokenizer``)
- Method-style ``.get_bos_token_id()`` (e.g. nanochat's RustBPE wrapper)

Returns ``int`` for either contract; ``None`` if both are unavailable
(downstream callers raise a domain error in that case).
"""
from __future__ import annotations

from steering.segmentation import resolve_bos_token_id
from tests.cross_tokenizer_helpers import SimpleTokenizer


class _HFStyle:
    """Mimics ``transformers.GemmaTokenizer``: ``bos_token_id`` is an
    attribute exposed by the base ``PreTrainedTokenizerBase`` class."""

    def __init__(self, bos: int | None) -> None:
        self.bos_token_id = bos


class _NanochatStyle:
    """Mimics nanochat RustBPE / TransformersTokenizer: ``get_bos_token_id()``
    is a bound method."""

    def __init__(self, bos: int | None) -> None:
        self._bos = bos

    def get_bos_token_id(self) -> int | None:
        return self._bos


class _NoBosContract:
    """Neither attribute nor method."""

    def encode(self, text: str) -> list[int]:
        return [0]


class _MethodRaises:
    """Method exists but raises — defensive coverage."""

    def get_bos_token_id(self) -> int:
        raise RuntimeError("simulated failure")


# ---------- HF attribute contract ----------


def test_returns_int_from_hf_attribute():
    tok = _HFStyle(bos=2)
    assert resolve_bos_token_id(tok) == 2


def test_returns_int_from_hf_attribute_zero_is_valid():
    """BOS id of 0 (SimpleTokenizer's choice) is a valid int — NOT
    falsy-skipped. The helper distinguishes ``None`` from ``0`` so a
    tokenizer that exposes ``bos_token_id = 0`` doesn't accidentally
    fall through to the method probe."""
    tok = _HFStyle(bos=0)
    assert resolve_bos_token_id(tok) == 0


def test_skips_hf_attribute_when_none():
    """If ``.bos_token_id`` is None, fall through to the method probe.
    Some HF tokenizers expose a None ``bos_token_id`` when their
    special-tokens map is incomplete; the helper still tries the method."""
    class _BothContracts:
        bos_token_id = None
        def get_bos_token_id(self) -> int:
            return 7
    assert resolve_bos_token_id(_BothContracts()) == 7


# ---------- Method contract ----------


def test_returns_int_from_method():
    tok = _NanochatStyle(bos=42)
    assert resolve_bos_token_id(tok) == 42


def test_returns_int_from_method_when_attribute_absent():
    """A tokenizer with no ``bos_token_id`` attribute at all (only the
    method) still resolves correctly."""
    tok = _NanochatStyle(bos=99)
    # Sanity: no attribute.
    assert not hasattr(tok, "bos_token_id")
    assert resolve_bos_token_id(tok) == 99


def test_method_returning_none_yields_none():
    tok = _NanochatStyle(bos=None)
    assert resolve_bos_token_id(tok) is None


def test_method_raising_yields_none():
    """A tokenizer whose method raises must NOT propagate — the helper
    returns None so callers see a uniform contract."""
    tok = _MethodRaises()
    assert resolve_bos_token_id(tok) is None


# ---------- Neither contract ----------


def test_no_contract_returns_none():
    tok = _NoBosContract()
    assert resolve_bos_token_id(tok) is None


def test_random_object_returns_none():
    """An object with no tokenizer-shape at all returns None rather
    than crashing — the helper is defensive at the boundary."""
    assert resolve_bos_token_id(object()) is None


# ---------- Symmetric defensive coverage ----------


def test_attribute_property_raising_falls_through_to_method():
    """An HF tokenizer with a property-style ``bos_token_id`` that
    raises (e.g. lazy I/O failure on the special-tokens map) must not
    propagate. The helper falls through to the method probe.

    This pins the symmetric exception-handling fix: pre-fix the
    attribute path didn't catch property exceptions, while the method
    path did. A tokenizer with a raising property AND a working method
    would crash on the former despite being recoverable.
    """
    class _RaisingProp:
        @property
        def bos_token_id(self):
            raise RuntimeError("simulated lazy-load failure")

        def get_bos_token_id(self):
            return 7

    assert resolve_bos_token_id(_RaisingProp()) == 7


def test_attribute_property_raising_with_no_method_yields_none():
    """A tokenizer whose only contract is a raising property — and
    no method — returns None instead of propagating."""
    class _RaisingPropOnly:
        @property
        def bos_token_id(self):
            raise RuntimeError("simulated failure")

    assert resolve_bos_token_id(_RaisingPropOnly()) is None


def test_attribute_with_uncoercible_value_falls_through_to_method():
    """If the attribute is an object that ``int()`` can't coerce
    (e.g. a list), the helper falls through rather than crashing.
    Pins symmetric int()-defensive coverage with the method path."""
    class _UncoercibleAttr:
        bos_token_id = ["not", "an", "int"]

        def get_bos_token_id(self):
            return 5

    # Falls through to method since int(list) raises TypeError.
    assert resolve_bos_token_id(_UncoercibleAttr()) == 5


def test_method_with_uncoercible_value_yields_none():
    """Symmetric int()-coverage for the method path."""
    class _UncoercibleMethod:
        def get_bos_token_id(self):
            return ["not", "an", "int"]

    assert resolve_bos_token_id(_UncoercibleMethod()) is None


# ---------- Sanity: production tokenizers ----------


def test_simple_tokenizer_from_test_fixtures_resolves():
    """The SimpleTokenizer used throughout the test suite exposes
    ``get_bos_token_id()`` returning 0; the helper resolves to 0."""
    tok = SimpleTokenizer()
    assert resolve_bos_token_id(tok) == 0
