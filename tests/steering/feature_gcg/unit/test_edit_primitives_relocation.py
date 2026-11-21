from __future__ import annotations

import pytest

from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    replace_single_token,
    token_sequence_digest,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError


def test_span_geometry():
    span = EditableSpan(start=3, count=2)
    assert span.stop == 5
    assert list(span.positions()) == [3, 4]


def test_digest_is_deterministic_and_order_sensitive():
    assert token_sequence_digest([1, 2, 3]) == token_sequence_digest([1, 2, 3])
    assert token_sequence_digest([1, 2, 3]) != token_sequence_digest([3, 2, 1])


def test_replace_single_token_guards_bos():
    assert replace_single_token([9, 8, 7], position=1, token_id=5) == [9, 5, 7]
    with pytest.raises(FeatureGcgError):
        replace_single_token([9, 8, 7], position=0, token_id=5)
