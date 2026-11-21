"""Unit tests for the opt-in `_check_collision()` helper added to
`tests.conftest.MockTokenizer` (consolidation manifest §1.17).

The helper exists to surface a long-standing footgun: ``MockTokenizer``
hardcodes the chat-special IDs (``<user_start>=64`` … ``<assistant_end>=67``)
which collide with regular ``encode()`` output for ASCII characters whose
ord modulo ``vocab_size-2`` plus 2 lands in [64, 67]. With ``vocab_size=128``,
these are ``> ? @ A`` (ord 62/63/64/65 → ids 64/65/66/67). Existing tests
survive only because their prompts are crafted to avoid those characters.

The helper raises ``ValueError`` if any input text would alias to a
special-token id. Callers opt in via ``MockTokenizer.with_collision_check()``
to keep the original permissive behaviour for the ~580 existing call sites.
"""

from __future__ import annotations

import pytest

from tests.conftest import MockTokenizer


def test_default_no_collision_check_silent_on_colliding_input():
    """Default ``MockTokenizer`` permits colliding inputs silently."""
    tok = MockTokenizer()
    # encode("A") == 67 == <assistant_end>; without collision check, no raise.
    ids = tok.encode("A")
    assert ids == [67]


def test_collision_check_opt_in_raises_on_colliding_character_A():
    tok = MockTokenizer.with_collision_check()
    # encode("A") would yield id 67 which equals <assistant_end>.
    with pytest.raises(ValueError, match=r"colli"):
        tok.encode("A")


def test_collision_check_opt_in_raises_on_question_mark():
    tok = MockTokenizer.with_collision_check()
    # '?' has ord 63; encode yields 65 which is <user_end>.
    with pytest.raises(ValueError, match=r"colli"):
        tok.encode("?")


def test_collision_check_opt_in_raises_on_at_sign():
    tok = MockTokenizer.with_collision_check()
    # '@' has ord 64; encode yields 66 which is <assistant_start>.
    with pytest.raises(ValueError, match=r"colli"):
        tok.encode("@")


def test_collision_check_opt_in_raises_on_gt():
    tok = MockTokenizer.with_collision_check()
    # '>' has ord 62; encode yields 64 which is <user_start>.
    with pytest.raises(ValueError, match=r"colli"):
        tok.encode(">")


def test_collision_check_opt_in_silent_on_safe_character():
    tok = MockTokenizer.with_collision_check()
    # 'a' has ord 97; (97 % 126) + 2 = 99; not in {64,65,66,67}.
    ids = tok.encode("a")
    assert ids == [99]


def test_collision_check_opt_in_raises_inside_long_string():
    tok = MockTokenizer.with_collision_check()
    # a long string with one colliding char buried inside.
    with pytest.raises(ValueError, match=r"colli"):
        tok.encode("hello world A from the test")


def test_collision_check_with_safe_string():
    tok = MockTokenizer.with_collision_check()
    safe = "hello world from the test"
    ids = tok.encode(safe)
    assert len(ids) == len(safe)


def test_collision_check_with_list_dispatches_per_element():
    tok = MockTokenizer.with_collision_check()
    with pytest.raises(ValueError, match=r"colli"):
        tok.encode(["safe text", "an A in here"])


def test_collision_check_returns_self_via_constructor_factory():
    """`with_collision_check()` is the canonical way to opt in."""
    tok = MockTokenizer.with_collision_check()
    # we can still encode special tokens — the check applies to plain text.
    assert tok.encode_special("<user_start>") == 64
    assert tok.encode_special("<assistant_end>") == 67


def test_render_conversation_with_collision_check_rejects_colliding_content():
    """`render_conversation` must surface collisions from message content."""
    tok = MockTokenizer.with_collision_check()
    conversation = {"messages": [{"role": "user", "content": "Pick A please"}]}
    with pytest.raises(ValueError, match=r"colli"):
        tok.render_conversation(conversation)


def test_render_conversation_with_collision_check_passes_safe_content():
    tok = MockTokenizer.with_collision_check()
    conversation = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi back"},
        ]
    }
    ids, mask = tok.render_conversation(conversation)
    assert len(ids) == len(mask)
    assert all(m in (0, 1) for m in mask)
