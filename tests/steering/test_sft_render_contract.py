"""The one conversation-render contract shared by the SFT converter and trainer.

``convert_sft_to_parquet`` renders each conversation only up to its last
assistant turn, and the parquet ``text`` plus every extracted segment char
offset is built from that slice.  ``conditional_encdec_sft`` must render the
identical slice or its decoder text runs past the feature coverage (issue
#1465).  These tests pin both halves of that contract against each other.
"""

from pathlib import Path

import pytest

from errors import DataLoadError
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.preprocess.convert_sft_to_parquet import (
    prepare_messages_for_full_nanochat_render,
    render_conversation_to_full_nanochat_text,
)
from scripts.train.conditional_encdec_sft import _prepare_sft_render_messages
from steering.sft.loss_mask import render_conversation_with_mask, supervised_target_count
from tests.steering.test_encdec_sft_conditioned_integration import (
    _full_sequence_char_count,
)
from tests.steering.test_sft_loss_mask import _SpanTokenizer


def _real_tokenizer_available() -> bool:
    return (Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl").exists()


class _ContractTokenizer(_SpanTokenizer):
    """``_SpanTokenizer`` plus the ``decode`` the converter's text helper needs.

    ``_SpanTokenizer.render_conversation`` delegates to the real
    ``RustBPETokenizer.render_conversation``, so system-message merging,
    multipart ``python``/``python_output`` parts and the strict role-alternation
    asserts are all production behavior.  It has no ``decode``, which
    ``render_messages_to_full_nanochat_text`` requires; this subclass adds the
    exact inverse of its ``encode``/``SPECIAL`` tables, so special tokens
    round-trip to their literal text exactly as the production tokenizer's does.
    """

    def decode(self, ids: list[int]) -> str:
        literals = {value: key for key, value in self.SPECIAL.items()}
        literals[self.BOS] = "<bos>"
        return "".join(
            literals[token_id] if token_id in literals else chr(token_id - 100) for token_id in ids
        )


_TRAILING_ASSISTANT = {
    "messages": [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
    ]
}
_TRAILING_USER = {
    "messages": [
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
        {"role": "user", "content": "Thanks for the info."},
    ]
}
_MULTI_TURN_TRAILING_USER = {
    "messages": [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "two"},
        {"role": "user", "content": "three"},
        {"role": "assistant", "content": "four"},
        {"role": "user", "content": "five"},
    ]
}
_SYSTEM_PREFIXED_TRAILING_USER = {
    "messages": [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "hello there"},
        {"role": "assistant", "content": "hi back"},
        {"role": "user", "content": "and one more thing"},
    ]
}
_TOOL_CALL_TRAILING_USER = {
    "messages": [
        {"role": "user", "content": "compute"},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "sure"},
                {"type": "python", "text": "1+1"},
                {"type": "python_output", "text": "2"},
                {"type": "text", "text": "it is two"},
            ],
        },
        {"role": "user", "content": "thanks!"},
    ]
}

_TRIMMING_CASES = [
    pytest.param(_TRAILING_USER, id="trailing_user"),
    pytest.param(_MULTI_TURN_TRAILING_USER, id="multi_turn_trailing_user"),
    pytest.param(_SYSTEM_PREFIXED_TRAILING_USER, id="system_prefixed_trailing_user"),
    pytest.param(_TOOL_CALL_TRAILING_USER, id="tool_call_trailing_user"),
]
_ALL_CASES = [pytest.param(_TRAILING_ASSISTANT, id="trailing_assistant"), *_TRIMMING_CASES]
_FIXTURE_PARITY_CASES = [
    pytest.param(_TRAILING_ASSISTANT, id="trailing_assistant"),
    pytest.param(_TRAILING_USER, id="trailing_user"),
]


@pytest.fixture
def tokenizer() -> _ContractTokenizer:
    return _ContractTokenizer()


def _render(tokenizer, conversation) -> tuple[list[int], list[int]]:
    messages = _prepare_sft_render_messages(conversation, surface="test", original_index=0)
    return render_conversation_with_mask(tokenizer, {"messages": messages})


class TestSharedRenderContract:
    @pytest.mark.parametrize("conversation", _ALL_CASES)
    def test_trimmed_render_is_an_exact_prefix_of_the_untrimmed_render(
        self, tokenizer, conversation
    ) -> None:
        """Trimming drops trailing messages only, so ids/mask stay prefixes."""
        trimmed_ids, trimmed_mask = _render(tokenizer, conversation)
        untrimmed_ids, untrimmed_mask = render_conversation_with_mask(tokenizer, conversation)
        assert trimmed_ids == untrimmed_ids[: len(trimmed_ids)]
        assert trimmed_mask == untrimmed_mask[: len(trimmed_mask)]

    def test_trailing_assistant_render_is_bit_identical(self, tokenizer) -> None:
        """The 99.99% path must be provably untouched by the trim."""
        trimmed_ids, trimmed_mask = _render(tokenizer, _TRAILING_ASSISTANT)
        untrimmed_ids, untrimmed_mask = render_conversation_with_mask(
            tokenizer, _TRAILING_ASSISTANT
        )
        assert trimmed_ids == untrimmed_ids
        assert trimmed_mask == untrimmed_mask

    @pytest.mark.parametrize("conversation", _TRIMMING_CASES)
    def test_trimming_actually_removes_tokens(self, tokenizer, conversation) -> None:
        """Non-vacuity: every trimming case must really lose a trailing turn."""
        trimmed_ids, _ = _render(tokenizer, conversation)
        untrimmed_ids, _ = render_conversation_with_mask(tokenizer, conversation)
        assert len(trimmed_ids) < len(untrimmed_ids)

    @pytest.mark.parametrize("conversation", _ALL_CASES)
    def test_completion_only_supervision_is_unchanged_by_trimming(
        self, tokenizer, conversation
    ) -> None:
        """Trailing user tokens carry mask 0, so completion-only counts hold."""
        trimmed_ids, trimmed_mask = _render(tokenizer, conversation)
        untrimmed_ids, untrimmed_mask = render_conversation_with_mask(tokenizer, conversation)
        assert supervised_target_count(trimmed_mask) == supervised_target_count(untrimmed_mask)


class TestSkippedRowsFailLoud:
    def test_no_assistant_row_raises(self) -> None:
        conversation = {"messages": [{"role": "user", "content": "hi"}]}
        assert prepare_messages_for_full_nanochat_render(conversation) == (
            None,
            "no_assistant",
        )
        with pytest.raises(DataLoadError, match=r"data generator.*original_index=17.*no_assistant"):
            _prepare_sft_render_messages(conversation, surface="data generator", original_index=17)

    def test_empty_final_assistant_row_raises(self) -> None:
        conversation = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "   "},
            ]
        }
        assert prepare_messages_for_full_nanochat_render(conversation) == (
            None,
            "empty_response",
        )
        with pytest.raises(DataLoadError, match=r"empty_response"):
            _prepare_sft_render_messages(conversation, surface="data generator", original_index=17)

    def test_empty_multipart_final_assistant_row_raises(self) -> None:
        """``_is_empty_multipart_response`` is separate logic and needs its own case."""
        conversation = {
            "messages": [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "  "}],
                },
            ]
        }
        with pytest.raises(DataLoadError, match=r"startup content probe.*empty_response"):
            _prepare_sft_render_messages(
                conversation, surface="startup content probe", original_index=3
            )


class TestFixtureSanity:
    """Cheap fixture checks, plus the ONE assertion that pins production.

    Only ``test_tokenizer_uses_the_real_nanochat_renderer`` and
    ``test_real_tokenizer_marker_widths_exceed_the_guard_tolerance`` say
    anything about production. The decode round-trip below is true by
    construction (``_ContractTokenizer.decode`` is defined to emit those
    literals) and is here to catch a broken fixture, not to falsify anything.
    """

    def test_tokenizer_uses_the_real_nanochat_renderer(self, tokenizer) -> None:
        ids, mask = tokenizer.render_conversation(_TRAILING_ASSISTANT)
        assert ids[0] == tokenizer.get_bos_token_id()
        assert ids.count(tokenizer.encode_special("<assistant_end>")) == 1
        # Only the real renderer marks <assistant_end> as supervised.
        assert mask[ids.index(tokenizer.encode_special("<assistant_end>"))] == 1

    def test_decode_is_the_exact_inverse_of_encode(self, tokenizer) -> None:
        ids, _ = tokenizer.render_conversation(_TRAILING_USER)
        assert tokenizer.decode(ids).startswith("<bos><user_start>hello there<user_end>")
        assert tokenizer.decode(ids).endswith("<user_start>Thanks for the info.<user_end>")

    def test_trailing_user_turn_costs_at_least_the_guard_tolerance(self, tokenizer) -> None:
        """The dropped span is never shorter than the two markers."""
        marker_chars = len("<user_start>") + len("<user_end>")
        trimmed, _ = _render(tokenizer, _TRAILING_USER)
        untrimmed, _ = render_conversation_with_mask(tokenizer, _TRAILING_USER)
        added = tokenizer.decode(untrimmed[len(trimmed) :])
        assert len(added) >= marker_chars


@pytest.mark.skipif(not _real_tokenizer_available(), reason="no tokenizer.pkl")
def test_real_tokenizer_marker_widths_exceed_the_guard_tolerance() -> None:
    """The load-bearing production fact: a trailing user turn cannot hide.

    ``<user_start>`` and ``<user_end>`` decode to 12 and 10 characters under the
    REAL tokenizer, so the minimum possible trailing-user coverage gap is 22 —
    already past the guard's 20-char tolerance. That is why issue #1465 always
    crashed and never silently mis-conditioned. Asserting this on the fixture's
    own string literals would prove nothing; this asserts it on production.
    """
    real = get_tokenizer()
    widths = {
        name: len(real.decode([real.encode_special(name)]))
        for name in ("<user_start>", "<user_end>")
    }
    assert widths == {"<user_start>": 12, "<user_end>": 10}
    assert sum(widths.values()) > 20


class TestSyntheticFixtureMatchesTheConverter:
    """The test fixture must model extraction, which segments the parquet text."""

    @pytest.mark.parametrize("conversation", _FIXTURE_PARITY_CASES)
    def test_char_count_matches_converter_text_length(self, tokenizer, conversation) -> None:
        expected, skip_reason = render_conversation_to_full_nanochat_text(tokenizer, conversation)
        assert skip_reason is None
        assert expected is not None, f"conversation unexpectedly skipped: {skip_reason}"
        assert _full_sequence_char_count(tokenizer, conversation) == len(expected)
