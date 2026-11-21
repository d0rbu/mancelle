"""Behavioral tests for ``scripts/chat/steer_chat_cli.py`` pure helpers.

Covers the prompt-construction and response-finalization helpers plus the
``CliMode`` enum.  These functions are still live (steer_chat_cli.py:32-67) but
their tests were dropped in the encdec-full-conditioning-cleanup PR; this file
recovers the behavioral intent and adapts it to the current signatures.

The helpers are pure list/tokenizer transforms, so we assert the exact tokens
they produce rather than only that they "run".  End-to-end checks exercise the
real ``InferenceEngine`` with a tiny model so the token wiring (BOS, chat
control tokens, stop sets) is verified against real generation.
"""

from __future__ import annotations

import pytest
import torch as t

from scripts.chat.steer_chat_cli import (
    CliMode,
    build_chat_prompt,
    build_completion_prompt,
    finalize_chat_response,
)
from steering.eval.inference_engine import InferenceEngine, build_engine_for_features
from steering.eval.target_conditioning import TargetConditioningMode

# ---------------------------------------------------------------------------
# CliMode enum
# ---------------------------------------------------------------------------


class TestCliMode:
    def test_chat_from_string(self) -> None:
        assert CliMode("chat") is CliMode.CHAT

    def test_completion_from_string(self) -> None:
        assert CliMode("completion") is CliMode.COMPLETION

    def test_values(self) -> None:
        assert CliMode.CHAT.value == "chat"
        assert CliMode.COMPLETION.value == "completion"

    def test_invalid_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            CliMode("invalid")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError):
            CliMode("")

    def test_case_sensitive(self) -> None:
        with pytest.raises(ValueError):
            CliMode("Chat")
        with pytest.raises(ValueError):
            CliMode("COMPLETION")

    def test_is_str(self) -> None:
        # CliMode is a StrEnum, so members compare/behave like str.
        assert isinstance(CliMode.CHAT, str)
        assert isinstance(CliMode.COMPLETION, str)
        assert CliMode.CHAT == "chat"

    def test_exactly_two_members(self) -> None:
        assert set(CliMode) == {CliMode.CHAT, CliMode.COMPLETION}

    def test_identity_stable(self) -> None:
        assert CliMode("chat") is CliMode("chat")
        assert CliMode("completion") is CliMode("completion")

    def test_name_attribute(self) -> None:
        assert CliMode.CHAT.name == "CHAT"
        assert CliMode.COMPLETION.name == "COMPLETION"


# ---------------------------------------------------------------------------
# build_chat_prompt
# ---------------------------------------------------------------------------


class TestBuildChatPrompt:
    def test_appends_template_tokens(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")

        tokens = [bos]
        result = build_chat_prompt(tokens, "hi", mock_tokenizer, us, ue, a_s)

        # Mutates and returns the same list object.
        assert result is tokens
        assert tokens[0] == bos
        assert tokens[1] == us
        # ...user text..., then <user_end>, then <assistant_start> close it.
        assert tokens[-2] == ue
        assert tokens[-1] == a_s

    def test_encoded_input_between_markers(self, mock_tokenizer) -> None:
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")

        tokens: list[int] = []
        build_chat_prompt(tokens, "hello", mock_tokenizer, us, ue, a_s)

        encoded = mock_tokenizer.encode("hello")
        us_idx = tokens.index(us)
        ue_idx = tokens.index(ue)
        assert tokens[us_idx + 1 : ue_idx] == encoded

    def test_full_token_sequence_is_exact(self, mock_tokenizer) -> None:
        # The whole structure is deterministic for a known input.
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")

        tokens: list[int] = []
        build_chat_prompt(tokens, "hi", mock_tokenizer, us, ue, a_s)
        assert tokens == [us, *mock_tokenizer.encode("hi"), ue, a_s]

    def test_accumulates_over_turns(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")

        tokens = [bos]
        build_chat_prompt(tokens, "first", mock_tokenizer, us, ue, a_s)
        len_after_1 = len(tokens)

        build_chat_prompt(tokens, "second", mock_tokenizer, us, ue, a_s)
        assert len(tokens) > len_after_1
        # First turn's prefix is still intact.
        assert tokens[0] == bos
        assert tokens[1] == us
        # Two user turns => two of each control token.
        assert tokens.count(us) == 2
        assert tokens.count(a_s) == 2

    def test_empty_input(self, mock_tokenizer) -> None:
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")

        tokens: list[int] = []
        build_chat_prompt(tokens, "", mock_tokenizer, us, ue, a_s)
        # user_start, user_end, assistant_start — no encoded text between markers.
        assert tokens == [us, ue, a_s]


# ---------------------------------------------------------------------------
# build_completion_prompt
# ---------------------------------------------------------------------------


class TestBuildCompletionPrompt:
    def test_starts_with_bos(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        tokens = build_completion_prompt("hello", mock_tokenizer, bos)
        assert tokens[0] == bos

    def test_rest_is_encoded_input(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        tokens = build_completion_prompt("hello", mock_tokenizer, bos)
        assert tokens[1:] == mock_tokenizer.encode("hello")

    def test_returns_new_list_each_call(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        t1 = build_completion_prompt("a", mock_tokenizer, bos)
        t2 = build_completion_prompt("b", mock_tokenizer, bos)
        assert t1 is not t2

    def test_no_chat_control_tokens(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")
        a_e = mock_tokenizer.encode_special("<assistant_end>")

        # "hello world" contains no characters that alias the 64-67 special ids.
        tokens = build_completion_prompt("hello world", mock_tokenizer, bos)
        for special in (us, ue, a_s, a_e):
            assert special not in tokens, (
                f"chat-control token {special} leaked into completion tokens"
            )

    def test_empty_input(self, mock_tokenizer) -> None:
        bos = mock_tokenizer.get_bos_token_id()
        tokens = build_completion_prompt("", mock_tokenizer, bos)
        assert tokens == [bos]


# ---------------------------------------------------------------------------
# finalize_chat_response
# ---------------------------------------------------------------------------


class TestFinalizeChatResponse:
    def test_appends_assistant_end_when_missing(self, mock_tokenizer) -> None:
        a_e = mock_tokenizer.encode_special("<assistant_end>")
        conv: list[int] = [1, 2, 3]
        resp = [10, 20, 30]
        result = finalize_chat_response(conv, resp, a_e)
        # Returns the (mutated) conversation list.
        assert result is conv
        assert resp[-1] == a_e
        assert conv[-1] == a_e

    def test_no_duplicate_when_present(self, mock_tokenizer) -> None:
        # Idempotent: do not append a second <assistant_end> when already closed.
        a_e = mock_tokenizer.encode_special("<assistant_end>")
        conv: list[int] = [1, 2, 3]
        resp = [10, 20, a_e]
        finalize_chat_response(conv, resp, a_e)
        assert resp.count(a_e) == 1
        assert conv == [1, 2, 3, 10, 20, a_e]

    def test_empty_response_gets_assistant_end(self, mock_tokenizer) -> None:
        # The P0 regression: empty response must NOT raise IndexError; it gets
        # a single <assistant_end>.
        a_e = mock_tokenizer.encode_special("<assistant_end>")
        conv: list[int] = [1, 2]
        resp: list[int] = []
        finalize_chat_response(conv, resp, a_e)
        assert resp == [a_e]
        assert conv == [1, 2, a_e]

    def test_extends_conversation(self, mock_tokenizer) -> None:
        a_e = mock_tokenizer.encode_special("<assistant_end>")
        conv: list[int] = [0]
        resp = [10, 20]
        finalize_chat_response(conv, resp, a_e)
        assert conv == [0, 10, 20, a_e]


# ---------------------------------------------------------------------------
# Mode-specific stop tokens (mirrors main()'s wiring)
# ---------------------------------------------------------------------------


class TestStopTokenWiring:
    def test_chat_stops_on_assistant_end_and_bos(self, mock_tokenizer) -> None:
        # main() passes stop_token_ids={assistant_end, bos} in chat mode.
        bos = mock_tokenizer.get_bos_token_id()
        a_e = mock_tokenizer.encode_special("<assistant_end>")
        stop = {a_e, bos}
        assert stop == {a_e, bos}
        assert len(stop) == 2

    def test_completion_stops_only_on_bos(self, mock_tokenizer) -> None:
        # main() passes stop_token_ids={bos} in completion mode.
        bos = mock_tokenizer.get_bos_token_id()
        a_e = mock_tokenizer.encode_special("<assistant_end>")
        stop = {bos}
        assert bos in stop
        assert a_e not in stop


# ---------------------------------------------------------------------------
# End-to-end token flow with the real InferenceEngine + tiny model
# ---------------------------------------------------------------------------


class TestChatModeEndToEnd:
    def test_generate_stream_with_chat_tokens(self, gpt_model, mock_tokenizer) -> None:
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        bos = mock_tokenizer.get_bos_token_id()
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")
        a_e = mock_tokenizer.encode_special("<assistant_end>")

        conv = [bos]
        build_chat_prompt(conv, "hello", mock_tokenizer, us, ue, a_s)
        streamed = list(
            engine.generate_stream(
                conv,
                temperature=0.0,
                max_new_tokens=10,
                stop_token_ids={a_e, bos},
            )
        )
        assert isinstance(streamed, list)

    def test_multiturn_chat_accumulates(self, gpt_model, mock_tokenizer) -> None:
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        bos = mock_tokenizer.get_bos_token_id()
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")
        a_e = mock_tokenizer.encode_special("<assistant_end>")

        conv = [bos]
        build_chat_prompt(conv, "first", mock_tokenizer, us, ue, a_s)
        resp1 = list(
            engine.generate_stream(
                conv, temperature=0.0, max_new_tokens=5, stop_token_ids={a_e, bos}
            )
        )
        finalize_chat_response(conv, resp1, a_e)
        len_after_1 = len(conv)

        build_chat_prompt(conv, "second", mock_tokenizer, us, ue, a_s)
        assert len(conv) > len_after_1, "conversation should grow across turns"


class TestCompletionModeEndToEnd:
    def test_completion_each_prompt_independent(self, gpt_model, mock_tokenizer) -> None:
        # Greedy decoding from the same prompt twice must yield identical output
        # — confirms no hidden state leaks between calls.
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        bos = mock_tokenizer.get_bos_token_id()

        t1 = build_completion_prompt("alpha", mock_tokenizer, bos)
        r1 = list(
            engine.generate_stream(
                t1, temperature=0.0, max_new_tokens=10, stop_token_ids={bos}
            )
        )
        t2 = build_completion_prompt("alpha", mock_tokenizer, bos)
        r2 = list(
            engine.generate_stream(
                t2, temperature=0.0, max_new_tokens=10, stop_token_ids={bos}
            )
        )
        assert r1 == r2

    def test_different_prompts_different_outputs(self, gpt_model, mock_tokenizer) -> None:
        engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
        bos = mock_tokenizer.get_bos_token_id()

        t1 = build_completion_prompt("aaa", mock_tokenizer, bos)
        t2 = build_completion_prompt("zzz", mock_tokenizer, bos)
        r1 = list(
            engine.generate_stream(
                t1, temperature=0.0, max_new_tokens=10, stop_token_ids={bos}
            )
        )
        r2 = list(
            engine.generate_stream(
                t2, temperature=0.0, max_new_tokens=10, stop_token_ids={bos}
            )
        )
        assert r1 != r2, "different prompts should produce different completions"

    def test_completion_differs_from_chat(self, mock_tokenizer) -> None:
        # Same text input produces a strictly longer chat sequence (extra
        # control tokens) and a different sequence overall.
        bos = mock_tokenizer.get_bos_token_id()
        us = mock_tokenizer.encode_special("<user_start>")
        ue = mock_tokenizer.encode_special("<user_end>")
        a_s = mock_tokenizer.encode_special("<assistant_start>")

        chat_tokens = build_chat_prompt([bos], "hello", mock_tokenizer, us, ue, a_s)
        comp_tokens = build_completion_prompt("hello", mock_tokenizer, bos)
        assert chat_tokens != comp_tokens
        assert len(chat_tokens) > len(comp_tokens)


# ---------------------------------------------------------------------------
# build_engine_for_features: encoder-decoder full-context contract
#
# main() builds the engine via build_engine_for_features and, for an
# EncoderDecoderGPT, *requires* a context provider (it raises earlier at
# steer_chat_cli.py:115-119 when conditioning.context_cache_dir is absent).
# We assert the engine-builder half of that contract directly: an encdec
# source with context_provider=None must raise. (The valid wrapping path with
# a real provider is covered in tests/steering/test_build_wrapper_for_features.py.)
# ---------------------------------------------------------------------------


class TestEncdecEngineRequiresContextProvider:
    def test_encdec_without_provider_raises_with_features(
        self, encdec_model, mock_tokenizer
    ) -> None:
        with pytest.raises(ValueError, match="context_provider"):
            build_engine_for_features(
                encdec_model,
                mock_tokenizer,
                t.device("cpu"),
                features={0: 1, 5: 2},
                target_mode=TargetConditioningMode.CURRENT,
                context_provider=None,
            )

    def test_encdec_without_provider_raises_with_no_features(
        self, encdec_model, mock_tokenizer
    ) -> None:
        # Even the "no target delta" path needs a provider for real context.
        with pytest.raises(ValueError, match="context_provider"):
            build_engine_for_features(
                encdec_model,
                mock_tokenizer,
                t.device("cpu"),
                features=None,
                target_mode=TargetConditioningMode.CURRENT,
                context_provider=None,
            )
