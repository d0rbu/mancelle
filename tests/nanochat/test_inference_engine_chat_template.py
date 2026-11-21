"""Tests for the ``render_as_chat`` flag on ``InferenceEngine.generate*``.

The flag primes the model with the nanochat SFT chat template instead of
a raw-text prompt with BOS prepended. It was added because SFT'd models
learned to respond inside
``<user_start>…<user_end><assistant_start>…<assistant_end>`` — feeding
them raw prompts produced base-model babble.

These tests cover three concerns:

1. **Tokenization path.** ``generate(render_as_chat=True)`` calls
   ``tokenizer.render_for_completion`` instead of ``tokenizer.encode``.
   The sampled tokens flow through the KV-cache prefill path unchanged,
   so verifying the tokenization decision is the interesting contract.

2. **Stop tokens.** Both ``<bos>`` and ``<assistant_end>`` terminate
   generation in chat mode, for both the streaming/sampling path and
   the greedy batched path. The greedy-batched stop-token check was a
   latent bug (only matched ``<bos>``); the fix makes the set explicit.

3. **Capability gating.** A tokenizer that doesn't implement
   ``render_for_completion`` (e.g. ``TransformersTokenizer``,
   ``HuggingFaceTokenizer``) triggers a clear ``ValueError`` at the
   ``generate*`` entrypoint, not a cryptic ``AttributeError`` deep
   inside the tokenization step.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest
import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT
from steering.eval.inference_engine import InferenceEngine
from tests.conftest import MockTokenizer


# Force SDPA backend for all tests in this module so the CPU-only tiny GPT
# doesn't try to dispatch FA3 kernels that are CUDA-only. Matches the
# convention in tests/test_encdec_e2e.py.
@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ChatMockTokenizer(MockTokenizer):
    """MockTokenizer that records which render path was used.

    ``MockTokenizer`` already implements ``render_for_completion`` —
    override it so we can observe the call-site and distinguish chat
    rendering from raw ``encode`` in tests.
    """

    def __init__(self) -> None:
        self.render_for_completion_calls: list[dict] = []
        self.encode_calls: list[tuple] = []

    def render_for_completion(self, conversation: dict) -> list[int]:
        """Record the call and return a deterministic token list that
        includes a unique sentinel so downstream assertions can spot it.
        """
        self.render_for_completion_calls.append(conversation)
        # Unique sentinel: 77 is outside normal MockTokenizer encode range
        # (2..125) and distinct from BOS (0) / EOS (1).
        return [self._bos_id, 77, 78, 79]

    def encode(self, text, prepend=None, append=None, num_threads=8):
        self.encode_calls.append((text, prepend, append))
        return super().encode(text, prepend=prepend, append=append, num_threads=num_threads)

    def encode_special(self, name: str) -> int:
        # Deterministic, collision-free id for <assistant_end> that differs
        # from BOS/EOS/normal token range.
        if name == "<assistant_end>":
            return 95
        return super().encode_special(name)


class NoChatTokenizer:
    """Tokenizer that lacks ``render_for_completion`` — simulates
    ``TransformersTokenizer`` / ``HuggingFaceTokenizer``.
    """

    vocab_size = 128
    _bos_id = 0

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def encode(self, text, prepend=None, append=None, num_threads=8):
        ids: list[int] = []
        if prepend is not None:
            ids.append(prepend)
        ids.extend((ord(c) % 126) + 2 for c in text)
        if append is not None:
            ids.append(append)
        return ids

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(tok - 2) for tok in ids if tok not in (0, 1))

    def get_vocab_size(self) -> int:
        return self.vocab_size

    # Deliberately no render_for_completion and no encode_special.


class PartialChatTokenizer:
    """Tokenizer with ``render_for_completion`` but no ``encode_special``.

    Exercises the "both methods required" contract of the capability
    check — absence of either one should fail at the ``generate*``
    entrypoint.
    """

    vocab_size = 128
    _bos_id = 0

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def render_for_completion(self, conversation: dict) -> list[int]:
        return [self._bos_id]

    def encode(self, text, prepend=None, append=None, num_threads=8):
        return [self._bos_id]

    def decode(self, ids: list[int]) -> str:
        return ""

    def get_vocab_size(self) -> int:
        return self.vocab_size


class NoneEncodeSpecialTokenizer(ChatMockTokenizer):
    """Simulates ``HuggingFaceTokenizer.encode_special`` semantics: the
    method exists (so the hasattr-based capability check passes) but
    returns ``None`` when the token isn't in the tokenizer's vocabulary.

    This shape was the load-bearing concern behind the ``encode_special``
    validation fix — without it, ``stop_token_ids = {bos, None}`` would
    silently accept the ``None`` and then never match a sampled int,
    letting generation run the full ``max_new_tokens`` budget even when
    the model emitted ``<assistant_end>``.
    """

    def encode_special(self, name: str) -> int:
        if name == "<assistant_end>":
            return cast(int, None)  # deliberate runtime contract violation
        return super().encode_special(name)


class RaisingEncodeSpecialTokenizer(ChatMockTokenizer):
    """Simulates ``TransformersTokenizer.encode_special`` semantics: the
    method exists and raises on missing tokens (real code raises
    ``ParseError``). The ``_build_chat_prompt`` guard must wrap that
    raise with a clear ``ValueError`` so the caller gets a tokenizer-
    aware error message rather than a backend-specific leak.
    """

    def encode_special(self, name: str) -> int:
        if name == "<assistant_end>":
            raise KeyError(f"special token {name!r} not in vocab")
        return super().encode_special(name)


# ---------------------------------------------------------------------------
# 1. Capability detection in __init__
# ---------------------------------------------------------------------------


class TestChatRenderCapabilityDetection:
    """``InferenceEngine.__init__`` caches chat-render support based on
    tokenizer attributes. The cache controls the clean ValueError vs a
    cryptic AttributeError from deep inside tokenization."""

    def test_supports_chat_render_true_for_chat_tokenizer(
        self, gpt_model: GPT
    ) -> None:
        """A RustBPE-shaped tokenizer (has render_for_completion + encode_special)
        is detected as supporting chat rendering."""
        engine = InferenceEngine(gpt_model, ChatMockTokenizer(), t.device("cpu"))
        assert engine._supports_chat_render is True

    def test_supports_chat_render_false_for_no_chat_tokenizer(
        self, gpt_model: GPT
    ) -> None:
        """A tokenizer without render_for_completion does not support chat."""
        engine = InferenceEngine(gpt_model, NoChatTokenizer(), t.device("cpu"))
        assert engine._supports_chat_render is False

    def test_supports_chat_render_false_for_partial_chat_tokenizer(
        self, gpt_model: GPT
    ) -> None:
        """Has render_for_completion but no encode_special → not supported.
        (We need both: the former tokenizes, the latter resolves the
        <assistant_end> stop token id.)"""
        engine = InferenceEngine(gpt_model, PartialChatTokenizer(), t.device("cpu"))
        assert engine._supports_chat_render is False

    def test_non_chat_tokenizer_still_works_for_raw_generation(
        self, gpt_model: GPT
    ) -> None:
        """Init must NOT eagerly fail — non-chat callers with a plain
        tokenizer must still be able to build an engine and call
        generate without render_as_chat."""
        engine = InferenceEngine(gpt_model, NoChatTokenizer(), t.device("cpu"))
        out = engine.generate("hi", max_new_tokens=2, temperature=0.0)
        assert isinstance(out, str)  # no raise


# ---------------------------------------------------------------------------
# 2. ValueError at the generate* entrypoint when capability missing
# ---------------------------------------------------------------------------


class TestChatRenderRequiresCapability:
    """``render_as_chat=True`` against an unsupported tokenizer must raise
    ``ValueError`` at the entrypoint, not an ``AttributeError`` deep in
    tokenization."""

    def test_generate_render_as_chat_on_no_chat_tokenizer_raises(
        self, gpt_model: GPT
    ) -> None:
        engine = InferenceEngine(gpt_model, NoChatTokenizer(), t.device("cpu"))
        with pytest.raises(ValueError, match="render_as_chat=True requires"):
            engine.generate("hi", render_as_chat=True, temperature=0.0)

    def test_generate_batch_render_as_chat_on_no_chat_tokenizer_raises(
        self, gpt_model: GPT
    ) -> None:
        engine = InferenceEngine(gpt_model, NoChatTokenizer(), t.device("cpu"))
        with pytest.raises(ValueError, match="render_as_chat=True requires"):
            engine.generate_batch(["hi"], render_as_chat=True, temperature=0.0)

    def test_error_message_names_tokenizer_class(self, gpt_model: GPT) -> None:
        """Users should see *which* tokenizer lacks support, to pick a fix."""
        engine = InferenceEngine(gpt_model, NoChatTokenizer(), t.device("cpu"))
        with pytest.raises(ValueError, match="NoChatTokenizer"):
            engine.generate("hi", render_as_chat=True, temperature=0.0)

    def test_partial_chat_tokenizer_raises(self, gpt_model: GPT) -> None:
        """render_for_completion present but encode_special missing → raise."""
        engine = InferenceEngine(gpt_model, PartialChatTokenizer(), t.device("cpu"))
        with pytest.raises(ValueError, match="render_as_chat=True requires"):
            engine.generate("hi", render_as_chat=True, temperature=0.0)

    def test_generate_batch_raises_before_tokenization(
        self, gpt_model: GPT
    ) -> None:
        """The capability check must happen BEFORE any per-prompt work
        (avoid wasting time on 100 prompts before raising)."""
        tok = NoChatTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        # Exploit the fact that NoChatTokenizer has no encode_special —
        # attempting to build stop_token_ids for the batch would explode
        # halfway through. The entrypoint guard must raise first.
        with pytest.raises(ValueError, match="render_as_chat=True requires"):
            engine.generate_batch(
                ["hi", "there"] * 50, render_as_chat=True, temperature=0.0
            )


# ---------------------------------------------------------------------------
# 2b. encode_special edge cases — None returns and exceptions
# ---------------------------------------------------------------------------


class TestEncodeSpecialEdgeCases:
    """``_build_chat_prompt`` calls ``encode_special('<assistant_end>')``.
    Real tokenizer backends vary in how they signal "token not in
    vocabulary": HuggingFaceTokenizer returns ``None``,
    TransformersTokenizer raises ``ParseError``, RustBPETokenizer raises
    ``KeyError``. All three paths must surface as a clear ``ValueError``
    that names the tokenizer class — silently passing ``None`` through
    would disable the ``<assistant_end>`` stop token and let generation
    run to ``max_new_tokens`` every time."""

    def test_none_return_raises_value_error_with_clear_message(
        self, gpt_model: GPT
    ) -> None:
        """HuggingFaceTokenizer-style: encode_special returns None for
        missing token. We must catch that and raise, not let ``None``
        leak into ``stop_token_ids``."""
        tok = NoneEncodeSpecialTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        with pytest.raises(ValueError) as exc_info:
            engine.generate("hi", render_as_chat=True, temperature=0.0)
        # Error message must mention None and the tokenizer class.
        assert "None" in str(exc_info.value)
        assert "NoneEncodeSpecialTokenizer" in str(exc_info.value)

    def test_raising_encode_special_is_wrapped_in_value_error(
        self, gpt_model: GPT
    ) -> None:
        """TransformersTokenizer-style: encode_special raises on missing
        token. Users should see a tokenizer-aware ValueError rather than
        the raw backend error."""
        tok = RaisingEncodeSpecialTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        with pytest.raises(ValueError) as exc_info:
            engine.generate("hi", render_as_chat=True, temperature=0.0)
        # Message names the tokenizer and mentions the original exception.
        assert "RaisingEncodeSpecialTokenizer" in str(exc_info.value)
        assert "KeyError" in str(exc_info.value)

    def test_raising_encode_special_chains_original_exception(
        self, gpt_model: GPT
    ) -> None:
        """``raise ... from exc`` preserves the original traceback for
        debugging. Verify the chain is set up."""
        tok = RaisingEncodeSpecialTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        with pytest.raises(ValueError) as exc_info:
            engine.generate("hi", render_as_chat=True, temperature=0.0)
        assert isinstance(exc_info.value.__cause__, KeyError), (
            "ValueError must chain the original KeyError via `from exc` "
            "so debuggers can see the root cause"
        )

    def test_none_return_does_not_silently_disable_stop_token(
        self, gpt_model: GPT
    ) -> None:
        """The load-bearing regression guard: a None return value is a
        footgun precisely BECAUSE it doesn't crash — it produces a set
        ``{bos, None}`` that never matches a sampled int, so generation
        runs forever. Verify the fix turns this silent failure into a
        loud one.
        """
        tok = NoneEncodeSpecialTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        # Pre-fix: this call would have succeeded but generation would
        # run the full max_new_tokens budget. Post-fix: it raises loudly.
        with pytest.raises(ValueError, match="None|int token id"):
            engine.generate(
                "hi", render_as_chat=True, temperature=0.0, max_new_tokens=5
            )

    def test_non_int_return_also_rejected(self, gpt_model: GPT) -> None:
        """Defensive guard: if a tokenizer returns a non-int, non-None
        value (e.g. a list or string from a broken wrapper), we still
        reject cleanly rather than quietly using it as a set element."""

        class StringReturningTokenizer(ChatMockTokenizer):
            def encode_special(self, name: str) -> int:
                if name == "<assistant_end>":
                    return cast(int, "assistant_end")  # deliberate invalid payload
                return super().encode_special(name)

        tok = StringReturningTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        with pytest.raises(ValueError, match="int token id"):
            engine.generate("hi", render_as_chat=True, temperature=0.0)

    def test_generate_batch_also_validates_assistant_end(
        self, gpt_model: GPT
    ) -> None:
        """Both entrypoints route through _build_chat_prompt, so both
        must surface the same ValueError. Regression guard for if someone
        accidentally skips the call in generate_batch's greedy path."""
        tok = NoneEncodeSpecialTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        with pytest.raises(ValueError, match="None|int token id"):
            engine.generate_batch(
                ["hi", "there"], render_as_chat=True, temperature=0.0
            )


# ---------------------------------------------------------------------------
# 3. Tokenization path switches correctly on the flag
# ---------------------------------------------------------------------------


class TestChatRenderTokenization:
    """``render_as_chat=True`` routes through ``render_for_completion``;
    ``False`` (default) routes through raw ``encode``."""

    def test_generate_render_as_chat_uses_render_for_completion(
        self, gpt_model: GPT
    ) -> None:
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        engine.generate("hello", render_as_chat=True, temperature=0.0, max_new_tokens=2)
        assert len(tok.render_for_completion_calls) == 1
        convo = tok.render_for_completion_calls[0]
        # Pin the conversation shape the engine builds.
        assert convo["messages"][0] == {"role": "user", "content": "hello"}
        assert convo["messages"][1]["role"] == "assistant"
        # The second message content is the empty string prime — removed
        # by render_for_completion before tokenizing.
        assert convo["messages"][1]["content"] == ""

    def test_generate_raw_uses_encode(self, gpt_model: GPT) -> None:
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        engine.generate("hello", render_as_chat=False, temperature=0.0, max_new_tokens=2)
        assert len(tok.render_for_completion_calls) == 0
        assert len(tok.encode_calls) >= 1
        # First encode call is for the prompt with BOS prepended.
        text, prepend, _append = tok.encode_calls[0]
        assert text == "hello"
        assert prepend == tok.get_bos_token_id()

    def test_generate_default_is_raw(self, gpt_model: GPT) -> None:
        """Backwards-compat regression guard: default behavior unchanged."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        engine.generate("hello", temperature=0.0, max_new_tokens=2)
        assert len(tok.render_for_completion_calls) == 0

    def test_generate_batch_greedy_chat_renders_all_prompts(
        self, gpt_model: GPT
    ) -> None:
        """Greedy batched path in chat mode must render EVERY prompt."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        engine.generate_batch(
            ["a", "b", "c"],
            render_as_chat=True,
            temperature=0.0,
            max_new_tokens=2,
        )
        assert len(tok.render_for_completion_calls) == 3
        prompts_used = [c["messages"][0]["content"] for c in tok.render_for_completion_calls]
        assert prompts_used == ["a", "b", "c"]

    def test_generate_batch_sampling_chat_delegates_per_prompt(
        self, gpt_model: GPT
    ) -> None:
        """Sampling batched path forwards render_as_chat to each
        sequential generate() call."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        engine.generate_batch(
            ["a", "b"],
            render_as_chat=True,
            temperature=0.7,
            seed=42,
            max_new_tokens=2,
        )
        assert len(tok.render_for_completion_calls) == 2


# ---------------------------------------------------------------------------
# 4. Stop-token behavior in chat mode — the latent-bug regression guard
# ---------------------------------------------------------------------------


class TestChatModeStopTokens:
    """In chat mode, BOTH <bos> and <assistant_end> must terminate
    generation — streaming AND greedy-batched. The greedy branch
    previously only checked <bos>, so a chat-mode greedy call silently
    never stopped on <assistant_end>."""

    def _make_engine(self, gpt_model: GPT):
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        return engine, tok

    def test_generate_stops_on_assistant_end(self, gpt_model: GPT) -> None:
        """Single-prompt generate (streaming) must stop when the model
        emits <assistant_end>."""
        engine, tok = self._make_engine(gpt_model)
        assistant_end = tok.encode_special("<assistant_end>")
        # Force the first sampled token to be <assistant_end>.
        with patch(
            "steering.eval.inference_engine.sample_next_token",
            lambda logits, rng, temp, k: t.tensor(
                [[assistant_end]], dtype=t.long
            ),
        ):
            out = engine.generate(
                "hi", render_as_chat=True, temperature=0.0, max_new_tokens=10
            )
        # assistant_end is a stop token — must NOT appear in the decoded output.
        # MockTokenizer.decode maps token N to chr(N-2); assistant_end=95 → chr(93)
        # = ']'. The output must not contain this character.
        assert "]" not in out, f"decoded output should not contain ']' but got: {out!r}"

    def test_generate_stops_on_bos(self, gpt_model: GPT) -> None:
        """BOS is still a stop token in chat mode."""
        engine, tok = self._make_engine(gpt_model)
        bos_id = tok.get_bos_token_id()
        with patch(
            "steering.eval.inference_engine.sample_next_token",
            lambda logits, rng, temp, k: t.tensor([[bos_id]], dtype=t.long),
        ):
            out = engine.generate(
                "hi", render_as_chat=True, temperature=0.0, max_new_tokens=10
            )
        assert out == ""  # stopped immediately on BOS

    def test_generate_batch_greedy_stops_on_assistant_end(
        self, gpt_model: GPT
    ) -> None:
        """Regression guard for the latent bug: greedy batched path in
        chat mode must stop on <assistant_end>, not only <bos>."""
        engine, tok = self._make_engine(gpt_model)
        assistant_end = tok.encode_special("<assistant_end>")

        # Force every sampled token (across the batch) to be <assistant_end>.
        # If the greedy loop only checked <bos>, all items would append
        # <assistant_end> to `generated` instead of marking `finished`.
        calls = {"n": 0}

        def fake_sample(logits, rng, temp, k):
            calls["n"] += 1
            # Shape must be (B, 1) like the real _sample_token.
            B = logits.shape[0]
            return t.full((B, 1), assistant_end, dtype=t.long)

        with patch(
            "steering.eval.inference_engine.sample_next_token",
            fake_sample,
        ):
            results = engine.generate_batch(
                ["p1", "p2", "p3"],
                render_as_chat=True,
                temperature=0.0,
                max_new_tokens=20,
            )
        # Every prompt should have stopped immediately → empty generated
        # token list → empty decoded string.
        assert results == ["", "", ""]
        # The sample call count tells us we stopped after the first token:
        # loop checks all(finished) after marking, then breaks. So the
        # fake_sample was called exactly once (for the single iteration).
        assert calls["n"] == 1

    def test_generate_batch_greedy_raw_mode_only_bos_stops(
        self, gpt_model: GPT
    ) -> None:
        """Raw mode (render_as_chat=False) must continue to stop only on
        BOS — <assistant_end> is NOT a stop token there (it's just an
        ordinary vocab id). Regression guard that the chat fix didn't
        leak stop-token behavior into raw mode."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        assistant_end = tok.encode_special("<assistant_end>")

        # Emit <assistant_end> first, then BOS. In raw mode, the first
        # should be appended as a regular token; the second stops.
        emitted = iter([assistant_end, tok.get_bos_token_id()])

        def fake_sample(logits, rng, temp, k):
            B = logits.shape[0]
            tok_id = next(emitted)
            return t.full((B, 1), tok_id, dtype=t.long)

        with patch(
            "steering.eval.inference_engine.sample_next_token",
            fake_sample,
        ):
            results = engine.generate_batch(
                ["hello"], render_as_chat=False, temperature=0.0, max_new_tokens=10
            )
        # <assistant_end> was appended; its decoded form (chr(95-2)=']')
        # must be in the output. Only BOS stopped the loop.
        assert results[0] == "]"


# ---------------------------------------------------------------------------
# 5. harmfulness_eval.py pipes render_as_chat=True
# ---------------------------------------------------------------------------


class TestHarmfulnessEvalPipesChatRender:
    """Source-string + call-site guard that ``scripts/harmfulness_eval.py``
    and the inline ``scripts/training_progression.py`` harmfulness code
    pass ``render_as_chat=True``. Prevents silent regression if someone
    rewrites either call site without threading the flag through."""

    def test_harmfulness_eval_source_contains_render_as_chat(self) -> None:
        src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "eval"
            / "harmfulness_eval.py"
        )
        text = src.read_text()
        assert "render_as_chat=True" in text, (
            "scripts/eval/harmfulness_eval.py must pass render_as_chat=True "
            "to InferenceEngine.generate_batch"
        )

    def test_training_progression_inline_harmfulness_uses_render_as_chat(
        self,
    ) -> None:
        src = (
            Path(__file__).resolve().parent.parent.parent
            / "scripts"
            / "train"
            / "training_progression.py"
        )
        functions = {
            node.name: node for node in ast.parse(src.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        decoder_calls = [node for node in ast.walk(functions["run_harmfulness_eval"])
                         if isinstance(node, ast.Call)]
        assert any(
            isinstance(call.func, ast.Attribute) and call.func.attr == "generate_batch"
            and any(keyword.arg == "render_as_chat" and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True for keyword in call.keywords)
            for call in decoder_calls
        ), (
            "scripts/train/training_progression.py must pass "
            "render_as_chat=True to InferenceEngine.generate_batch so "
            "progression and standalone harmfulness eval don't diverge on "
            "the same checkpoint"
        )
        self_conditioned_calls = [
            node for node in ast.walk(functions["run_self_conditioned_harmfulness_eval"])
            if isinstance(node, ast.Call)
        ]
        assert any(
            isinstance(call.func, ast.Name) and call.func.id == "render_prompt_for_chat_completion"
            for call in self_conditioned_calls
        ), (
            "scripts/train/training_progression.py self-conditioned harmfulness "
            "must use the shared chat renderer before cache admission and generation"
        )


# ---------------------------------------------------------------------------
# 6. End-to-end with real tiny model (no sample-token mocks)
# ---------------------------------------------------------------------------


class TestChatGenerationEndToEnd:
    """Drive ``engine.generate(render_as_chat=True)`` through a real tiny
    GPT's forward pass, with no ``_sample_token`` mocks. These tests
    exercise the full pipeline: render → prefill → greedy decode → stop.

    Previous tests verified:
      - the tokenization decision (render_for_completion vs encode)
      - the stop-token semantics (by forcing sampled values)

    What they COULDN'T verify is that the real decode loop against real
    logits (a) runs without raising, (b) actually stops cleanly within
    max_new_tokens, (c) returns a string.
    """

    def test_generate_chat_real_decode_returns_string(
        self, gpt_model: GPT
    ) -> None:
        """A real forward pass shouldn't raise in chat mode with a
        sensibly-configured tiny model. Output is a string. No masking of
        internal steps."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        out = engine.generate(
            "hello world",
            render_as_chat=True,
            temperature=0.0,
            max_new_tokens=4,
        )
        assert isinstance(out, str)

    def test_generate_chat_real_decode_respects_max_new_tokens(
        self, gpt_model: GPT
    ) -> None:
        """When the model never emits a stop token, generation must cap
        at max_new_tokens. A decoded length proxy is enough: every
        MockTokenizer.decode char maps to one token (minus BOS/EOS
        filtering), so len(out) <= max_new_tokens."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        out = engine.generate(
            "hi",
            render_as_chat=True,
            temperature=0.0,
            max_new_tokens=3,
        )
        assert len(out) <= 3, (
            f"generation exceeded max_new_tokens=3: len(out)={len(out)}, out={out!r}"
        )

    def test_generate_chat_real_decode_vs_raw_produces_different_prefill(
        self, gpt_model: GPT
    ) -> None:
        """Same prompt with render_as_chat=True vs False → different
        prefill token sequences → different (but both valid) outputs.
        This is an end-to-end observable that the flag actually changed
        the input to the model, not just the path through the function."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        out_chat = engine.generate(
            "hi", render_as_chat=True, temperature=0.0, max_new_tokens=3
        )
        # Reset tokenizer call log so raw mode's calls are separable.
        tok.render_for_completion_calls = []
        out_raw = engine.generate(
            "hi", render_as_chat=False, temperature=0.0, max_new_tokens=3
        )
        # The raw prompt was encoded, not chat-rendered.
        assert len(tok.render_for_completion_calls) == 0
        # Both outputs are strings; we don't assert inequality (greedy
        # decode could converge on the same output) — but at minimum the
        # tokenization path differed.
        assert isinstance(out_chat, str)
        assert isinstance(out_raw, str)

    def test_generate_batch_chat_real_decode_returns_list(
        self, gpt_model: GPT
    ) -> None:
        """generate_batch greedy path in chat mode, real decode. One
        string per input prompt; none exceeds max_new_tokens."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        results = engine.generate_batch(
            ["one", "two", "three"],
            render_as_chat=True,
            temperature=0.0,
            max_new_tokens=3,
        )
        assert len(results) == 3
        assert all(isinstance(r, str) for r in results)
        assert all(len(r) <= 3 for r in results), (
            f"some batch element exceeded max_new_tokens=3: {results}"
        )

    def test_generate_batch_chat_sampling_real_decode(
        self, gpt_model: GPT
    ) -> None:
        """Sampling path (temperature > 0) uses sequential per-prompt
        generate() calls. Verify real decode works and returns strings."""
        tok = ChatMockTokenizer()
        engine = InferenceEngine(gpt_model, tok, t.device("cpu"))
        results = engine.generate_batch(
            ["a", "b"],
            render_as_chat=True,
            temperature=0.7,
            seed=123,
            max_new_tokens=3,
        )
        assert len(results) == 2
        assert all(isinstance(r, str) for r in results)
