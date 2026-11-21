"""Shared fixtures and pytest configuration for the test suite."""

from __future__ import annotations

import logging
import sys
import types
from pathlib import Path
from typing import cast

import pytest
import torch as t
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger as _loguru_logger

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import scripts.train.conditional_encdec_sft as conditional_encdec_sft
from nanochat.gpt import GPT, Block, GPTConfig
from nanochat.tokenizer import get_tokenizer
from steering.conditional.encoder_decoder import sft_features
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer
from steering.conditional.encoder_decoder.encoder import FeatureEncoder
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.context_cache_io import CacheIdentity
from steering.refined_types import PositiveInt
from tests.axbench_base._multiprocess_helpers import run_gloo_multiprocess
from tests.padded_encoder_reference import (
    cross_attention_forward_with_cpu_reference,
    feature_encoder_forward_packed_with_cpu_reference,
    feature_encoder_forward_with_cpu_reference,
)

# Shared default identity for context-cache tests. Each test that needs a
# different identity should build its own CacheIdentity instance.
TEST_CACHE_IDENTITY = CacheIdentity(
    model_id="test-model",
    sae_id="test-sae",
    layer=0,
    pooling="mean",
    feature_selection_hash="0" * 64,
)
TEST_CACHE_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(8)
TEST_CACHE_NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)
_SFT_CANONICAL_FINGERPRINT_CHECK: object | None = None


@pytest.fixture(autouse=True)
def _install_test_owned_cpu_encoder_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep CPU semantic coverage without shipping a CPU encoder implementation."""
    monkeypatch.setattr(
        FeatureEncoder,
        "forward",
        feature_encoder_forward_with_cpu_reference,
    )
    monkeypatch.setattr(
        FeatureEncoder,
        "forward_packed",
        feature_encoder_forward_packed_with_cpu_reference,
    )
    monkeypatch.setattr(
        CrossAttentionLayer,
        "forward",
        cross_attention_forward_with_cpu_reference,
    )


def pytest_addoption(parser):
    parser.addoption(
        "--model-scale",
        action="store",
        default="full",
        choices=["full", "tiny"],
        help=(
            "Model scale for integration tests. "
            "'full' loads real GPT-2 + SAE (requires ~2GB download, GPU recommended). "
            "'tiny' uses a small random HookedTransformer with mock SAE weights (fast, CPU-only)."
        ),
    )


@pytest.fixture(scope="session")
def model_scale(request):
    """Return the model scale chosen via --model-scale CLI option."""
    return request.config.getoption("--model-scale")


@pytest.fixture(autouse=True)
def _assert_sft_loader_patch_is_scoped(request: pytest.FixtureRequest):
    """Fail at the boundary of an SFT test if its loader patch escaped.

    ``patch_load_sft_features`` replaces the loader imported into the training
    script (and, by default, its fingerprint assertion alongside it).  A
    suspended generator can otherwise keep those replacements alive until
    garbage collection, making the next test depend on execution order.  The
    training script is already imported at module collection time; limiting
    this fixture to SFT-named nodes keeps only the patch-leak assertions off
    unrelated tests. Setup and teardown checks make the first victim and every
    subsequent victim identify the leaking producer.
    """

    if "sft" not in request.node.nodeid.lower():
        yield
        return

    global _SFT_CANONICAL_FINGERPRINT_CHECK
    if _SFT_CANONICAL_FINGERPRINT_CHECK is None:
        _SFT_CANONICAL_FINGERPRINT_CHECK = (
            conditional_encdec_sft._assert_sft_mixture_fingerprint_matches_task
        )
    expected_loader = sft_features.load_sft_features
    expected_fingerprint_check = _SFT_CANONICAL_FINGERPRINT_CHECK

    def assert_loader_restored(phase: str) -> None:
        actual_loader = conditional_encdec_sft.load_sft_features
        assert actual_loader is expected_loader, (
            f"SFT feature loader leaked across {phase} of {request.node.nodeid}: "
            f"expected {expected_loader!r}, got {actual_loader!r}"
        )
        actual_fingerprint_check = (
            conditional_encdec_sft._assert_sft_mixture_fingerprint_matches_task
        )
        assert actual_fingerprint_check is expected_fingerprint_check, (
            f"SFT fingerprint check leaked across {phase} of "
            f"{request.node.nodeid}: expected {expected_fingerprint_check!r}, "
            f"got {actual_fingerprint_check!r}"
        )

    assert_loader_restored("test setup")
    yield
    assert_loader_restored("test teardown")


@pytest.fixture()
def hydra_compose():
    """Compose a Hydra config from a config directory + name + optional overrides."""

    def _compose(
        config_dir: str,
        config_name: str,
        overrides: list[str] | None = None,
    ):
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            return compose(config_name=config_name, overrides=overrides or [])

    return _compose


# ---------------------------------------------------------------------------
# Module-level constants for the tiny test models
# ---------------------------------------------------------------------------

TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_HEAD_DIM: int = 32
TINY_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(64)
TINY_NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)


# ---------------------------------------------------------------------------
# Validated accessors for nanochat's dynamically typed module containers
# ---------------------------------------------------------------------------


def _gpt_embedding(model: GPT) -> t.nn.Embedding:
    embedding = model.transformer["wte"]
    if not isinstance(embedding, t.nn.Embedding):
        raise TypeError(
            f"GPT transformer['wte'] must be nn.Embedding, got {type(embedding).__name__}"
        )
    return embedding


def _gpt_blocks(model: GPT) -> tuple[Block, ...]:
    modules = model.transformer["h"]
    if not isinstance(modules, t.nn.ModuleList):
        raise TypeError(f"GPT transformer['h'] must be nn.ModuleList, got {type(modules).__name__}")
    blocks: list[Block] = []
    for index, module in enumerate(modules):
        if not isinstance(module, Block):
            raise TypeError(f"GPT decoder block {index} must be Block, got {type(module).__name__}")
        blocks.append(module)
    return tuple(blocks)


def _gpt_block(model: GPT, index: int) -> Block:
    return _gpt_blocks(model)[index]


def _gpt_value_embedding(model: GPT, layer_key: str) -> t.nn.Embedding:
    embedding = model.value_embeds[layer_key]
    if not isinstance(embedding, t.nn.Embedding):
        raise TypeError(
            f"GPT value_embeds[{layer_key!r}] must be nn.Embedding, got {type(embedding).__name__}"
        )
    return embedding


# ---------------------------------------------------------------------------
# MockTokenizer
# ---------------------------------------------------------------------------


class MockTokenizer:
    """Minimal char-level tokenizer for testing.

    * vocab_size = 128
    * Token 0 = BOS, token 1 = EOS.
    * ``encode`` maps each character to ``(ord(c) % (vocab_size - 2)) + 2``.
    * ``decode`` reverses the mapping back to characters.
    """

    vocab_size: int = TINY_VOCAB
    _bos_id: int = 0
    _eos_id: int = 1
    # Opt-in collision check: when True, ``encode`` and
    # ``render_conversation`` raise ``ValueError`` if any input character
    # would alias to a chat-special-token id (see ``encode_special``).
    # Default ``False`` preserves the historical permissive behaviour
    # for the ~580 existing call sites.  Manifest §1.17.
    _collision_check: bool = False

    @classmethod
    def with_collision_check(cls) -> MockTokenizer:
        """Construct a ``MockTokenizer`` with opt-in collision checking.

        Equivalent to ``MockTokenizer(_collision_check=True)``; provided
        as a named factory so test files document their intent at the
        call site (the historical permissive default is preserved).
        """
        instance = cls()
        instance._collision_check = True
        return instance

    def _check_collision(self, text: str) -> None:
        """Raise ``ValueError`` when *text* contains characters whose
        regular ``encode`` ids would alias the chat-special token ids.

        With ``vocab_size=128`` and the formula
        ``(ord(c) % (vocab_size-2)) + 2 = (ord(c) % 126) + 2``,
        the special-token ids 64–67 alias to ord 62 (``>``), 63 (``?``),
        64 (``@``), 65 (``A``).  Any of those characters in ``text``
        means downstream tests cannot disambiguate plain text from
        chat-special tokens.
        """
        # Compute the exact alias set for the current vocab_size, so that
        # changes to ``vocab_size`` keep the helper accurate.
        special_ids = {64, 65, 66, 67}  # <user_start> .. <assistant_end>
        for c in text:
            tok_id = (ord(c) % (self.vocab_size - 2)) + 2
            if tok_id in special_ids:
                raise ValueError(
                    f"MockTokenizer collision: character {c!r} (ord={ord(c)}) "
                    f"encodes to id {tok_id}, which aliases a chat-special token. "
                    "This MockTokenizer was created with collision-checking enabled "
                    "(see manifest §1.17). Either rephrase the test prompt to avoid "
                    "the colliding character or use a different fixture."
                )

    def __call__(self, *args, **kwargs) -> list[int] | list[list[int]]:
        """Allow ``tokenizer(text, ...)`` as a shorthand for ``tokenizer.encode(...)``."""
        return self.encode(*args, **kwargs)

    def encode(
        self,
        text: str | list[str],
        prepend: int | None = None,
        append: int | None = None,
        num_threads: int = 8,
    ) -> list[int] | list[list[int]]:
        """Encode *text* to a list of token ids.

        Matches the real ``RustBPETokenizer.encode`` signature: *prepend*
        and *append* are single ``int`` token ids (not lists).  When *text*
        is a list of strings, each element is encoded independently and
        a list of lists is returned.
        """
        if isinstance(text, list):
            return [cast(list[int], self.encode(t, prepend=prepend, append=append)) for t in text]
        if self._collision_check:
            self._check_collision(text)
        ids: list[int] = []
        if prepend is not None:
            ids.append(prepend)
        ids.extend((ord(c) % (self.vocab_size - 2)) + 2 for c in text)
        if append is not None:
            ids.append(append)
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token ids back to a string.

        BOS (0) and EOS (1) tokens are silently skipped.
        """
        chars: list[str] = []
        for tok in ids:
            if tok in (self._bos_id, self._eos_id):
                continue
            chars.append(chr(tok - 2))
        return "".join(chars)

    def get_bos_token_id(self) -> int:
        """Return the BOS token id."""
        return self._bos_id

    def encode_special(self, name: str) -> int:
        """Return the id for a named special token.

        The chat special IDs (64–67) DO collide with ``encode()`` output for
        some ASCII characters (e.g. ``encode("A")`` → 66,
        ``encode("?")`` → 65) via the ``(ord(c) % 126) + 2`` formula, so
        they are not globally collision-free.  They are safe for
        ``tests/test_target_eval_adapter.py`` specifically, which uses
        CACHED mode + MockTokenizer and never passes the colliding ASCII
        characters in its test prompts.  Unknown tokens return a fixed
        sentinel derived from the name string so that Engine's tool-use
        state machine can resolve specials without crashing.
        """
        mapping = {
            "bos": self._bos_id,
            "eos": self._eos_id,
            # Chat special tokens: fixed IDs in [64, 67] that don't collide
            # with any ASCII text token ids used in the test suite.
            "<user_start>": 64,
            "<user_end>": 65,
            "<assistant_start>": 66,
            "<assistant_end>": 67,
        }
        if name in mapping:
            return mapping[name]
        # Unknown tokens: deterministic sentinel in upper vocab range.
        # Simple sum-of-bytes formula avoids hash-seed randomisation.
        sentinel = (sum(ord(c) for c in name) % (self.vocab_size // 4)) + 3 * (self.vocab_size // 4)
        return sentinel

    def render_conversation(
        self, conversation: dict, max_tokens: int = 2048
    ) -> tuple[list[int], list[int]]:
        """Render a full conversation (including assistant response) as (ids, mask).

        Mirrors the behaviour of ``RustBPETokenizer.render_conversation``:
        mask[i] == 1 for assistant content tokens and the ``<assistant_end>``
        token; 0 for all other tokens (BOS, user turns, special tokens
        surrounding the user turn, ``<assistant_start>``).
        """
        ids: list[int] = []
        mask: list[int] = []

        def _add(token_ids: list[int], mask_val: int) -> None:
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        user_start = self.encode_special("<user_start>")
        user_end = self.encode_special("<user_end>")
        assistant_start = self.encode_special("<assistant_start>")
        assistant_end = self.encode_special("<assistant_end>")

        _add([self._bos_id], 0)
        messages = conversation.get("messages", [])
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "user":
                _add([user_start], 0)
                _add(cast(list[int], self.encode(content)), 0)
                _add([user_end], 0)
            elif role == "assistant":
                _add([assistant_start], 0)
                if content:
                    _add(cast(list[int], self.encode(content)), 1)
                _add([assistant_end], 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation: dict) -> list[int]:
        """Render a conversation dict as token ids for completion.

        Supports two formats:
        1. ``{"text": "..."}`` — encodes the text with BOS prepended.
        2. ``{"messages": [{"role": "user", "content": "..."}, ...]}`` — mimics
           the nanochat chat template: BOS + user_start + content + user_end +
           assistant_start (for the last user turn followed by an empty assistant
           turn, which is the pattern used by ``InferenceEngine._build_chat_prompt``
           for chat-style steering evals).
        """
        if "messages" in conversation:
            ids: list[int] = [self._bos_id]
            user_start = self.encode_special("<user_start>")
            user_end = self.encode_special("<user_end>")
            assistant_start = self.encode_special("<assistant_start>")
            for msg in conversation["messages"]:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    ids.append(user_start)
                    ids.extend(cast(list[int], self.encode(content)))
                    ids.append(user_end)
                elif role == "assistant":
                    ids.append(assistant_start)
                    if content:
                        ids.extend(cast(list[int], self.encode(content)))
                    # Only append assistant_end if content is non-empty (completion mode
                    # ends at assistant_start for empty assistant turns).
                    else:
                        pass  # leave open for generation (no assistant_end)
            return ids
        # Legacy "text" format
        text = conversation.get("text", "")
        # encode(str) always returns list[int] (not list[list[int]])
        result = self.encode(text, prepend=self._bos_id)
        assert isinstance(result[0], int)
        return cast(list[int], result)

    def get_vocab_size(self) -> int:
        """Return the vocabulary size."""
        return self.vocab_size


# ---------------------------------------------------------------------------
# Tiny model factories
# ---------------------------------------------------------------------------


def _make_gpt_config() -> GPTConfig:
    """Return a GPTConfig suitable for fast CPU tests."""
    return GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )


def _make_gpt(device: str = "cpu") -> GPT:
    """Build a tiny GPT, initialise weights, and return in eval mode."""
    config = _make_gpt_config()
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device=device)
    model.init_weights()
    model.eval()
    return model


def _make_encdec(device: str = "cpu") -> EncoderDecoderGPT:
    """Build a tiny EncoderDecoderGPT, initialise weights, and return in eval mode."""
    dec_config = _make_gpt_config()
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(
        frequency=1,
    )
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=xattn_config,
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=TINY_NUM_BUCKETS,
        pad_vocab_size_to=TINY_VOCAB,
    )
    model = model.to(device)
    model.init_weights()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


def build_tiny_decoder(device: str = "cpu") -> GPT:
    """Public alias for ``_make_gpt`` — for use in test files that need a
    decoder-only GPT model (e.g. to test rejection of feature conditioning)."""
    return _make_gpt(device=device)


def build_tiny_encdec(device: str = "cpu") -> EncoderDecoderGPT:
    """Public alias for ``_make_encdec`` — for use in test files that import a
    named helper rather than the private ``_make_encdec`` function."""
    return _make_encdec(device=device)


def build_tiny_tokenizer() -> MockTokenizer:
    """Return a fresh MockTokenizer instance — public alias for test files."""
    return MockTokenizer()


def get_tokenizer_or_skip():
    """Return the real cached nanochat tokenizer, or ``pytest.skip`` when its
    git-ignored artifact (``nanochat_cache/tokenizer/tokenizer.pkl``) is absent
    — fresh clone / CI / git worktree.

    Tests that need real BPE fidelity (SFT char-offset / segment alignment)
    must use this instead of a bare ``get_tokenizer()`` so a missing cache is
    a clean, self-documenting SKIP rather than a setup ERROR. Mirrors the
    proven pattern in ``tests/scripts/test_token_to_char_offsets.py``.
    """
    try:
        return get_tokenizer()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(
            "nanochat tokenizer cache unavailable "
            "(build it with `python -m scripts.preprocess.tok_train`): "
            f"{exc}"
        )


def real_chat_rendered_prompt(tokenizer, query: str) -> list[int]:
    """Render a chat-style prompt ending in <assistant_start>."""
    conv = {
        "messages": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": ""},
        ]
    }
    return tokenizer.render_for_completion(conv)


@pytest.fixture()
def mock_tokenizer() -> MockTokenizer:
    """Return a fresh MockTokenizer instance."""
    return MockTokenizer()


@pytest.fixture()
def gpt_model() -> GPT:
    """Return a tiny GPT model on CPU, initialised and in eval mode."""
    return _make_gpt(device="cpu")


@pytest.fixture()
def encdec_model() -> EncoderDecoderGPT:
    """Return a tiny EncoderDecoderGPT on CPU, initialised and in eval mode."""
    return _make_encdec(device="cpu")


@pytest.fixture()
def encdec_engine(mock_tokenizer: MockTokenizer):
    """Return a namespace with (model, tokenizer, device) for a tiny encoder-decoder model.

    Exposes:
    - ``.model``: the ``EncoderDecoderGPT`` instance
    - ``.tokenizer``: a ``MockTokenizer``
    - ``.device``: ``torch.device("cpu")``
    - ``.enc_config``: shortcut to ``model.enc_config``
    """
    model = _make_encdec(device="cpu")
    device = t.device("cpu")
    ns = types.SimpleNamespace(
        model=model,
        tokenizer=mock_tokenizer,
        device=device,
        # enc_config: the FeatureEncoderConfig (has n_sae_features + num_buckets)
        enc_config=model.encoder.config,
    )
    return ns


# ---------------------------------------------------------------------------
# Distributed / multiprocess fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def gloo_multiprocess():
    """Spawn N gloo-backed processes; run a target in each.

    Usage:
        def my_target(rank, world_size, *args):
            ...
            return some_result
        results = gloo_multiprocess(world_size, my_target, *args)

    Skips the test cleanly when the gloo backend is not available
    (e.g., torch built without distributed support). Without this
    skip the test would error loudly inside the worker child process
    with a confusing ``RuntimeError`` instead of ``SKIP``.
    """
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed gloo backend not available")
    return run_gloo_multiprocess


# ---------------------------------------------------------------------------
# Make loguru log records visible to pytest's caplog.
# ---------------------------------------------------------------------------


class _LoguruToStdlibHandler(logging.Handler):
    """Forward loguru records to stdlib logging so pytest's caplog sees them."""

    def emit(self, record: logging.LogRecord) -> None:
        logging.getLogger(record.name).handle(record)


@pytest.fixture
def caplog(caplog):
    """Augment pytest's caplog so loguru.logger.* calls are captured."""
    handler_id = _loguru_logger.add(
        _LoguruToStdlibHandler(),
        format="{message}",
        level=0,
    )
    caplog.set_level(0)
    yield caplog
    _loguru_logger.remove(handler_id)
