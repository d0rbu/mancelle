"""CPU token-equality oracle for ChatSession and InferenceEngine."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import Block, Linear
from scripts.chat_tui.session import (
    ChatSession,
    GenerationEvent,
    GenerationEventKind,
    build_session_self_cond_config,
)
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import SingleEntryContextProvider
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import build_tiny_model
from tests.test_chat_tui.test_session_driver import (
    StubExtractionService,
    _build_session,
    _config,
    _setup,
)

_CPU = t.device("cpu")
_SEED = 18
_CONTROL_SEED = 20
_PROMPT_TEXT = "1"


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _engine_self_cond_config(session: ChatSession, *, seed: int) -> SelfCondConfig:
    """Use the session driver's builder with a fresh selection RNG."""

    selection_rng = t.Generator(device="cpu")
    selection_rng.manual_seed(seed + 1)
    return build_session_self_cond_config(
        session._setup,
        session._config,
        selection_rng,
        session._score_policy,
        segment_mode=session._setup.segment_mode,
    )


def _run_oracle(
    *,
    seed: int = _SEED,
    segment_size: int = 4,
    max_new_tokens: int,
    sequence_len: int = 96,
    head_layers: int = 0,
) -> tuple[
    ChatSession,
    MockTokenizer,
    StubExtractionService,
    tuple[int, ...],
    list[int],
    list[int],
    int,
    int,
]:
    session, tokenizer, extraction = _build_session(
        segment_size=segment_size,
        max_new_tokens=max_new_tokens,
        sequence_len=sequence_len,
    )
    if head_layers:
        model = session._model
        assert isinstance(model.feature_pred_head, Linear)
        assert model.feature_pred_config is not None
        model.feature_pred_head = ResidualFeaturePredictionHead(
            model.feature_pred_head, head_layers, FeaturePredictionBlock.TRANSFORMER,
            decoder_config=model.dec_config,
        )
        model.feature_pred_config = replace(
            model.feature_pred_config,
            block=FeaturePredictionBlock.TRANSFORMER,
            head_layers=head_layers,
        )
        with t.no_grad():
            for block in model.feature_pred_head.residual_blocks:
                assert isinstance(block, Block)
                block.attn.c_proj.weight.normal_(0, 0.03)
                block.mlp.c_proj.weight.normal_(0, 0.03)
    if seed != session._config.seed:
        session._config = replace(session._config, seed=seed)
        session.clear()

    events: list[GenerationEvent] = []
    session.send_user_turn(_PROMPT_TEXT, events.append)

    prompt = extraction.calls[0]
    aligned = session._aligned_segments
    assert aligned is not None
    conversation = session.conversation_tokens()
    assert session.token_budget()[0] == len(conversation)
    sampled = list(conversation[len(prompt) : -1])
    event_tokens = [
        int(event.detail) for event in events if event.kind is GenerationEventKind.TOKEN
    ]
    assert event_tokens == [*sampled, tokenizer.encode_special("<assistant_end>")]

    sampling_cap = min(
        max_new_tokens,
        sequence_len - len(prompt) - 1,
    )
    engine_config = _engine_self_cond_config(session, seed=seed)
    engine = build_engine_self_conditioned(
        session._model,
        tokenizer,
        _CPU,
        engine_config,
        context_provider=SingleEntryContextProvider(prompt, aligned),
    )
    engine_tokens = list(
        engine.generate_stream(
            list(prompt),
            max_new_tokens=sampling_cap,
            temperature=session._config.temperature,
            top_k=session._config.top_k,
            seed=seed,
            stop_token_ids={
                tokenizer.get_bos_token_id(),
                tokenizer.encode_special("<assistant_end>"),
            },
        )
    )
    engine_boundaries = engine._model.self_cond_runtime_snapshot().state.target_slot_idx - len(
        aligned
    )
    return (
        session,
        tokenizer,
        extraction,
        prompt,
        sampled,
        engine_tokens,
        sampling_cap,
        engine_boundaries,
    )


def _assert_equal_streams(
    session_tokens: list[int],
    engine_tokens: list[int],
    *,
    seed: int,
) -> None:
    mismatch = next(
        (
            index
            for index, (session_token, engine_token) in enumerate(
                zip(session_tokens, engine_tokens, strict=False)
            )
            if session_token != engine_token
        ),
        min(len(session_tokens), len(engine_tokens)),
    )
    session_value: int | str = (
        session_tokens[mismatch] if mismatch < len(session_tokens) else "<end>"
    )
    engine_value: int | str = engine_tokens[mismatch] if mismatch < len(engine_tokens) else "<end>"
    assert engine_tokens == session_tokens, (
        f"token-stream divergence at index {mismatch}: "
        f"session={session_value}, engine={engine_value}; "
        "session RNGs: token_rng=t.Generator(device='cpu').manual_seed"
        f"({seed}), selection_rng=t.Generator(device='cpu').manual_seed({seed + 1}); "
        "engine RNGs: generate_stream(seed="
        f"{seed}) creates t.Generator(device='cpu').manual_seed({seed}), "
        "SelfCondConfig selection_rng=t.Generator(device='cpu').manual_seed"
        f"({seed + 1})"
    )


def _assert_session_contract(
    session: ChatSession,
    tokenizer: MockTokenizer,
    prompt: tuple[int, ...],
    sampled: list[int],
) -> None:
    close = tokenizer.encode_special("<assistant_end>")
    conversation = session.conversation_tokens()
    assert conversation == (*prompt, *sampled, close)
    assert session._kv_cache is not None
    assert int(session._kv_cache.get_pos()) == len(conversation)
    assert session.token_budget()[0] == len(conversation)
    assert len(conversation) == len(prompt) + len(sampled) + 1


def test_stop_token_stream_equals_generate_stream() -> None:
    # A wide cap keeps the "a stop was actually sampled" non-vacuity guard
    # robust: the equality assertions compare two in-process streams and are
    # immune to float perturbation, but WHERE the stop lands can shift under
    # environment-dependent kernel reduction order (thread count, ISA). With
    # ~200 draws a stop before the cap is certain for any such perturbation.
    (
        session,
        tokenizer,
        _extraction,
        prompt,
        sampled,
        engine_tokens,
        sampling_cap,
        _engine_boundaries,
    ) = _run_oracle(max_new_tokens=200)

    assert len(sampled) < sampling_cap, "fixed seed did not exercise a sampled stop"
    _assert_equal_streams(sampled, engine_tokens, seed=_SEED)
    _assert_session_contract(session, tokenizer, prompt, sampled)
    # generate_stream never forwards the sampled stop, so its cache ends at
    # prefill + n_sampled. The session alone forwards one forced close; its
    # public KV/token contract above is the observable contract asserted here.


def test_learned_transformer_head_stream_matches_independent_engine_across_boundaries() -> None:
    session, tokenizer, _extraction, prompt, sampled, expected, _cap, boundaries = _run_oracle(
        seed=_CONTROL_SEED, segment_size=2, max_new_tokens=11, head_layers=2,
    )
    _assert_equal_streams(sampled, expected, seed=_CONTROL_SEED)
    assert len(sampled) >= 5 and boundaries >= 2
    _assert_session_contract(session, tokenizer, prompt, sampled)
    assert session._kv_cache is not None
    assert session._kv_cache.n_layers == session._model.num_cache_layers
    assert session._kv_cache.n_layers == session._model.dec_config.n_layer + 2


def test_cap_stream_equals_generate_stream_and_forces_close() -> None:
    cap = 3
    (
        session,
        tokenizer,
        _extraction,
        prompt,
        sampled,
        engine_tokens,
        sampling_cap,
        _engine_boundaries,
    ) = _run_oracle(max_new_tokens=cap)

    assert sampling_cap == cap
    assert len(sampled) == cap, "fixed seed unexpectedly sampled a stop before the cap"
    _assert_equal_streams(sampled, engine_tokens, seed=_SEED)
    _assert_session_contract(session, tokenizer, prompt, sampled)
    # The engine yields its final capped token but does not forward it. The
    # session forwards all three sampled tokens and then its forced close.


def test_small_k_crosses_multiple_boundaries_with_full_token_equality() -> None:
    cap = 11
    (
        session,
        tokenizer,
        _extraction,
        prompt,
        sampled,
        engine_tokens,
        sampling_cap,
        engine_boundaries,
    ) = _run_oracle(segment_size=4, max_new_tokens=cap)

    assert sampling_cap == cap
    assert len(sampled) == cap, "fixed seed unexpectedly sampled a stop before the cap"
    assert len(session._snapshots) - 1 >= 2
    assert engine_boundaries >= 2
    _assert_equal_streams(sampled, engine_tokens, seed=_SEED)
    _assert_session_contract(session, tokenizer, prompt, sampled)


def test_different_seed_changes_the_sampled_sequence() -> None:
    first = _run_oracle(max_new_tokens=11, seed=_SEED)
    second = _run_oracle(max_new_tokens=11, seed=_CONTROL_SEED)

    _assert_equal_streams(first[4], first[5], seed=_SEED)
    _assert_equal_streams(second[4], second[5], seed=_CONTROL_SEED)
    assert first[4] != second[4]


def test_compact_vocab_session_send_rewind_modify_uses_original_ids() -> None:
    original_ids = (8, 10, 20, 25, 27, 30)
    vocab = FeatureVocab(32, original_ids)
    t.manual_seed(0)
    model = build_tiny_model(
        n_sae_features=32,
        num_buckets=2,
        seq_len=96,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
        feature_vocab=vocab,
    )
    model.eval()
    tokenizer = MockTokenizer()

    class CompactExtraction(StubExtractionService):
        def prepare(self, token_ids: tuple[int, ...]):
            prepared = super().prepare(token_ids)
            segments = tuple(
                replace(segment, feature_ids=(original_ids[index],))
                for index, segment in enumerate(prepared.segments)
            )
            return replace(prepared, segments=segments)

    setup = _setup()
    setup.n_sae_features = 32
    setup.feature_vocab = vocab
    setup.percentile_table = t.zeros((32, 99), dtype=t.float64)
    extraction = CompactExtraction(tokenizer)
    session = ChatSession(
        setup=setup,
        model=model,
        tokenizer=tokenizer,
        device=_CPU,
        config=_config(segment_size=4, max_new_tokens=5),
        extraction=extraction,
    )

    session.send_user_turn("1", lambda _event: None)
    session.rewind(0, lambda _event: None)
    session.modify(0, {20: 0, 25: 1}, lambda _event: None)

    allowed = set(original_ids)
    assert session.ledger()
    assert all(set(entry.features) <= allowed for entry in session.ledger())
