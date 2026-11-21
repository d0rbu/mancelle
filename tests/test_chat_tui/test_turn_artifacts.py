"""Opt-in real post-training checkpoint + Gemma/SAE qualification for #1627.

Set CHAT_TURN_META and CHAT_TURN_STATS to existing matching artifacts, plus
CHAT_TURN_PRETRAIN_META when the leaf needs its exact pretraining metadata. Missing
variables skip; explicitly supplied wrong/missing artifacts FAIL. No metadata
shim, training job, paid inference call or artifact upload is performed by this test.
"""

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t

import nanochat.flash_attention as flash_attention
import scripts.chat_tui.turn_session as turn_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.sampling import sample_next_token
from scripts.chat_tui.extraction import TurnGemmaExtractionService
from scripts.chat_tui.session import ChatSessionConfig, SegmentKind
from scripts.chat_tui.setup import load_chat_setup, load_chat_target_model
from scripts.chat_tui.turn_session import TurnChatSession, _render_prompt
from steering.eval import inference_engine
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import SingleEntryContextProvider
from tests.test_chat_tui.test_turn_session import (
    TurnExtraction,
    assert_unchanged,
    build_turn,
    owned_state,
)

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA"),
]


def _tensor_identity(value):
    raw = value.detach().cpu().contiguous().view(t.uint8).numpy().tobytes()
    return str(value.dtype), tuple(value.shape), hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("force_full_budget", [False, True])
def test_real_turn_checkpoint_multi_epoch_parity_and_failure_retry(
    monkeypatch, record_property, force_full_budget
):
    meta = os.environ.get("CHAT_TURN_META")
    stats = os.environ.get("CHAT_TURN_STATS")
    if meta is None or stats is None:
        pytest.skip("set both CHAT_TURN_META and CHAT_TURN_STATS for real artifact acceptance")
    meta_path, stats_path = Path(meta), Path(stats)
    pretrain = os.environ.get("CHAT_TURN_PRETRAIN_META")
    setup = load_chat_setup(
        meta_path, stats_path, pretrain_meta_path=None if pretrain is None else Path(pretrain)
    )
    model, tokenizer = load_chat_target_model(setup, "cuda")
    cadence = setup.turn_self_cond_segment_size_rustbpe
    assert type(cadence) is int and cadence > 0
    config = ChatSessionConfig(
        self_cond_segment_size=cadence,
        max_new_tokens_per_turn=2 * cadence + 3,
        temperature=0.7,
        top_k=20,
        seed=42,
        deterministic_rewind=True,
    )
    extraction = TurnGemmaExtractionService(setup, tokenizer, "cuda", stats_path)

    def new_session():
        return TurnChatSession(
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            device="cuda",
            config=config,
            extraction=extraction,
        )

    for label, path in (
        ("metadata", meta_path),
        ("statistics", stats_path),
        ("weights", meta_path.with_name(f"model_{int(setup.meta['step']):06d}.pt")),
    ):
        with path.open("rb") as stream:
            record_property(f"{label}_sha256", hashlib.file_digest(stream, "sha256").hexdigest())
    record_property("turn_cadence_rustbpe", cadence)
    if pretrain is not None:
        with Path(pretrain).open("rb") as stream:
            record_property(
                "pretrain_metadata_sha256", hashlib.file_digest(stream, "sha256").hexdigest()
            )

    session = new_session()
    prompt_text = (
        "Explain why scientific measurements need a control group. "
        "Write a detailed explanation of at least 500 words with several concrete examples."
    )
    prompt = _render_prompt(tokenizer, [{"role": "user", "content": prompt_text}])
    prepared = extraction.prepare(prompt)
    driver = session._driver
    oracle_rng = t.Generator(device="cpu").set_state(driver._selection_rng.get_state())
    oracle_config = driver._build_self_cond_config(driver._score_policy)
    oracle_config = replace(
        oracle_config, selection=replace(oracle_config.selection, rng=oracle_rng)
    )
    oracle = build_engine_self_conditioned(
        model,
        tokenizer,
        t.device("cuda"),
        oracle_config,
        context_provider=SingleEntryContextProvider(
            prompt,
            prepared.segments,
            segment_mode=prepared.segment_mode,
            token_assignments=prepared.token_assignments,
        ),
    )
    record_property(
        "sampling_policy", "non_structural_boundary_probe" if force_full_budget else "natural"
    )
    if force_full_budget:
        # Some real SFT pilots stop before K. This distinct controlled probe
        # exercises two actual refreshes; it is NOT a natural-response claim.
        # Only the sampled stop logits are masked, identically on both paths.
        # LM/FP model outputs, routing, feature selection and RNG draws stay real.
        stop_ids = sorted(oracle._model.generation_stop_token_ids)

        def sample_ordinary(logits, rng, temperature, top_k):
            logits = logits.clone()
            # Tiny fixtures can name structural IDs outside their toy vocab.
            present_stops = [token for token in stop_ids if 0 <= token < logits.shape[-1]]
            logits[..., present_stops] = t.finfo(logits.dtype).min
            return sample_next_token(logits, rng, temperature, top_k)

        monkeypatch.setattr(turn_module, "sample_next_token", sample_ordinary)
        monkeypatch.setattr(inference_engine, "sample_next_token", sample_ordinary)
    calls = []
    original_forward = model.forward

    def observe(input_ids, *args, **kwargs):
        result = original_forward(input_ids, *args, **kwargs)
        calls.append(
            (
                _tensor_identity(input_ids),
                _tensor_identity(kwargs["seg_map"]),
                tuple(_tensor_identity(value) for value in result),
            )
        )
        return result

    monkeypatch.setattr(model, "forward", observe)
    expected = tuple(
        oracle.generate_stream(
            list(prompt),
            max_new_tokens=config.max_new_tokens_per_turn,
            temperature=config.temperature,
            top_k=config.top_k,
            seed=config.seed,
        )
    )
    oracle_calls = list(calls)
    calls.clear()
    session.send_user_turn(prompt_text, lambda _: None)
    record_property("first_turn_messages", json.dumps(session._driver.messages))
    record_property(
        "first_turn_predicted_segments",
        sum(row.kind is SegmentKind.SELFCOND for row in session.ledger()),
    )
    if force_full_budget:
        assert len(expected) == config.max_new_tokens_per_turn
        assert sum(row.kind is SegmentKind.SELFCOND for row in session.ledger()) >= 3
    assert session.conversation_tokens()[len(prompt) : -1] == expected
    assert calls == oracle_calls  # exact input, routing, LM and FP bytes
    runtime = session._driver._wrapper.self_cond_runtime_snapshot()
    reference = oracle._model.self_cond_runtime_snapshot()
    assert runtime.tokens_in_seg == reference.tokens_in_seg
    assert runtime.state.target_slot_idx == reference.state.target_slot_idx
    assert t.equal(runtime.state.encoder_out.values, reference.state.encoder_out.values)
    assert t.equal(session._driver._selection_rng.get_state(), oracle_rng.get_state())
    assert len(runtime.fp_buffer) == len(reference.fp_buffer)
    for actual, expected_row in zip(runtime.fp_buffer, reference.fp_buffer, strict=True):
        assert t.equal(actual, expected_row)
    calls.clear()
    monkeypatch.setattr(model, "forward", original_forward)

    first = session.conversation_tokens()
    for ordinal in (0, 1) if force_full_budget else (0,):
        session.rewind(ordinal, lambda _: None)
        assert session.conversation_tokens() == first
        row = next(row for row in session.ledger() if row.selfcond_ordinal == ordinal)
        session.modify(ordinal, dict(row.features), lambda _: None)
        assert (
            next(
                row for row in session.ledger() if row.selfcond_ordinal == ordinal
            ).provenance.value
            == "overridden"
        )
        session.rewind(0, lambda _: None)
        assert session.conversation_tokens() == first
    session.rewind(0, lambda _: None)
    assert session.conversation_tokens() == first

    before = owned_state(session)
    for stage in ("extraction", "prefill"):
        with monkeypatch.context() as failure_patch:
            if stage == "extraction":

                def fail_extract(*_args, **_kwargs):
                    raise RuntimeError("injected extraction failure")

                failure_patch.setattr(extraction, "prepare", fail_extract)
            else:

                def fail_prefill(*args, **kwargs):
                    original_forward(*args, **kwargs)
                    raise RuntimeError("injected prefill failure after KV write")

                failure_patch.setattr(model, "forward", fail_prefill)
            with pytest.raises(RuntimeError, match="injected"):
                session.send_user_turn("Give a short example.", lambda _: None)
        assert_unchanged(session, before)

    old = session._driver
    session.send_user_turn("Give a short example.", lambda _: None)
    record_property("second_turn_messages", json.dumps(session._driver.messages))
    assert session._driver._kv_cache is not old._kv_cache
    assert session._driver.messages[:2] == old.messages
    assert sum(row.kind is SegmentKind.PREFIX_REAL for row in session.ledger()) >= 3
    exact_second = _render_prompt(tokenizer, session._driver.prompt_messages)
    assert extraction._cache[exact_second].token_ids == exact_second
    assert {row.epoch_id for row in session.ledger()} == {session.identity[0]}
    control = new_session()
    control.send_user_turn(prompt_text, lambda _: None)
    control.send_user_turn("Give a short example.", lambda _: None)
    assert session.conversation_tokens() == control.conversation_tokens()
    assert t.equal(session._driver._token_rng.get_state(), control._driver._token_rng.get_state())
    assert t.equal(
        session._driver._selection_rng.get_state(), control._driver._selection_rng.get_state()
    )


@pytest.mark.parametrize("force_full_budget", [False, True])
def test_acceptance_harness_on_tiny_cuda_model_not_real_artifact(
    monkeypatch, tmp_path, record_property, force_full_budget
):
    """Exercise the entire acceptance harness without claiming Gemma coverage."""
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    tiny, tokenizer, _ = build_turn(cadence=4, sequence_len=512, device="cuda")
    setup, model = tiny._driver._setup, tiny._driver._model
    setup.meta = {"step": 7}
    meta, stats = tmp_path / "meta_000007.json", tmp_path / "stats.safetensors"
    meta.write_text('{"step": 7}')
    stats.write_bytes(b"harness only, not statistics")
    (tmp_path / "model_000007.pt").write_bytes(b"harness only, not weights")
    monkeypatch.setenv("CHAT_TURN_META", str(meta))
    monkeypatch.setenv("CHAT_TURN_STATS", str(stats))
    monkeypatch.delenv("CHAT_TURN_PRETRAIN_META", raising=False)
    monkeypatch.setattr(__name__ + ".load_chat_setup", lambda *_a, **_k: setup)
    monkeypatch.setattr(__name__ + ".load_chat_target_model", lambda *_: (model, tokenizer))

    class HarnessExtraction(TurnExtraction):
        def __init__(self, _setup, tokenizer, _device, _stats):
            super().__init__(tokenizer)
            self._cache = {}

        def prepare(self, token_ids):
            prepared = super().prepare(token_ids)
            self._cache[token_ids] = prepared
            return prepared

    monkeypatch.setattr(__name__ + ".TurnGemmaExtractionService", HarnessExtraction)
    test_real_turn_checkpoint_multi_epoch_parity_and_failure_retry(
        monkeypatch, record_property, force_full_budget
    )
