"""Continuous-stream prefill and decode driver for the chat TUI."""

from __future__ import annotations

from dataclasses import fields
from itertools import pairwise
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
import scripts.chat_tui.session as session_module
from nanochat.flash_attention import FlashAttnImpl
from scripts.chat_tui.session import (
    ChatSession,
    ChatSessionConfig,
    GenerationEvent,
    GenerationEventKind,
    SegmentKind,
    SegmentProvenance,
)
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
)
from steering.eval.axbench_upstream.methods import PreparedPromptContext
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import (
    SegmentMode,
    align_segments_to_decoded_prefix,
    token_to_char_offsets,
)
from tests.conftest import MockTokenizer
from tests.encdec_test_helpers import N_SAE_FEATURES, build_tiny_model

_CPU = t.device("cpu")
_ALIGNMENT_COUNTER = "segmentation.char_end_extended.chat_tui"


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA because flex_attention's compile path fails on CPU."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


class StubExtractionService:
    """Build deterministic real-feature segments from actual prompt offsets."""

    def __init__(self, tokenizer: MockTokenizer) -> None:
        self.tokenizer = tokenizer
        self.calls: list[tuple[int, ...]] = []
        self.returned: list[PreparedPromptContext] = []

    def prepare(self, token_ids: tuple[int, ...]) -> PreparedPromptContext:
        self.calls.append(token_ids)
        offsets = token_to_char_offsets(self.tokenizer, list(token_ids))
        decoded_end = offsets[-1]
        split_token = max(2, len(token_ids) // 2)
        split_char = offsets[split_token]
        assert 0 < split_char < decoded_end
        segments = (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(0,),
                char_start=0,
                char_end=split_char,
            ),
            ContextFeatureSegment(
                feature_ids=(1,),
                buckets=(0,),
                char_start=split_char,
                char_end=decoded_end,
            ),
        )
        prepared = PreparedPromptContext(
            token_ids=token_ids,
            assistant_end_id=self.tokenizer.encode_special("<assistant_end>"),
            segments=segments,
        )
        self.returned.append(prepared)
        return prepared


def _setup() -> SimpleNamespace:
    positive = SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=4,
        mode=SamplingMode.top_k_percentile,
    )
    negative = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=0,
        max_k=3,
    )
    return SimpleNamespace(
        n_sae_features=int(N_SAE_FEATURES),
        feature_vocab=None,
        output_selection=positive,
        output_negative_selection=negative,
        percentile_table=t.zeros((int(N_SAE_FEATURES), 99), dtype=t.float64),
        feature_mask=None,
        consolidation=None,
        segment_mode=SegmentMode.fixed_tokens,
    )


def _config(
    *,
    segment_size: int = 4,
    max_new_tokens: int = 6,
) -> ChatSessionConfig:
    return ChatSessionConfig(
        self_cond_segment_size=segment_size,
        temperature=0.7,
        top_k=20,
        seed=17,
        max_new_tokens_per_turn=max_new_tokens,
        deterministic_rewind=True,
    )


def _build_session(
    *,
    segment_size: int = 4,
    max_new_tokens: int = 6,
    sequence_len: int = 96,
) -> tuple[ChatSession, MockTokenizer, StubExtractionService]:
    t.manual_seed(0)
    model = build_tiny_model(
        num_buckets=2,
        seq_len=sequence_len,
        feature_pred_config=FeaturePredictionConfig(
            weight=0.1,
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
    )
    model.eval()
    tokenizer = MockTokenizer()
    extraction = StubExtractionService(tokenizer)
    session = ChatSession(
        setup=_setup(),
        model=model,
        tokenizer=tokenizer,
        device=_CPU,
        config=_config(
            segment_size=segment_size,
            max_new_tokens=max_new_tokens,
        ),
        extraction=extraction,
    )
    return session, tokenizer, extraction


def _script_samples(
    monkeypatch: pytest.MonkeyPatch,
    token_ids: list[int],
) -> list[int]:
    remaining = list(token_ids)
    observed: list[int] = []

    def scripted(logits, rng, temperature, top_k):
        del rng, temperature, top_k
        if not remaining:
            raise AssertionError("sample script exhausted")
        token_id = remaining.pop(0)
        observed.append(token_id)
        return t.tensor([[token_id]], dtype=t.long, device=logits.device)

    monkeypatch.setattr(session_module, "sample_next_token", scripted)
    return observed


def _collect_events() -> tuple[list[GenerationEvent], Any]:
    events: list[GenerationEvent] = []
    return events, events.append


def test_first_send_prefills_once_in_inference_mode_and_aligns_by_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=2)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [12, stop])
    calls: list[tuple[int, int, bool, bool]] = []
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward

    def forward_spy(wrapper, input_ids, *args, **kwargs):
        kv_cache = kwargs["kv_cache"]
        before = int(kv_cache.get_pos())
        logits = original_forward(wrapper, input_ids, *args, **kwargs)
        calls.append(
            (
                before,
                int(input_ids.shape[1]),
                bool(t.is_inference(logits)),
                bool(logits.requires_grad),
            )
        )
        return logits

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        forward_spy,
    )

    session.send_user_turn("1", lambda _event: None)

    prefill_calls = [call for call in calls if call[0] == 0]
    assert len(prefill_calls) == 1
    assert prefill_calls[0][1] == len(extraction.calls[0])
    assert prefill_calls[0][2:4] == (True, False)
    assert all(before > 0 for before, _width, _inference, _grad in calls[1:])

    aligned = session._aligned_segments
    assert aligned is not None
    realigned = align_segments_to_decoded_prefix(
        tokenizer=tokenizer,
        prefix_tokens=list(extraction.calls[0]),
        segments=aligned,
        counter_name=_ALIGNMENT_COUNTER,
    )
    assert realigned == aligned
    for actual, expected in zip(realigned, aligned, strict=True):
        assert all(
            getattr(actual, field.name) == getattr(expected, field.name)
            for field in fields(ContextFeatureSegment)
        )
    assert session._wrapper is not None
    assert session._wrapper.context_provider._segments == aligned


def test_prefill_ledger_tiles_real_prefix_and_records_s0_features(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop])
    seeded: list[dict[int, int]] = []
    original_seed = SelfConditionedFeatureProvider.seed_from_prompt

    def seed_spy(provider, state, prompt_fp_logits):
        result = original_seed(provider, state, prompt_fp_logits)
        assert provider.last_appended_segment is not None
        seeded.append(dict(provider.last_appended_segment))
        return result

    monkeypatch.setattr(SelfConditionedFeatureProvider, "seed_from_prompt", seed_spy)

    session.send_user_turn("1", lambda _event: None)

    prefill_len = len(extraction.calls[0])
    ledger = session.ledger()
    real = [entry for entry in ledger if entry.kind is SegmentKind.PREFIX_REAL]
    selfcond = [entry for entry in ledger if entry.kind is SegmentKind.SELFCOND]
    assert [entry.encoder_slot_idx for entry in real] == list(range(len(real)))
    assert real[0].token_start == 0
    assert real[-1].token_end == prefill_len - 1
    assert all(
        left.token_end == right.token_start for left, right in pairwise(real)
    )
    assert all(entry.provenance is SegmentProvenance.REAL for entry in real)
    assert selfcond[0].encoder_slot_idx == len(real)
    assert selfcond[0].selfcond_ordinal == 0
    assert selfcond[0].token_start == prefill_len - 1
    assert selfcond[0].features == seeded[0]


def test_k_cadence_and_boundary_snapshots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment_size = 4
    session, _tokenizer, _extraction = _build_session(
        segment_size=segment_size,
        max_new_tokens=9,
    )
    _script_samples(monkeypatch, [10] * 9)

    session.send_user_turn("1", lambda _event: None)

    selfcond = [entry for entry in session.ledger() if entry.kind is SegmentKind.SELFCOND]
    assert len(selfcond) >= 3
    assert all(entry.token_end - entry.token_start == segment_size for entry in selfcond[:-1])
    assert [snapshot.selfcond_ordinal for snapshot in session._snapshots] == list(
        range(len(session._snapshots))
    )
    for snapshot in session._snapshots:
        assert snapshot.kv_pos == snapshot.token_count
        assert snapshot.runtime.fp_buffer == ()
        assert snapshot.runtime.tokens_in_seg == 0
        assert snapshot.next_logits.device.type == "cpu"
        assert snapshot.next_logits.dtype is t.float32
        assert snapshot.next_logits.shape == (1, session._model.dec_config.vocab_size)


def test_forced_sampled_and_close_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=3)
    sampled = [11, 12, 13]
    _script_samples(monkeypatch, sampled)
    events, on_event = _collect_events()

    session.send_user_turn("1", on_event)

    close = tokenizer.encode_special("<assistant_end>")
    assert session.conversation_tokens() == (*extraction.calls[0], *sampled, close)
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == len(session.conversation_tokens())
    assert sum(event.kind is GenerationEventKind.TURN_CLOSED for event in events) == 1


@pytest.mark.parametrize("sampled_stop", [True, False])
def test_sampled_stop_is_not_committed_and_cap_is_exact(
    monkeypatch: pytest.MonkeyPatch,
    sampled_stop: bool,
) -> None:
    cap = 3
    session, tokenizer, extraction = _build_session(max_new_tokens=cap)
    stop = tokenizer.encode_special("<assistant_end>")
    script = [15, stop] if sampled_stop else [15, 16, 17]
    observed = _script_samples(monkeypatch, script)

    session.send_user_turn("1", lambda _event: None)

    suffix = session.conversation_tokens()[len(extraction.calls[0]) :]
    if sampled_stop:
        assert observed == [15, stop]
        assert suffix == (15, stop)
    else:
        assert observed == [15, 16, 17]
        assert suffix == (15, 16, 17, stop)
    assert session.conversation_tokens().count(stop) == 1


def test_stale_cancel_before_send_is_cleared_and_does_not_truncate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cancel() targets the ACTIVE operation only: a request left over from an
    idle period must not silently truncate the next generation."""

    session, tokenizer, extraction = _build_session(max_new_tokens=2)
    stop = tokenizer.encode_special("<assistant_end>")
    observed = _script_samples(monkeypatch, [10, stop])
    session.cancel()
    events, on_event = _collect_events()

    session.send_user_turn("1", on_event)

    assert observed == [10, stop]
    assert session.conversation_tokens()[len(extraction.calls[0]) :] == (10, stop)
    assert events[-1].kind is GenerationEventKind.TURN_CLOSED
    assert session._cancel_requested is False


def test_stale_cancel_before_rewind_is_cleared_and_does_not_truncate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stale cancel() must not break the rewind's auto-continue at the loop top
    (it previously regenerated only the forced close token)."""

    session, tokenizer, _extraction = _build_session(max_new_tokens=4)
    stop = tokenizer.encode_special("<assistant_end>")
    observed = _script_samples(monkeypatch, [10, 11, 12, stop, 20, stop])
    session.send_user_turn("1", lambda _event: None)
    prompt_len = session._snapshots[0].token_count

    session.cancel()
    session.rewind(0, lambda _event: None)

    assert session._cancel_requested is False
    assert observed == [10, 11, 12, stop, 20, stop]
    assert session.conversation_tokens()[prompt_len:] == (20, stop)


def test_job_cancel_check_true_closes_without_sampling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=2)
    stop = tokenizer.encode_special("<assistant_end>")
    observed = _script_samples(monkeypatch, [10])
    events, on_event = _collect_events()

    session.send_user_turn("1", on_event, cancel_check=lambda: True)

    assert observed == []
    assert session.conversation_tokens()[len(extraction.calls[0]) :] == (stop,)
    assert events[-1].kind is GenerationEventKind.TURN_CLOSED


def test_cancel_from_token_callback_closes_at_loop_top(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=4)
    stop = tokenizer.encode_special("<assistant_end>")
    observed = _script_samples(monkeypatch, [10, 11])
    events: list[GenerationEvent] = []

    def on_event(event: GenerationEvent) -> None:
        events.append(event)
        if event.kind is GenerationEventKind.TOKEN and event.detail == "10":
            session.cancel()

    session.send_user_turn("1", on_event)

    assert observed == [10]
    assert session.conversation_tokens()[len(extraction.calls[0]) :] == (10, stop)
    assert session.turn_state().value == "between"
    assert session._kv_cache is not None
    assert session._kv_cache.get_pos() == len(session.conversation_tokens())
    assert events[-1].kind is GenerationEventKind.TURN_CLOSED
    assert session._cancel_requested is False


def test_forced_close_can_complete_window_and_consume_selection_rng(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    segment_size = 4
    session, _tokenizer, _extraction = _build_session(
        segment_size=segment_size,
        max_new_tokens=segment_size - 1,
    )
    _script_samples(monkeypatch, [20, 21, 22])
    events, on_event = _collect_events()

    session.send_user_turn("1", on_event)

    assert len(session._snapshots) == 2
    assert not t.equal(
        session._snapshots[0].selection_rng_state,
        session._snapshots[1].selection_rng_state,
    )
    assert any(
        event.kind is GenerationEventKind.SEGMENT_APPENDED and event.detail == "1"
        for event in events
    )
    selfcond = [entry for entry in session.ledger() if entry.kind is SegmentKind.SELFCOND]
    assert [entry.selfcond_ordinal for entry in selfcond] == [0, 1]


def test_second_send_continues_stream_without_second_prefill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(segment_size=3, max_new_tokens=2)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop, stop])
    prefill_count = 0
    original_forward = session_module.TargetAwareEncoderDecoderEvalWrapper.forward

    def forward_spy(wrapper, input_ids, *args, **kwargs):
        nonlocal prefill_count
        if kwargs["kv_cache"].get_pos() == 0:
            prefill_count += 1
        return original_forward(wrapper, input_ids, *args, **kwargs)

    monkeypatch.setattr(
        session_module.TargetAwareEncoderDecoderEvalWrapper,
        "forward",
        forward_spy,
    )

    session.send_user_turn("1", lambda _event: None)
    before_second = session.conversation_tokens()
    session.send_user_turn("2", lambda _event: None)

    forced = (
        tokenizer.encode_special("<user_start>"),
        *tokenizer.encode("2"),
        tokenizer.encode_special("<user_end>"),
        tokenizer.encode_special("<assistant_start>"),
    )
    assert session.conversation_tokens()[len(before_second) : len(before_second) + 4] == forced
    assert prefill_count == 1
    assert len(extraction.calls) == 1
    ordinals = [
        entry.selfcond_ordinal for entry in session.ledger() if entry.kind is SegmentKind.SELFCOND
    ]
    assert ordinals == list(range(len(ordinals)))


def test_ledger_live_tail_ends_at_current_token_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=2)
    _script_samples(monkeypatch, [14, tokenizer.encode_special("<assistant_end>")])

    session.send_user_turn("1", lambda _event: None)

    live = session.ledger()[-1]
    assert live.kind is SegmentKind.SELFCOND
    assert live.token_end == len(session.conversation_tokens())


def test_token_events_reassemble_post_prefill_transcript_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=3)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [30, 31, stop])
    events, on_event = _collect_events()

    session.send_user_turn("1", on_event)

    token_text = "".join(
        event.text_delta.appended
        for event in events
        if event.kind is GenerationEventKind.TOKEN and event.text_delta is not None
    )
    prefill_text = tokenizer.decode(list(extraction.calls[0]))
    assert session._conversation.text == prefill_text + token_text


def test_first_send_emits_authoritative_prefill_transcript_before_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=1)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    events, on_event = _collect_events()

    session.send_user_turn("1", on_event)

    transcript_index = next(
        index for index, event in enumerate(events) if event.kind is GenerationEventKind.TRANSCRIPT
    )
    first_token_index = next(
        index for index, event in enumerate(events) if event.kind is GenerationEventKind.TOKEN
    )
    assert transcript_index < first_token_index
    assert events[transcript_index].detail == tokenizer.decode(list(extraction.calls[0]))


def test_tight_second_turn_fails_preflight_without_any_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(
        max_new_tokens=1,
        sequence_len=10,
    )
    _script_samples(monkeypatch, [20])
    session.send_user_turn("1", lambda _event: None)
    assert session._kv_cache is not None
    tokens_before = session.conversation_tokens()
    kv_pos_before = session._kv_cache.get_pos()

    with pytest.raises(ValueError, match=r"requires .* sequence_len is 10"):
        session.send_user_turn("1", lambda _event: None)

    assert session.conversation_tokens() == tokens_before
    assert session._kv_cache.get_pos() == kv_pos_before


def test_load_prefix_and_clear_discard_runtime_and_restart_rng_story(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, _extraction = _build_session(max_new_tokens=1)
    _script_samples(monkeypatch, [tokenizer.encode_special("<assistant_end>")])
    initial_token_rng = session._token_rng.get_state().clone()
    initial_selection_rng = session._selection_rng.get_state().clone()
    session.send_user_turn("1", lambda _event: None)
    assert session._wrapper is not None

    session.clear()

    assert session._wrapper is None
    assert session._kv_cache is None
    assert session.ledger() == ()
    assert session._snapshots == []
    assert t.equal(session._token_rng.get_state(), initial_token_rng)
    assert t.equal(session._selection_rng.get_state(), initial_selection_rng)
    assert session.conversation_tokens() == (tokenizer.get_bos_token_id(),)


def test_temperature_must_be_non_negative() -> None:
    with pytest.raises(ValueError, match="temperature"):
        ChatSessionConfig(
            self_cond_segment_size=4,
            temperature=-0.01,
            top_k=20,
            seed=17,
            max_new_tokens_per_turn=2,
            deterministic_rewind=True,
        )


def test_prefill_rejects_extraction_assistant_end_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, _tokenizer, extraction = _build_session(max_new_tokens=1)
    original_prepare = extraction.prepare

    def wrong_stop(token_ids: tuple[int, ...]) -> PreparedPromptContext:
        prepared = original_prepare(token_ids)
        return PreparedPromptContext(
            token_ids=prepared.token_ids,
            assistant_end_id=prepared.assistant_end_id + 1,
            segments=prepared.segments,
        )

    monkeypatch.setattr(extraction, "prepare", wrong_stop)
    with pytest.raises(ValueError, match=r"assistant_end_id.*does not match.*assistant_end_id"):
        session.send_user_turn("1", lambda _event: None)


def test_trailing_real_segment_owned_only_by_final_prompt_token_is_empty_span(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session, tokenizer, extraction = _build_session(max_new_tokens=1)
    stop = tokenizer.encode_special("<assistant_end>")
    _script_samples(monkeypatch, [stop])

    def trailing_final(token_ids: tuple[int, ...]) -> PreparedPromptContext:
        extraction.calls.append(token_ids)
        offsets = token_to_char_offsets(tokenizer, list(token_ids))
        final_start = offsets[-2]
        return PreparedPromptContext(
            token_ids=token_ids,
            assistant_end_id=stop,
            segments=(
                ContextFeatureSegment((0,), (0,), 0, final_start),
                ContextFeatureSegment((1,), (0,), final_start, offsets[-1]),
            ),
        )

    monkeypatch.setattr(extraction, "prepare", trailing_final)
    session.send_user_turn("1", lambda _event: None)

    prefix = [entry for entry in session.ledger() if entry.kind is SegmentKind.PREFIX_REAL]
    assert prefix[-1].token_start == prefix[-1].token_end == session._prefill_len - 1
    assert prefix[-1].provenance is SegmentProvenance.REAL


def test_middle_empty_real_segment_still_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    session, _tokenizer, _extraction = _build_session(max_new_tokens=1)
    prompt = (1, 2, 3, 4)
    aligned = tuple(
        ContextFeatureSegment((feature_id,), (0,), feature_id, feature_id + 1)
        for feature_id in range(3)
    )
    monkeypatch.setattr(
        session_module,
        "compute_context_token_segment_assignments",
        lambda **_kwargs: t.tensor([0, 0, 2, 2]),
    )

    with pytest.raises(RuntimeError, match="before a later non-empty slot"):
        session._append_prefix_ledger(prompt, aligned)


def test_session_config_builder_strips_delta_weighted_output_selection() -> None:
    """Pretrain-shaped setups carry delta_weights=percentile_delta on their
    output selection; the builder must strip it exactly like axbench's
    build_self_conditioning_config, or SelfCondConfig construction fails.
    Regression for the GPU parity failure on the pretrain target."""
    from types import SimpleNamespace

    from scripts.chat_tui.session import build_session_self_cond_config
    from scripts.chat_tui.steering import (
        SteeringMethod,
        SteeringSelection,
        build_score_policy,
    )
    from steering.conditional.delta import DeltaSamplingWeights
    from steering.conditional.sampling import (
        Distribution,
        NegativeSamplingSelection,
        SamplingMode,
        SamplingSelection,
    )

    delta_selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=12.0,
        std=12.0,
        min_k=1,
        max_k=49,
        mode=SamplingMode.top_k_percentile,
        delta_weights=DeltaSamplingWeights.percentile_delta,
    )
    negative = NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=12.0,
        std=12.0,
        min_k=0,
        max_k=49,
    )
    setup = SimpleNamespace(
        output_selection=delta_selection,
        output_negative_selection=negative,
        percentile_table=t.zeros((8, 99), dtype=t.float64),
        n_sae_features=8,
        feature_mask=None,
        consolidation=None,
        feature_vocab=None,
    )
    config = ChatSessionConfig(
        self_cond_segment_size=4,
        temperature=1.0,
        top_k=50,
        seed=42,
        max_new_tokens_per_turn=8,
        deterministic_rewind=False,
    )
    policy = build_score_policy(
        SteeringSelection(SteeringMethod.OFF, ()),
        positive_selection=delta_selection,
        negative_selection=negative,
        head_width=8,
    )
    rng = t.Generator(device="cpu")
    rng.manual_seed(43)

    built = build_session_self_cond_config(
        setup, config, rng, policy, segment_mode=SegmentMode.fixed_tokens
    )
    assert built.selection.selection.delta_weights is None
