"""Full-size leaf multi-turn session acceptance battery."""

from __future__ import annotations

import pytest
import torch as t

from scripts.chat_tui.convo import (
    ConvoCatalog,
    prefix_headroom_error,
    prefix_messages,
)
from scripts.chat_tui.session import (
    ChatSession,
    ChatSessionConfig,
    SegmentKind,
    SegmentProvenance,
    TurnState,
)
from scripts.chat_tui.steering import (
    SignalSource,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")


def _assert_budget(session: ChatSession) -> None:
    used, capacity = session.token_budget()
    assert used == len(session.conversation_tokens())
    assert 0 < used <= capacity


def _selfcond(session: ChatSession):
    return [entry for entry in session.ledger() if entry.kind is SegmentKind.SELFCOND]


def _turn_state_at(tokenizer, tokens: tuple[int, ...]) -> TurnState:
    state = TurnState.BETWEEN
    markers = {
        tokenizer.encode_special("<user_start>"): TurnState.IN_USER,
        tokenizer.encode_special("<user_end>"): TurnState.BETWEEN,
        tokenizer.encode_special("<assistant_start>"): TurnState.IN_ASSISTANT,
        tokenizer.encode_special("<assistant_end>"): TurnState.BETWEEN,
    }
    for token_id in tokens:
        state = markers.get(int(token_id), state)
    return state


def _first_kept_multiturn(catalog, tokenizer, sequence_len: int):
    for index in range(len(catalog)):
        entry = catalog.entry(index)
        if entry.skip_reason is not None or entry.n_user_turns < 2:
            continue
        if (
            prefix_headroom_error(
                entry,
                1,
                tokenizer,
                sequence_len=sequence_len,
                max_new_tokens_per_turn=300,
            )
            is None
        ):
            return entry
    pytest.skip("real standard-SFT validation mixture has no usable multi-turn entry")


def test_real_leaf_session_battery(
    leaf_setup,
    leaf_model_tokenizer,
    extraction_stack,
    leaf_diffmean: t.Tensor,
    leaf_second_diffmean: t.Tensor,
) -> None:
    model, tokenizer = leaf_model_tokenizer
    try:
        catalog = ConvoCatalog.load(tokenizer)
    except Exception as exc:
        pytest.skip(f"real standard-SFT validation mixture is unavailable: {exc}")
    entry = _first_kept_multiturn(catalog, tokenizer, int(model.dec_config.sequence_len))
    prefix = prefix_messages(entry, 1)

    session = ChatSession(
        setup=leaf_setup,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        config=ChatSessionConfig(
            self_cond_segment_size=256,
            temperature=1.0,
            top_k=50,
            seed=42,
            max_new_tokens_per_turn=300,
            deterministic_rewind=True,
        ),
        extraction=extraction_stack,
    )
    session.load_prefix(prefix)
    _assert_budget(session)

    turn_token_deltas: list[int] = []
    turn_text_lengths: list[int] = []
    before = len(session.conversation_tokens())
    session.send_user_turn(
        "Continue this conversation with a careful and useful response.",
        lambda _event: None,
    )
    after = session.conversation_tokens()
    turn_token_deltas.append(len(after) - before)
    turn_text_lengths.append(len(tokenizer.decode(list(after[before:]))))
    _assert_budget(session)

    signal_one = SteeringSignal(
        name="measurement",
        target_scores=leaf_diffmean,
        source=SignalSource.AXBENCH_CONCEPT,
        identity="battery-measurement",
    )
    session.set_steering(
        SteeringSelection(
            SteeringMethod.INTERPOLATION,
            ((signal_one, 3.0),),
        )
    )
    ordinal_before = max(entry.selfcond_ordinal or 0 for entry in _selfcond(session))
    before = len(session.conversation_tokens())
    session.send_user_turn(
        "Give a detailed response of roughly 280 words about reproducible experiments, "
        "including controls, calibration, uncertainty, and independent replication.",
        lambda _event: None,
    )
    after = session.conversation_tokens()
    turn_token_deltas.append(len(after) - before)
    turn_text_lengths.append(len(tokenizer.decode(list(after[before:]))))
    _assert_budget(session)
    newly_steered = [
        ledger_entry
        for ledger_entry in _selfcond(session)
        if ledger_entry.selfcond_ordinal is not None
        and ledger_entry.selfcond_ordinal > ordinal_before
    ]
    assert newly_steered, "steered real turn did not reach a K=256 boundary"
    assert all(entry.provenance is SegmentProvenance.STEERED for entry in newly_steered)
    assert all(
        entry.steering is not None
        and entry.steering.method is SteeringMethod.INTERPOLATION
        and entry.steering.signal_names == ("measurement",)
        and entry.steering.total_strength == 3.0
        for entry in newly_steered
    )

    before_rewind = session.conversation_tokens()
    rewind_entry = next(
        (
            ledger_entry
            for ledger_entry in reversed(_selfcond(session))
            if ledger_entry.selfcond_ordinal is not None
            and ledger_entry.selfcond_ordinal >= 1
            and _turn_state_at(
                tokenizer,
                before_rewind[: ledger_entry.token_start + 1],
            )
            is TurnState.IN_ASSISTANT
        ),
        None,
    )
    assert rewind_entry is not None, "no assistant-open K=256 boundary was available"
    rewind_ordinal = rewind_entry.selfcond_ordinal
    assert rewind_ordinal is not None
    suffix_start = rewind_entry.token_start + 1
    expected_regenerated = before_rewind[suffix_start:-1]
    assert expected_regenerated, (
        "rewind bit-exactness would be vacuous: the selected boundary has no "
        "sampled suffix to regenerate"
    )
    session.rewind(rewind_ordinal, lambda _event: None)
    regenerated = session.conversation_tokens()
    assert regenerated[suffix_start : suffix_start + len(expected_regenerated)] == (
        expected_regenerated
    )
    _assert_budget(session)

    assert leaf_setup.allowed_ids is not None
    representatives = [int(value) for value in leaf_setup.allowed_ids[:2].tolist()]
    override = {representatives[0]: 0, representatives[1]: 1}
    session.modify(1, override, lambda _event: None)
    edited = next(entry for entry in _selfcond(session) if entry.selfcond_ordinal == 1)
    assert edited.provenance is SegmentProvenance.OVERRIDDEN
    assert edited.features == override
    _assert_budget(session)

    signal_two = SteeringSignal(
        name="measurement-control",
        target_scores=leaf_second_diffmean,
        source=SignalSource.AXBENCH_CONCEPT,
        identity="battery-measurement-control",
    )
    session.set_steering(
        SteeringSelection(
            SteeringMethod.RESTRICTION,
            ((signal_one, 0.0), (signal_two, 0.0)),
        )
    )
    ordinal_before = max(entry.selfcond_ordinal or 0 for entry in _selfcond(session))
    before = len(session.conversation_tokens())
    session.send_user_turn(
        "Now provide one more detailed, structured explanation with enough depth to cover "
        "data collection, analysis, failure modes, and reporting.",
        lambda _event: None,
    )
    after = session.conversation_tokens()
    turn_token_deltas.append(len(after) - before)
    turn_text_lengths.append(len(tokenizer.decode(list(after[before:]))))
    _assert_budget(session)
    restriction_entries = [
        ledger_entry
        for ledger_entry in _selfcond(session)
        if ledger_entry.selfcond_ordinal is not None
        and ledger_entry.selfcond_ordinal > ordinal_before
    ]
    assert restriction_entries, "restricted real turn did not cross a new K=256 boundary"
    assert any(
        entry.provenance is SegmentProvenance.STEERED
        and entry.steering is not None
        and entry.steering.method is SteeringMethod.RESTRICTION
        and entry.steering.signal_names == ("measurement", "measurement-control")
        for entry in restriction_entries
    )

    print(
        "chat-tui GPU battery "
        f"catalog_index={entry.index} prefix_messages={len(prefix)} "
        f"turn_token_deltas={turn_token_deltas} decoded_text_lengths={turn_text_lengths} "
        f"final_tokens={len(session.conversation_tokens())}"
    )
    session.clear()
    assert session.ledger() == ()
    assert session.conversation_tokens() == tuple(
        tokenizer.render_conversation({"messages": prefix}, max_tokens=2**31)[0]
    )
    _assert_budget(session)
