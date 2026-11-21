"""Multi-boundary exact parity with the public inference engine."""

from __future__ import annotations

import dataclasses

import pytest
import torch as t

from scripts.chat_tui.session import (
    ChatSession,
    ChatSessionConfig,
    SegmentKind,
    build_session_self_cond_config,
)
from scripts.chat_tui.steering import (
    SteeringMethod,
    SteeringSelection,
    build_score_policy,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.conditional.sampling import SamplingBundle
from steering.eval.axbench_upstream.methods import build_self_conditioning_config
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.inference_engine import build_engine_self_conditioned
from steering.eval.target_conditioning import SingleEntryContextProvider

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")

PROMPT = (
    "Write a detailed, continuous explanation of how a seed grows into a mature tree. "
    "Use at least 150 words and do not conclude early."
)


def _first_divergence(left: list[int], right: list[int]) -> int:
    return next(
        (
            index
            for index, (left_id, right_id) in enumerate(zip(left, right, strict=False))
            if left_id != right_id
        ),
        min(len(left), len(right)),
    )


def test_small_k_session_matches_engine_across_boundaries(
    leaf_setup,
    leaf_model_tokenizer,
    extraction_stack,
) -> None:
    model, tokenizer = leaf_model_tokenizer
    config = ChatSessionConfig(
        self_cond_segment_size=32,
        temperature=1.0,
        top_k=50,
        seed=42,
        max_new_tokens_per_turn=96,
        deterministic_rewind=False,
    )
    session = ChatSession(
        setup=leaf_setup,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        config=config,
        extraction=extraction_stack,
    )
    session.send_user_turn(PROMPT, lambda _event: None)

    rendered = render_prompt_for_chat_completion(tokenizer, PROMPT)
    prompt = tuple(int(token_id) for token_id in rendered.token_ids)
    assert session.conversation_tokens()[: len(prompt)] == prompt
    prepared = extraction_stack.prepare(prompt)
    assert prepared.token_ids == prompt
    session_tokens = list(session.conversation_tokens()[len(prompt) : -1])
    selfcond = [entry for entry in session.ledger() if entry.kind is SegmentKind.SELFCOND]
    assert len(selfcond) >= 3, "fixed real-model case did not cross two K=32 boundaries"

    head_width = (
        int(leaf_setup.feature_vocab.size)
        if leaf_setup.feature_vocab is not None
        else int(leaf_setup.n_sae_features)
    )
    policy = build_score_policy(
        SteeringSelection(SteeringMethod.OFF, ()),
        positive_selection=leaf_setup.output_selection,
        negative_selection=leaf_setup.output_negative_selection,
        head_width=head_width,
        feature_mask=leaf_setup.feature_mask,
        head_row_original_ids=(
            None
            if leaf_setup.feature_vocab is None
            else leaf_setup.feature_vocab.new_to_old_tensor()
        ),
    )
    selection_rng = t.Generator(device="cpu").manual_seed(43)
    engine_config = build_session_self_cond_config(
        leaf_setup,
        config,
        selection_rng,
        policy,
        segment_mode=leaf_setup.segment_mode,
    )
    engine = build_engine_self_conditioned(
        model,
        tokenizer,
        t.device("cuda"),
        engine_config,
        context_provider=SingleEntryContextProvider(prompt, prepared.segments),
    )
    engine_tokens = list(
        engine.generate_stream(
            list(prompt),
            max_new_tokens=96,
            temperature=1.0,
            top_k=50,
            seed=42,
            stop_token_ids={
                int(tokenizer.get_bos_token_id()),
                int(tokenizer.encode_special("<assistant_end>")),
            },
        )
    )

    divergence = _first_divergence(session_tokens, engine_tokens)
    assert session_tokens == engine_tokens, (
        f"small-K real-model streams diverged at index {divergence}: "
        f"session={session_tokens}, engine={engine_tokens}"
    )


def test_session_config_builder_matches_axbench_builder_except_segment_size(
    leaf_setup,
    leaf_model_tokenizer,
) -> None:
    """The stream parity above builds both sides' configs with the session's
    own builder, so its fields need an independent oracle: axbench's
    ``build_self_conditioning_config`` (which cannot express K != sequence_len)
    must agree on every field except the intended segment-size difference."""

    model, _tokenizer = leaf_model_tokenizer
    head_width = (
        int(leaf_setup.feature_vocab.size)
        if leaf_setup.feature_vocab is not None
        else int(leaf_setup.n_sae_features)
    )
    policy = build_score_policy(
        SteeringSelection(SteeringMethod.OFF, ()),
        positive_selection=leaf_setup.output_selection,
        negative_selection=leaf_setup.output_negative_selection,
        head_width=head_width,
        feature_mask=leaf_setup.feature_mask,
        head_row_original_ids=(
            None
            if leaf_setup.feature_vocab is None
            else leaf_setup.feature_vocab.new_to_old_tensor()
        ),
    )
    config = ChatSessionConfig(
        self_cond_segment_size=32,
        temperature=1.0,
        top_k=50,
        seed=42,
        max_new_tokens_per_turn=96,
        deterministic_rewind=False,
    )
    session_rng = t.Generator(device="cpu").manual_seed(43)
    session_config = build_session_self_cond_config(
        leaf_setup, config, session_rng, policy, segment_mode=leaf_setup.segment_mode
    )
    axbench_config = build_self_conditioning_config(leaf_setup, model, policy, selection_seed=43)

    assert int(session_config.self_cond_segment_size) == 32
    assert int(axbench_config.self_cond_segment_size) == int(model.dec_config.sequence_len)
    assert isinstance(session_config.selection, SamplingBundle)
    assert isinstance(axbench_config.selection, SamplingBundle)
    assert session_config.selection.selection == axbench_config.selection.selection
    assert session_config.selection.percentile_table is axbench_config.selection.percentile_table
    assert t.equal(
        session_config.selection.rng.get_state(),
        axbench_config.selection.rng.get_state(),
    )
    # Sweep every other dataclass field so a future one-sided addition (e.g. a
    # logit bias) cannot silently evade this oracle.
    handled = {"self_cond_segment_size", "selection"}
    for config_field in dataclasses.fields(SelfCondConfig):
        if config_field.name in handled:
            continue
        session_value = getattr(session_config, config_field.name)
        axbench_value = getattr(axbench_config, config_field.name)
        if session_value is not axbench_value:
            assert session_value == axbench_value, (
                f"builders disagree on {config_field.name}: {session_value!r} != {axbench_value!r}"
            )
