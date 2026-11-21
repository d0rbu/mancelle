"""Token-exact ChatSession parity with the real AxBench generation adapter."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

from scripts.chat_tui.session import ChatSession, ChatSessionConfig
from scripts.chat_tui.setup import load_chat_setup
from scripts.chat_tui.steering import (
    SignalSource,
    SteeringMethod,
    SteeringSelection,
    SteeringSignal,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
)
from steering.eval.axbench_upstream.methods import (
    TEMPERATURE,
    TOP_K,
    build_self_conditioning_masked_policy,
    generate_self_conditioning,
    prepare_prompt_context,
    self_conditioning_weight,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion
from tests.test_chat_tui.gpu.conftest import LEAF_META, PRETRAIN_META

pytestmark = pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")


class _DecodeRecorder:
    """Delegate a tokenizer while retaining the exact IDs passed to decode."""

    def __init__(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self.decoded: list[tuple[int, ...]] = []

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tokenizer, name)

    def decode(self, token_ids) -> str:
        self.decoded.append(tuple(int(token_id) for token_id in token_ids))
        return self._tokenizer.decode(token_ids)


def _first_divergence(left: list[int], right: list[int]) -> int:
    return next(
        (
            index
            for index, (left_id, right_id) in enumerate(zip(left, right, strict=False))
            if left_id != right_id
        ),
        min(len(left), len(right)),
    )


@pytest.mark.parametrize("target", ["pretrain", "leaf"])
@pytest.mark.parametrize(
    ("method", "factor"),
    [
        (SteeringMethod.INTERPOLATION, 0.0),
        (SteeringMethod.INTERPOLATION, 1.0),
        (SteeringMethod.RESTRICTION, 0.0),
        (SteeringMethod.RESTRICTION, 1.0),
    ],
)
def test_session_matches_axbench_generation_exactly(
    request: pytest.FixtureRequest,
    target: str,
    method: SteeringMethod,
    factor: float,
    first_concept_prompt: str,
    extraction_service_factory,
    stats_path,
) -> None:
    meta_path = PRETRAIN_META if target == "pretrain" else LEAF_META
    model, tokenizer = request.getfixturevalue(f"{target}_model_tokenizer")
    diffmean = request.getfixturevalue(f"{target}_diffmean")

    # Spec 3.8: each side gets a fresh setup, so its prompt-selection stream
    # starts at the checkpoint seed independently of every other case.
    oracle_setup = load_chat_setup(meta_path, stats_path)
    session_setup = load_chat_setup(meta_path, stats_path)
    rendered = render_prompt_for_chat_completion(tokenizer, first_concept_prompt)
    exact_prompt = tuple(int(token_id) for token_id in rendered.token_ids)
    base_model, sae = request.getfixturevalue("shared_extraction_weights")
    prepared = prepare_prompt_context(
        oracle_setup,
        base_model,
        sae,
        tokenizer,
        token_ids=exact_prompt,
        assistant_end_id=rendered.assistant_end_id,
    )

    if method is SteeringMethod.INTERPOLATION:
        oracle_policy = SelfCondScoreInterpolation(
            target_scores=diffmean,
            negative_selection=oracle_setup.output_negative_selection,
            weight=self_conditioning_weight(factor),
        )
    else:
        oracle_policy = build_self_conditioning_masked_policy(
            oracle_setup,
            diffmean,
            factor,
        )

    recording_tokenizer = _DecodeRecorder(tokenizer)
    oracle_text = generate_self_conditioning(
        oracle_setup,
        model,
        recording_tokenizer,
        prepared,
        oracle_policy,
        selection_seed=43,
        seed=42,
        max_new_tokens=64,
    )
    oracle_tokens = list(recording_tokenizer.decoded[-1])

    signal = SteeringSignal(
        name="parity",
        target_scores=diffmean,
        source=SignalSource.AXBENCH_CONCEPT,
        identity="parity",
    )
    session = ChatSession(
        setup=session_setup,
        model=model,
        tokenizer=tokenizer,
        device="cuda",
        config=ChatSessionConfig(
            self_cond_segment_size=int(model.dec_config.sequence_len),
            temperature=TEMPERATURE,
            top_k=TOP_K,
            seed=42,
            max_new_tokens_per_turn=64,
            deterministic_rewind=False,
        ),
        extraction=extraction_service_factory(session_setup, tokenizer),
    )
    session.set_steering(SteeringSelection(method, ((signal, factor),)))
    session.send_user_turn(first_concept_prompt, lambda _event: None)

    conversation = session.conversation_tokens()
    assert conversation[: len(exact_prompt)] == exact_prompt
    sampled = list(conversation[len(exact_prompt) : -1])
    assert len(sampled) <= 64
    session_text = tokenizer.decode(sampled)
    divergence = _first_divergence(sampled, oracle_tokens)
    assert session_text == oracle_text and sampled == oracle_tokens, (
        f"{target}/{method.value}/factor={factor} diverged at sampled-token "
        f"index {divergence}: session={sampled}, oracle={oracle_tokens}"
    )


def test_max_weight_interpolation_changes_the_sampled_stream(
    leaf_model_tokenizer,
    leaf_diffmean,
    first_concept_prompt,
    extraction_service_factory,
    stats_path,
) -> None:
    """Guard the parity oracle against double-cancellation: both parity sides
    share the provider dispatch, so token equality alone could pass with
    steering silently inactive. A full-weight interpolation signal must change
    the sampled stream relative to weight zero on the same prompt and seed."""

    model, tokenizer = leaf_model_tokenizer
    s0_features_by_factor: dict[float, dict[int, int]] = {}
    sampled_by_factor: dict[float, tuple[int, ...]] = {}
    for factor in (0.0, 6.0):
        setup = load_chat_setup(LEAF_META, stats_path)
        signal = SteeringSignal(
            name="difference-probe",
            target_scores=leaf_diffmean,
            source=SignalSource.AXBENCH_CONCEPT,
            identity="difference-probe",
        )
        session = ChatSession(
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            device="cuda",
            config=ChatSessionConfig(
                self_cond_segment_size=int(model.dec_config.sequence_len),
                temperature=TEMPERATURE,
                top_k=TOP_K,
                seed=42,
                max_new_tokens_per_turn=64,
                deterministic_rewind=False,
            ),
            extraction=extraction_service_factory(setup, tokenizer),
        )
        session.set_steering(SteeringSelection(SteeringMethod.INTERPOLATION, ((signal, factor),)))
        session.send_user_turn(first_concept_prompt, lambda _event: None)
        conversation = session.conversation_tokens()
        assert len(conversation) > 2
        sampled_by_factor[factor] = conversation
        s0_entry = next(entry for entry in session.ledger() if entry.selfcond_ordinal == 0)
        assert s0_entry.features, "S0 boundary recorded no conditioning features"
        s0_features_by_factor[factor] = dict(s0_entry.features)

    # The sound reachability invariant: at full weight the S0 selection is
    # drawn from the diffmean scores instead of the model's own predictions
    # (identical RNG state on both runs), so the selected features must differ
    # if and only if the policy actually reached the provider's selection.
    assert s0_features_by_factor[0.0] != s0_features_by_factor[6.0], (
        "full-weight interpolation left the S0 selected features unchanged: "
        "steering is not reaching the provider"
    )
    if sampled_by_factor[0.0] == sampled_by_factor[6.0]:
        print(
            "note: S0 conditioning differed but the sampled streams coincided "
            "for this artifact/seed"
        )
