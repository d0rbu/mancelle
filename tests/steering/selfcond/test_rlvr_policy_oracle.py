"""Independent native-forward oracle for turn-RLVR policy traces."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch as t

import steering.eval.inference_engine as inference
from steering.conditional.encoder_decoder.rlvr_policy import collect_rlvr_group
from steering.eval.target_conditioning import ContextRoutingMode
from tests.steering.selfcond.rlvr_test_helpers import policy as _policy
from tests.steering.selfcond.rlvr_test_helpers import script as _script
from tests.steering.test_chat_self_conditioned_generation import make_fixture


@dataclass(frozen=True)
class _NativeFPRow:
    position: int
    token_id: int
    slot: int
    fp: t.Tensor


@dataclass(frozen=True)
class _NativeAppend:
    owner: str
    rows: t.Tensor
    source_slot: int
    result_slot: int


def _recording_factory(
    fixture: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Any, list[_NativeFPRow], list[_NativeAppend]]:
    """Observe real KV forwards and provider inputs without reading trace callbacks."""

    native_rows: list[_NativeFPRow] = []
    native_appends: list[_NativeAppend] = []
    original_forward = fixture.model.forward

    def forward(ids: t.Tensor, *args: Any, **kwargs: Any) -> Any:
        kv_cache = kwargs.get("kv_cache")
        position = None if kv_cache is None else int(kv_cache.get_pos())
        result = original_forward(ids, *args, **kwargs)
        if position is not None and kwargs.get("return_fp_logits"):
            _logits, fp_logits = result
            slots = kwargs["seg_map"][0].tolist()
            for offset, (token_id, slot) in enumerate(zip(ids[0].tolist(), slots, strict=True)):
                native_rows.append(
                    _NativeFPRow(
                        position=position + offset,
                        token_id=int(token_id),
                        slot=int(slot),
                        fp=fp_logits[0, offset].detach().clone(),
                    )
                )
        return result

    monkeypatch.setattr(fixture.model, "forward", forward)

    def factory(_rollout_index: int) -> Any:
        wrapper_template = fixture.engine._model
        engine = inference.build_engine_self_conditioned(
            fixture.model,
            fixture.tokenizer,
            t.device("cpu"),
            wrapper_template._self_cond.config,
            context_provider=fixture.provider,
            context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE,
            turn_contract=fixture.contract,
        )
        wrapper = engine._model
        original_swap = wrapper.swap_self_cond_provider

        def swap(provider: Any, config: Any) -> None:
            for method_name, owner in (
                ("seed_from_prompt", "prompt"),
                ("step", "assistant"),
                ("append_tool_window", "tool"),
            ):
                original_append = getattr(provider, method_name)

                def append(
                    state: Any,
                    rows: t.Tensor,
                    *args: Any,
                    _owner: str = owner,
                    _original: Any = original_append,
                    **kwargs: Any,
                ) -> Any:
                    result = _original(state, rows, *args, **kwargs)
                    native_appends.append(
                        _NativeAppend(
                            owner=_owner,
                            rows=rows.detach().clone(),
                            source_slot=int(state.target_slot_idx),
                            result_slot=int(result.target_slot_idx),
                        )
                    )
                    return result

                monkeypatch.setattr(provider, method_name, append)
            original_swap(provider, config)

        monkeypatch.setattr(wrapper, "swap_self_cond_provider", swap)
        return engine

    return factory, native_rows, native_appends


def _positions_for_rows(
    native_rows: list[_NativeFPRow],
    appended: t.Tensor,
    *,
    already_used: set[int],
) -> tuple[int, ...]:
    """Find the unique contiguous native FP window passed to one provider call."""

    width = int(appended.shape[1])
    candidates: list[tuple[int, ...]] = []
    for start in range(len(native_rows) - width + 1):
        window = native_rows[start : start + width]
        positions = tuple(row.position for row in window)
        if any(position in already_used for position in positions) or positions != tuple(
            range(positions[0], positions[0] + width)
        ):
            continue
        if all(t.equal(appended[0, offset], row.fp) for offset, row in enumerate(window)):
            candidates.append(positions)
    assert len(candidates) == 1, candidates
    already_used.update(candidates[0])
    return candidates[0]


def _assert_trace_matches_native_policy_state(
    group: Any,
    native_rows: list[_NativeFPRow],
    native_appends: list[_NativeAppend],
) -> None:
    trajectory = group.trajectories[0]
    assert len(trajectory.actions) == len(native_appends)
    by_position = {row.position: row for row in native_rows}
    assert len(by_position) == len(native_rows)
    used_fp_positions: set[int] = set()

    for decision_index, (action, appended) in enumerate(
        zip(trajectory.actions, native_appends, strict=True)
    ):
        pool_positions = _positions_for_rows(
            native_rows,
            appended.rows,
            already_used=used_fp_positions,
        )
        prefix_length = pool_positions[-1] + 1
        expected_prefix = tuple(by_position[position].token_id for position in range(prefix_length))
        expected_routing = tuple(by_position[position].slot for position in range(prefix_length))

        assert expected_prefix == trajectory.token_ids[:prefix_length]
        assert action.decision_index == decision_index
        assert action.source == appended.owner
        assert action.pool_positions == pool_positions
        assert action.first_committed_position == prefix_length
        assert action.committed_prefix == expected_prefix
        assert action.outgoing_seg_map == expected_routing
        assert appended.result_slot == group.prompt.encoder_slot_count + decision_index
        assert appended.source_slot == appended.result_slot - 1

        assert all(
            t.equal(appended.rows[0, offset], by_position[position].fp)
            for offset, position in enumerate(action.pool_positions)
        )


@pytest.mark.parametrize(("segment_size", "generated_count"), [(1, 2), (33, 34)])
@pytest.mark.parametrize("head_depth", [0, 2])
def test_assistant_policy_trace_is_exact_native_fp_and_slot_oracle(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    segment_size: int,
    generated_count: int,
    head_depth: int,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=segment_size, head_depth=head_depth)
    factory, native_rows, native_appends = _recording_factory(fixture, monkeypatch)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [token] * generated_count + [end])

    group = collect_rlvr_group(
        fixture.model,
        factory,
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=1701 + segment_size,
        max_tokens=generated_count + 1,
        extraction_tokenizer=fixture.source,
    )

    _assert_trace_matches_native_policy_state(group, native_rows, native_appends)
    if segment_size == 1:
        trajectory = group.trajectories[0]
        mutated = replace(
            trajectory.actions[1],
            first_committed_position=trajectory.actions[1].first_committed_position + 1,
        )
        bad_group = replace(
            group,
            trajectories=(
                replace(
                    trajectory,
                    actions=(trajectory.actions[0], mutated, *trajectory.actions[2:]),
                ),
            ),
        )
        with pytest.raises(AssertionError):
            _assert_trace_matches_native_policy_state(bad_group, native_rows, native_appends)


@pytest.mark.parametrize("head_depth", [0, 2])
def test_multiwindow_calculator_trace_is_exact_native_fp_and_slot_oracle(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    head_depth: int,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2, head_depth=head_depth)
    factory, native_rows, native_appends = _recording_factory(fixture, monkeypatch)
    tokenizer = fixture.tokenizer
    special = tokenizer.encode_special
    assistant = [
        special("<python_start>"),
        *tokenizer.encode("123456789*123456789"),
        special("<python_end>"),
    ]
    forced = [
        special("<output_start>"),
        *tokenizer.encode("15241578750190521"),
        special("<output_end>"),
    ]
    resumed = tokenizer.encode(" done")
    script = [*assistant, *forced, *resumed, special("<assistant_end>")]
    _script(monkeypatch, script)

    group = collect_rlvr_group(
        fixture.model,
        factory,
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=1801,
        max_tokens=len(script),
        extraction_tokenizer=fixture.source,
    )

    assert sum(record.owner == "tool" for record in native_appends) >= 2
    _assert_trace_matches_native_policy_state(group, native_rows, native_appends)


@pytest.mark.parametrize("head_depth", [0, 2])
def test_k1_marker_action_followed_immediately_by_eos_is_not_live(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    head_depth: int,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=1, head_depth=head_depth)
    factory, native_rows, native_appends = _recording_factory(fixture, monkeypatch)
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end])

    group = collect_rlvr_group(
        fixture.model,
        factory,
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=1901,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )

    _assert_trace_matches_native_policy_state(group, native_rows, native_appends)
    trajectory = group.trajectories[0]
    assert [action.source for action in trajectory.actions] == ["prompt", "assistant"]
    assert [action.decision_index for action in trajectory.live_actions] == [0]
    assert trajectory.actions[1].first_committed_position == len(fixture.prompt)
    assert trajectory.last_nonforced_logits_prefix_length == len(fixture.prompt)


@pytest.mark.parametrize("final_draw", ["eos", "budget"])
@pytest.mark.parametrize("head_depth", [0, 2])
def test_completed_final_tool_action_without_later_logits_is_not_live(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    final_draw: str,
    head_depth: int,
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2, head_depth=head_depth)
    factory, native_rows, native_appends = _recording_factory(fixture, monkeypatch)
    tokenizer = fixture.tokenizer
    special = tokenizer.encode_special
    assistant = [special("<python_start>"), *tokenizer.encode("1+1"), special("<python_end>")]
    forced = [special("<output_start>"), *tokenizer.encode("2"), special("<output_end>")]
    final_token = special("<assistant_end>") if final_draw == "eos" else tokenizer.encode("x")[0]
    script = [*assistant, *forced, final_token]
    _script(monkeypatch, script)

    group = collect_rlvr_group(
        fixture.model,
        factory,
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=2001,
        max_tokens=len(script),
        extraction_tokenizer=fixture.source,
    )

    _assert_trace_matches_native_policy_state(group, native_rows, native_appends)
    trajectory = group.trajectories[0]
    final_action = trajectory.actions[-1]
    assert final_action.source == "tool"
    assert final_action not in trajectory.live_actions
    assert final_action.first_committed_position == trajectory.last_nonforced_logits_prefix_length
    if final_draw == "budget":
        assert trajectory.token_ids[-1] == final_token
        assert trajectory.sampled_mask[-1] == 1
    else:
        assert final_token not in trajectory.token_ids[len(fixture.prompt) :]
