"""Script token choices while retaining real scalar Engine, wrapper and model forwards."""

import pytest
import torch as t
import yaml

import nanochat.engine as scalar_engine
from scripts.eval import chat_eval
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
)
from steering.eval.target_conditioning import MultiTargetSegments
from steering.eval.target_real_runner import build_context_feature_provider
from tasks.gsm8k import GSM8K
from tests.steering.test_chat_eval_turn_cache_integration import turn_main_fixture


@pytest.mark.parametrize("lockstep", [False, True])
@pytest.mark.parametrize("ending", ["<assistant_end>", "<bos>", "budget"])
def test_calculator_forced_positions_boundaries_stops_and_budget(tmp_path, monkeypatch, lockstep, ending):
    # kills: teacher-forced generation, calculator loss, filtered tool transcript, FP refresh,
    # kills: repeated/skipped KV boundary inputs, fixed decode origin, missing reference clamp or budget cap.
    fixture = turn_main_fixture(tmp_path, monkeypatch, lockstep=lockstep)
    tokenizer = fixture.tokenizer
    conversation = fixture.tasks["GSM8K"][0]
    prompt = chat_eval.render_chat_completion(tokenizer, conversation).token_ids
    p0 = len(prompt) - 1
    forced = [tokenizer.encode_special("<output_start>"), *tokenizer.encode("2"), tokenizer.encode_special("<output_end>")]
    generated = [tokenizer.encode_special("<python_start>"), *tokenizer.encode("1+1"),
        tokenizer.encode_special("<python_end>"), *forced, *tokenizer.encode("thinking " * 5 + "#### 2")]
    script = generated + [tokenizer.encode_special("<assistant_end>" if ending == "budget" else ending)]
    fixture.cfg.task_name = "GSM8K"
    # Terminal runs exceed the model window if the existing cap is lost. The budget
    # run stops inside pending forced output and must return exactly that partial transcript.
    fixture.cfg.max_new_tokens = 6 if ending == "budget" else 4096
    expected_tokens = generated[:6] if ending == "budget" else generated
    expected_masks = ([1] * 5 + [0] * len(forced) + [1] * (len(generated) - 5 - len(forced)))[:len(expected_tokens)]
    traces = []
    original_forward = fixture.model.forward

    def scripted_forward(ids, *args, **kwargs):
        cache = kwargs["kv_cache"]
        absolute = cache.get_pos()
        logits = original_forward(ids, *args, **kwargs)
        assert t.isfinite(logits).all()
        traces.append((absolute, ids.tolist()[0], kwargs["seg_map"].tolist()[0]))
        next_position = cache.get_pos() - len(prompt)
        assert next_position < len(script)
        scripted = logits.clone()
        scripted[:, -1, :] = -20.0
        scripted[:, -1, script[next_position]] = 20.0
        return scripted

    monkeypatch.setattr(fixture.model, "forward", scripted_forward)
    calculator_calls = []
    original_calculator = scalar_engine.use_calculator

    def calculator(expr):
        calculator_calls.append(expr)
        return original_calculator(expr)

    monkeypatch.setattr(scalar_engine, "use_calculator", calculator)
    outputs = []
    budgets = []
    original_generate = scalar_engine.Engine.generate_batch

    def capture_generate(self, *args, **kwargs):
        budgets.append(kwargs["max_tokens"])
        output = original_generate(self, *args, **kwargs)
        outputs.append(output)
        return output

    monkeypatch.setattr(scalar_engine.Engine, "generate_batch", capture_generate)
    predictions = []
    original_evaluate = GSM8K.evaluate

    def capture_evaluate(self, example, prediction):
        predictions.append(prediction)
        return original_evaluate(self, example, prediction)

    monkeypatch.setattr(GSM8K, "evaluate", capture_evaluate)
    wrappers = []
    original_artifacts = chat_eval.build_target_real_artifacts_for_example

    def capture_artifacts(**kwargs):
        artifacts = original_artifacts(**kwargs)
        wrappers.append(artifacts)
        return artifacts

    monkeypatch.setattr(chat_eval, "build_target_real_artifacts_for_example", capture_artifacts)
    monkeypatch.setattr(chat_eval, "generate_greedy_token_batch", lambda *args, **kwargs: pytest.fail("decoder-only batch path used"))
    chat_eval.main.__wrapped__(fixture.cfg)
    assert calculator_calls == ["1+1"]
    assert outputs == [([prompt + expected_tokens], [[0] * len(prompt) + expected_masks])]
    assert predictions == [tokenizer.decode(expected_tokens)]
    assert budgets == [6 if ending == "budget" else 2048 - len(prompt)]
    result = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    assert result["seeds"]["adhoc"]["results"] == {"GSM8K": 0.0 if ending == "budget" else 1.0}
    artifacts = wrappers[0]
    assert not artifacts.wrapper.is_self_conditioned
    state = artifacts.wrapper._generation_encoder_state
    assert state.target_start_pos == p0
    assert [position for position, _ids, _route in traces] == [0, *range(len(prompt), len(prompt) + len(expected_tokens))]
    assert [token for _position, ids, _route in traces[1:] for token in ids] == expected_tokens
    provider = build_context_feature_provider(fixture.cfg, cache_dir=fixture.cache_cfg.output_dir, turn_contract=fixture.contract)
    full_ids, _mask = tokenizer.render_conversation(conversation)
    segments = provider.get(full_ids)
    assert segments is not None
    full_routing = build_full_sequence_segment_routing(ids=full_ids, tokenizer=tokenizer,
        seg_features=([t.tensor(seg.feature_ids) for seg in segments], [t.tensor(seg.buckets) for seg in segments],
                      [seg.char_start for seg in segments], [seg.char_end for seg in segments]), source_bos_char_len=3)
    if lockstep:
        assert isinstance(artifacts.target, MultiTargetSegments)
        reference = full_routing.seg_map[p0:] - full_routing.seg_map[p0]
        assert t.equal(artifacts.target.token_assignments, reference)
    else:
        reference = t.zeros(len(full_ids) - p0, dtype=t.int64)
    assert traces[0][2][-1] == state.target_base_idx + int(reference[0])
    for position, _ids, route in traces[1:]:
        assert route == [state.target_base_idx + int(reference[min(position - p0, len(reference) - 1)])]
    if ending != "budget":
        assert len(expected_tokens) > len(reference)
        assert traces[-1][2] == [state.target_base_idx + int(reference[-1])]
    assert len(artifacts.wrapper._prepared_turn_contexts) == 1
