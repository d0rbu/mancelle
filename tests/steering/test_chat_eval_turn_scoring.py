"""Script completions through main while retaining task parsing and code execution."""

import pytest
import torch as t
import yaml

from nanochat.engine import Engine
from nanochat.execution import execute_code
from scripts.eval import chat_eval
from tasks import humaneval, spellingbee
from tests.steering.test_chat_eval_turn_cache_integration import turn_main_fixture


@pytest.mark.parametrize("human_success", [True, False])
@pytest.mark.parametrize("spelling_success", [True, False])
def test_main_propagates_real_humaneval_and_spellingbee_scores(
    tmp_path, monkeypatch, human_success, spelling_success,
):
    fixture = turn_main_fixture(tmp_path, monkeypatch)
    tokenizer = fixture.tokenizer
    fixture.cfg.task_name = "HumanEval|SpellingBee"
    fixture.cfg.max_new_tokens = 128
    returned = 2 if human_success else 9
    predicted_count = "2" if spelling_success else "9"
    completions = {"HumanEval": f"```python\ndef f():\n    return {returned}\n```",
                   "SpellingBee": f"The count is:\n#### {predicted_count}"}
    prompts = {name: chat_eval.render_chat_completion(tokenizer, fixture.tasks[name][0]).token_ids
               for name in completions}
    scripts = {tuple(prompts[name]): [*tokenizer.encode(completion),
        tokenizer.encode_special("<assistant_end>")] for name, completion in completions.items()}
    current_prompt = ()
    forwards = []
    original_forward = fixture.model.forward

    def scripted_forward(ids, *args, **kwargs):
        nonlocal current_prompt
        cache = kwargs["kv_cache"]
        if cache.get_pos() == 0:
            current_prompt = tuple(ids[0].tolist())
        logits = original_forward(ids, *args, **kwargs)
        assert t.isfinite(logits).all()
        forwards.append(current_prompt)
        next_position = cache.get_pos() - len(current_prompt)
        scripted = t.full_like(logits, -20.0)
        scripted[:, -1, scripts[current_prompt][next_position]] = 20.0
        return scripted

    monkeypatch.setattr(fixture.model, "forward", scripted_forward)
    transcripts = []
    original_generate = Engine.generate_batch

    def capture_generate(self, *args, **kwargs):
        output = original_generate(self, *args, **kwargs)
        transcripts.append(output)
        return output

    monkeypatch.setattr(Engine, "generate_batch", capture_generate)
    executions = []
    original_execute = humaneval.python_program_passes

    def capture_execute(program):
        result = original_execute(program)
        executions.append((program, result))
        return result

    monkeypatch.setattr(humaneval, "python_program_passes", capture_execute)
    parsed_counts = []
    original_extract = spellingbee.extract_answer

    def capture_extract(completion):
        count = original_extract(completion)
        parsed_counts.append((completion, count))
        return count

    monkeypatch.setattr(spellingbee, "extract_answer", capture_extract)
    spelling_outcomes = []
    original_evaluate = spellingbee.SpellingBee.evaluate

    def capture_evaluate(self, conversation, prediction):
        outcome = original_evaluate(self, conversation, prediction)
        spelling_outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(spellingbee.SpellingBee, "evaluate", capture_evaluate)
    chat_eval.main.__wrapped__(fixture.cfg)
    assert len(executions) == 1
    program, passed = executions[0]
    assert program == (f"\n\ndef f():\n    return {returned}\n\n"
                       "def check(fn):\n    assert fn() == 2\ncheck(f)")
    execution = execute_code(program)
    assert passed is execution.success is human_success
    assert execution.error is None if human_success else execution.error == "AssertionError: "
    assert [count for _, count in parsed_counts] == ["2", predicted_count]
    assert parsed_counts[1][0] == completions["SpellingBee"]
    assert spelling_outcomes == [int(spelling_success)]
    for (name, completion), (rows, masks) in zip(completions.items(), transcripts, strict=True):
        generated = tokenizer.encode(completion)
        assert rows == [prompts[name] + generated]
        assert masks == [[0] * len(prompts[name]) + [1] * len(generated)]
        assert forwards.count(tuple(prompts[name])) == len(generated) + 1
    payload = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    assert payload["seeds"]["adhoc"]["results"] == {
        "HumanEval": 1.0 if human_success else 0.0,
        "SpellingBee": 1.0 if spelling_success else 0.0,
    }
