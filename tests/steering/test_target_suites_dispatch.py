"""PR-C: chat_target / core_target suites dispatch through enumerate_expected_eval_contexts.

Mirrors the test pattern in test_validate_eval_context_cache.py:291-320 —
monkeypatch the private _enumerate_*_target_contexts wrappers so we don't
need real eval bundles or chat task datasets.
"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import scripts.eval.validate_eval_context_cache as validator
from tests.conftest import build_tiny_tokenizer


def _cfg(**overrides):
    base = {
        "evals": "chat_target",
        "max_per_task": 1,
        "model_max_seq_len": 64,
        "base_seed": 1234,
        "num_seeds": 1,
        "chat_tasks": "ARC-Easy",
        "chat_max_problems": 1,
        "chat_batch_size": 1,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_chat_target_suite_dispatches_to_chat_target_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validator,
        "_enumerate_chat_target_contexts",
        lambda cfg, tokenizer: calls.append("chat_target") or [],
    )
    validator.enumerate_expected_eval_contexts(_cfg(evals="chat_target"), build_tiny_tokenizer())
    assert calls == ["chat_target"]


def test_core_target_suite_dispatches_to_core_target_helper(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validator,
        "_enumerate_core_target_contexts",
        lambda cfg, tokenizer, seed: calls.append(("core_target", seed)) or [],
    )
    validator.enumerate_expected_eval_contexts(
        _cfg(evals="core_target"), build_tiny_tokenizer(),
    )
    # One seed (num_seeds=1, base_seed=1234)
    assert calls == [("core_target", 1234)]


def test_combined_suites_concatenate(monkeypatch):
    calls = []
    monkeypatch.setattr(
        validator,
        "_enumerate_chat_target_contexts",
        lambda cfg, tokenizer: calls.append("chat_target") or [],
    )
    monkeypatch.setattr(
        validator,
        "_enumerate_core_target_contexts",
        lambda cfg, tokenizer, seed: calls.append(("core_target", seed)) or [],
    )
    validator.enumerate_expected_eval_contexts(
        _cfg(evals="core_target,chat_target"), build_tiny_tokenizer(),
    )
    assert calls == [("core_target", 1234), "chat_target"]


def test_unknown_suite_still_rejected():
    """Regression: unsupported suites continue to raise."""
    with pytest.raises(ValueError, match="Unsupported eval suite"):
        validator.enumerate_expected_eval_contexts(
            _cfg(evals="not_a_suite"), build_tiny_tokenizer(),
        )


def test_chat_target_helper_iterates_chat_tasks(monkeypatch):
    """The _enumerate_chat_target_contexts helper iterates cfg.chat_tasks
    (comma-separated string) and calls enumerate_chat_target_contexts per task."""
    captured = []

    class FakeTask:
        def __getitem__(self, idx):
            return {"messages": [
                {"role": "user", "content": "Q"},
                {"role": "assistant", "content": "A"},
            ]}

        def __len__(self):
            return 1

    fake_factories = {"FakeChat": FakeTask}
    monkeypatch.setattr(validator, "CHAT_TASK_FACTORIES", fake_factories)

    def _fake_enumerate(*, tokenizer, task_object, max_problems, task_name):
        captured.append((task_name, max_problems))
        return []

    monkeypatch.setattr(validator, "enumerate_chat_target_contexts", _fake_enumerate)

    cfg = _cfg(evals="chat_target", chat_tasks="FakeChat", chat_max_problems=3)
    validator.enumerate_expected_eval_contexts(cfg, build_tiny_tokenizer())
    assert captured == [("FakeChat", 3)]


def test_core_target_helper_iterates_core_tasks(monkeypatch, tmp_path):
    """The _enumerate_core_target_contexts helper iterates the eval_bundle
    task list, sparsely loads its cap, and enumerates those selected rows."""
    # Build a minimal fake eval_bundle directory so the helper resolves.
    bundle = tmp_path / "eval_bundle"
    (bundle / "eval_data").mkdir(parents=True)
    (bundle / "core.yaml").write_text(
        "icl_tasks:\n"
        "  - label: fake_mc\n"
        "    icl_task_type: multiple_choice\n"
        "    num_fewshot: [0]\n"
        "    dataset_uri: fake.jsonl\n"
        "    continuation_delimiter: ' '\n"
    )
    (bundle / "eval_meta_data.csv").write_text("Eval Task,Random baseline\nfake_mc,0.5\n")
    (bundle / "eval_data" / "fake.jsonl").write_text(
        '{"query": "Q1", "choices": ["A", "B"], "gold": 0}\n'
    )

    monkeypatch.setattr(validator, "get_base_dir", lambda: str(tmp_path))

    loaded = []
    captured = []

    def _fake_load(path, *, max_per_task):
        loaded.append((path.name, max_per_task))
        return [{"query": "Q1", "choices": ["A", "B"], "gold": 0}]

    def _fake_enumerate(*, tokenizer, data, task_meta, seed, task_label, max_seq_len):
        captured.append((task_label, seed, len(data)))
        return []

    monkeypatch.setattr(validator, "load_selected_core_task_data", _fake_load)
    monkeypatch.setattr(
        validator,
        "enumerate_selected_core_target_contexts",
        _fake_enumerate,
    )

    cfg = _cfg(evals="core_target", max_per_task=5)
    validator.enumerate_expected_eval_contexts(cfg, build_tiny_tokenizer())
    assert loaded == [("fake.jsonl", 5)]
    assert captured == [("fake_mc", 1234, 1)]
