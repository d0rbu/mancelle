"""Unit tests for the newly added capabilities and safety task loaders."""

from __future__ import annotations

from typing import Any

from tasks.nemotron_safety import NemotronSafety
from tasks.opencoder import OpenCoder


def test_opencoder_combines_all_stage2_configs(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_load_dataset(dataset_id: str, config: str, *, split: str) -> list[dict[str, str]]:
        calls.append((dataset_id, config))
        return [{"instruction": config, "output": f"answer-{config}"}]

    monkeypatch.setattr("tasks.opencoder.load_dataset", fake_load_dataset)
    task = OpenCoder()

    assert len(task) == 4
    assert len(calls) == 4
    assert {config for _, config in calls} == {
        "educational_instruct",
        "evol_instruct",
        "mceval_instruct",
        "package_instruct",
    }
    assert task[0]["messages"][0]["role"] == "user"
    assert task[0]["messages"][1]["role"] == "assistant"

    capped = OpenCoder(max_rows=2)
    assert len(capped) == 2


def test_nemotron_safety_normalizes_messages_and_drops_reasoning(monkeypatch) -> None:
    def fake_load_dataset(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        assert args == ("json",)
        assert kwargs["split"] == "train"
        return [
            {
                "messages": [
                    {"role": "user", "content": "question"},
                    {
                        "role": "assistant",
                        "content": "answer",
                        "reasoning_content": "private chain",
                    },
                ]
            }
        ]

    monkeypatch.setattr("tasks.nemotron_safety.load_dataset", fake_load_dataset)
    task = NemotronSafety()

    assert len(task) == 1
    assert task[0] == {
        "messages": [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ]
    }
