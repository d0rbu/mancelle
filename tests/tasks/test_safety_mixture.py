"""Integration tests for the safety SFT mixture.

Note: WildJailbreak and WildGuardMix are gated datasets on HuggingFace
and require authentication. Tests for those are skipped unless HF auth
is configured. Run `huggingface-cli login` to enable them.
"""

import os

import pytest

from tasks.coconot import CoCoNot
from tasks.common import TaskMixture
from tasks.wildguardmix import WildGuardMix
from tasks.wildjailbreak import WildJailbreak

_HAS_HF_AUTH = (
    os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    or "HF_TOKEN" in os.environ
)


class TestSafetyMixture:
    """Integration test: build the safety mixture and verify it works."""

    def test_safety_mixture_builds(self) -> None:
        """Verify non-gated safety task classes load and combine into a TaskMixture."""
        tasks = [CoCoNot(split="train", stop=5)]
        mixture = TaskMixture(tasks)
        assert len(mixture) == 5

        # Verify all examples have correct format
        for i in range(len(mixture)):
            example = mixture[i]
            assert isinstance(example, dict)
            assert "messages" in example
            messages = example["messages"]
            assert len(messages) >= 2
            assert messages[0]["role"] in ("user", "system")
            assert messages[-1]["role"] == "assistant"

    def test_safety_mixture_with_epochs(self) -> None:
        """Verify oversampling (multiple epochs) works correctly."""
        task = CoCoNot(split="train", stop=3)
        mixture = TaskMixture([task, task, task])  # 3 epochs
        assert len(mixture) == 9

    def test_all_public_safety_tasks_conversation_format(self) -> None:
        """Each non-gated safety task produces valid conversations with string content."""
        task_classes = [
            lambda: CoCoNot(split="train", stop=10),
        ]
        for make_task in task_classes:
            task = make_task()
            for i in range(len(task)):
                example = task[i]
                messages = example["messages"]
                for msg in messages:
                    assert isinstance(msg["content"], str), (
                        f"{task.__class__.__name__} index {i}: content is not a string"
                    )
                    assert len(msg["content"]) > 0, (
                        f"{task.__class__.__name__} index {i}: content is empty"
                    )
                    assert msg["role"] in ("user", "assistant", "system"), (
                        f"{task.__class__.__name__} index {i}: invalid role {msg['role']}"
                    )

    @pytest.mark.skipif(
        not _HAS_HF_AUTH, reason="WildGuardMix requires HF auth (gated dataset)"
    )
    def test_wildguardmix_loads(self) -> None:
        """WildGuardMix loads if HF auth is configured."""
        task = WildGuardMix(split="train", stop=5)
        assert len(task) == 5

    @pytest.mark.skipif(
        not _HAS_HF_AUTH, reason="WildJailbreak requires HF auth (gated dataset)"
    )
    def test_wildjailbreak_loads(self) -> None:
        """WildJailbreak loads if HF auth is configured."""
        task = WildJailbreak(split="train", stop=5)
        assert len(task) == 5
