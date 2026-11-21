"""
Tests for WildGuardMix safety moderation task.
Requires HuggingFace authentication to access the gated dataset.
"""

import os
import random

import pytest

from tasks.wildguardmix import WildGuardMix

_HAS_HF_AUTH = (
    os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    or "HF_TOKEN" in os.environ
)


@pytest.mark.skipif(
    not _HAS_HF_AUTH, reason="WildGuardMix requires HF auth (gated dataset)"
)
class TestWildGuardMix:
    """Tests for the WildGuardMix task class."""

    @pytest.fixture(scope="class")
    def task(self) -> WildGuardMix:
        """Load the WildGuardMix task once for all tests in this class."""
        return WildGuardMix(split="train")

    def test_num_examples_positive(self, task: WildGuardMix) -> None:
        """Verify that the dataset contains >20K examples after filtering."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 20_000, f"num_examples() should be > 20000 (expected ~38K), got {n}"

    def test_get_example_format(self, task: WildGuardMix) -> None:
        """Verify that each example has the correct conversation format:
        exactly 2 messages, correct roles, string content, non-empty."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

        user_msg, assistant_msg = messages
        assert user_msg["role"] == "user", (
            f"First message role should be 'user', got {user_msg['role']}"
        )
        assert assistant_msg["role"] == "assistant", (
            f"Second message role should be 'assistant', got {assistant_msg['role']}"
        )
        assert isinstance(user_msg["content"], str), "User content must be a string"
        assert isinstance(assistant_msg["content"], str), (
            "Assistant content must be a string"
        )
        assert len(user_msg["content"]) > 0, "User content must be non-empty"
        assert len(assistant_msg["content"]) > 0, "Assistant content must be non-empty"

    def test_only_rows_with_responses(self, task: WildGuardMix) -> None:
        """Sample 100 examples and verify all have non-empty assistant content."""
        rng = random.Random(123)
        n = task.num_examples()
        indices = rng.sample(range(n), k=100)
        for idx in indices:
            example = task.get_example(idx)
            assistant_content = example["messages"][1]["content"]
            assert isinstance(assistant_content, str), (
                f"Assistant content at index {idx} must be a string, got {type(assistant_content)}"
            )
            assert len(assistant_content) > 0, (
                f"Assistant content at index {idx} must be non-empty"
            )

    def test_slicing(self) -> None:
        """Verify that stop=5 gives exactly 5 examples."""
        task = WildGuardMix(split="train", stop=5)
        assert len(task) == 5, f"Expected 5 examples with stop=5, got {len(task)}"
        # Verify each example is valid
        for i in range(5):
            example = task[i]
            assert "messages" in example, f"Example {i} must have 'messages' key"

    def test_visual_inspection(self, task: WildGuardMix) -> None:
        """Print 5 random samples for human inspection."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=5)
        for i, idx in enumerate(indices):
            example = task.get_example(idx)
            messages = example["messages"]
            print(f"\n--- Sample {i + 1} (index={idx}) ---")
            print(f"USER: {messages[0]['content'][:200]}")
            print(f"ASSISTANT: {messages[1]['content'][:200]}")
