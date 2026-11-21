"""Tests for the AnthropicHH task class (Anthropic/hh-rlhf harmless-base).

Anthropic HH-RLHF harmless-base contains human-preferred harmless conversations
parsed from the chosen response field. These are integration tests that require
network access to download from HuggingFace.
"""

import random

import pytest

from tasks.anthropic_hh import AnthropicHH


class TestAnthropicHH:
    """Integration tests for the AnthropicHH task class."""

    @pytest.fixture(scope="class")
    def task(self) -> AnthropicHH:
        """Load the AnthropicHH task once for all tests in this class."""
        return AnthropicHH(split="train")

    def test_num_examples_positive(self, task: AnthropicHH) -> None:
        """Verify that the dataset contains >30K examples after parsing."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 30_000, f"num_examples() should be > 30000, got {n}"

    def test_get_example_format(self, task: AnthropicHH) -> None:
        """Verify that each example has the correct messages structure:
        >= 2 messages, first is user, last is assistant, all non-empty strings."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]
        assert len(messages) >= 2, f"Expected >= 2 messages, got {len(messages)}"

        # First message must be user, last must be assistant
        assert messages[0]["role"] == "user", (
            f"First message role should be 'user', got {messages[0]['role']}"
        )
        assert messages[-1]["role"] == "assistant", (
            f"Last message role should be 'assistant', got {messages[-1]['role']}"
        )

        # All messages must have non-empty string content
        for i, msg in enumerate(messages):
            assert isinstance(msg["content"], str), (
                f"Message {i} content must be a string, got {type(msg['content'])}"
            )
            assert len(msg["content"]) > 0, f"Message {i} content must be non-empty"

    def test_roles_alternate(self, task: AnthropicHH) -> None:
        """Sample 50 examples and verify roles strictly alternate user/assistant."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=50)
        for idx in indices:
            example = task.get_example(idx)
            messages = example["messages"]
            expected_roles = ["user", "assistant"] * ((len(messages) + 1) // 2)
            for i, msg in enumerate(messages):
                assert msg["role"] == expected_roles[i], (
                    f"Example {idx}, message {i}: expected role "
                    f"'{expected_roles[i]}', got '{msg['role']}'"
                )

    def test_slicing(self) -> None:
        """Verify that stop=5 gives exactly 5 examples."""
        task = AnthropicHH(split="train", stop=5)
        assert len(task) == 5, f"Expected 5 examples with stop=5, got {len(task)}"
        # Verify all 5 examples are accessible and well-formed
        for i in range(5):
            example = task[i]
            assert "messages" in example
            assert len(example["messages"]) >= 2

    def test_visual_inspection(self, task: AnthropicHH) -> None:
        """Print 5 random samples (first 4 messages each) for manual inspection."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=5)
        for i, idx in enumerate(indices):
            example = task.get_example(idx)
            messages = example["messages"]
            print(f"\n--- Sample {i + 1} (index={idx}, {len(messages)} messages) ---")
            for msg in messages[:4]:
                print(f"  [{msg['role'].upper()}]: {msg['content'][:200]}")
            if len(messages) > 4:
                print(f"  ... ({len(messages) - 4} more messages)")
