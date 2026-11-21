"""Tests for the UltraChat task class (HuggingFaceH4/ultrachat_200k).

UltraChat 200K is a high-quality multi-turn conversational dataset used as
the helpfulness component in safety SFT mixtures. These are integration tests
that require network access to download from HuggingFace.
"""

import random

import pytest

from tasks.ultrachat import UltraChat


class TestUltraChat:
    """Integration tests for the UltraChat task class."""

    @pytest.fixture(scope="class")
    def task(self) -> UltraChat:
        """Load the UltraChat task once for all tests in this class."""
        return UltraChat(split="train")

    def test_num_examples_positive(self, task: UltraChat) -> None:
        """Verify that the dataset contains >200K examples."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 200_000, f"num_examples() should be > 200000, got {n}"

    def test_get_example_format(self, task: UltraChat) -> None:
        """Verify that each example has the correct messages structure:
        >= 2 messages, first is user or system, all have string content."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]
        assert len(messages) >= 2, f"Expected >= 2 messages, got {len(messages)}"

        # First message must be user or system
        assert messages[0]["role"] in ("user", "system"), (
            f"First message role should be 'user' or 'system', got {messages[0]['role']}"
        )

        # All messages must have string content
        for i, msg in enumerate(messages):
            assert isinstance(msg["content"], str), (
                f"Message {i} content must be a string, got {type(msg['content'])}"
            )
            assert len(msg["content"]) > 0, f"Message {i} content must be non-empty"

    def test_subsampling_with_stop(self) -> None:
        """Verify that stop=100 gives exactly 100 examples, all well-formed."""
        task = UltraChat(split="train", stop=100)
        assert len(task) == 100, f"Expected 100 examples with stop=100, got {len(task)}"

        for i in range(100):
            example = task[i]
            assert "messages" in example, f"Example {i} must have 'messages' key"
            messages = example["messages"]
            assert len(messages) >= 2, (
                f"Example {i}: expected >= 2 messages, got {len(messages)}"
            )
            # First message must be user or system
            assert messages[0]["role"] in ("user", "system"), (
                f"Example {i}: first role should be user/system, got {messages[0]['role']}"
            )
            # All content must be non-empty strings
            for j, msg in enumerate(messages):
                assert isinstance(msg["content"], str) and len(msg["content"]) > 0, (
                    f"Example {i}, message {j}: content must be non-empty string"
                )

    def test_visual_inspection(self, task: UltraChat) -> None:
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
