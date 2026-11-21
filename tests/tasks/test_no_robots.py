"""Tests for the NoRobots task class (HuggingFaceH4/no_robots).

9.5K human-authored instruction-following examples across 10 task categories.
Used as a quality anchor in the safety SFT mixture.
"""

import random

import pytest

from tasks.no_robots import NoRobots


class TestNoRobots:
    """Integration tests for the NoRobots task class."""

    @pytest.fixture(scope="class")
    def task(self) -> NoRobots:
        """Load the NoRobots task once for all tests in this class."""
        return NoRobots(split="train")

    def test_num_examples_positive(self, task: NoRobots) -> None:
        """Verify that the dataset contains >9000 examples."""
        n = task.num_examples()
        assert isinstance(n, int), f"num_examples() should return int, got {type(n)}"
        assert n > 9000, f"num_examples() should be > 9000, got {n}"

    def test_get_example_format(self, task: NoRobots) -> None:
        """Verify that each example has the correct conversation format:
        >= 2 messages, first is user, second is assistant, non-empty strings."""
        example = task.get_example(0)
        assert isinstance(example, dict), "Example must be a dict"
        assert "messages" in example, "Example must have 'messages' key"
        messages = example["messages"]

        # May have an optional system message at the start
        if messages[0]["role"] == "system":
            rest = messages[1:]
        else:
            rest = messages

        assert len(rest) >= 2, f"Expected >= 2 non-system messages, got {len(rest)}"
        assert rest[0]["role"] == "user", (
            f"First non-system message should be 'user', got {rest[0]['role']}"
        )
        assert rest[1]["role"] == "assistant", (
            f"Second non-system message should be 'assistant', got {rest[1]['role']}"
        )
        assert isinstance(rest[0]["content"], str), "User content must be a string"
        assert isinstance(rest[1]["content"], str), "Assistant content must be a string"
        assert len(rest[0]["content"]) > 0, "User content must be non-empty"
        assert len(rest[1]["content"]) > 0, "Assistant content must be non-empty"

    def test_slicing(self) -> None:
        """Verify that stop=5 gives exactly 5 examples."""
        task = NoRobots(split="train", stop=5)
        assert len(task) == 5, f"Expected 5 examples with stop=5, got {len(task)}"
        # Verify all 5 examples are accessible and well-formed
        for i in range(5):
            example = task[i]
            assert "messages" in example
            assert len(example["messages"]) >= 2

    def test_visual_inspection(self, task: NoRobots) -> None:
        """Print 5 random samples for human inspection."""
        rng = random.Random(42)
        n = task.num_examples()
        indices = rng.sample(range(n), k=5)
        for i, idx in enumerate(indices):
            example = task.get_example(idx)
            messages = example["messages"]
            # Find first user and assistant messages
            user_msg = next(m for m in messages if m["role"] == "user")
            asst_msg = next(m for m in messages if m["role"] == "assistant")
            print(f"\n--- Sample {i + 1} (index={idx}) ---")
            print(f"USER: {user_msg['content'][:200]}")
            print(f"ASSISTANT: {asst_msg['content'][:200]}")
