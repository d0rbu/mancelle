"""Tests for safety-focused SFT task classes.

Covers WildJailbreak and future safety task classes.
Each test class loads the real HuggingFace dataset, so these are
integration tests that require network access and HF authentication.

Note: WildJailbreak is a gated dataset on HuggingFace and requires
authentication. Tests are skipped unless HF auth is configured.
Run `huggingface-cli login` to enable them.
"""

import os
import random

import pytest

from tasks.wildjailbreak import WildJailbreak

_HAS_HF_AUTH = (
    os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    or "HF_TOKEN" in os.environ
)


# ---------------------------------------------------------------------------
# WildJailbreak tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _HAS_HF_AUTH, reason="WildJailbreak requires HF auth (gated dataset)"
)
class TestWildJailbreak:
    """Integration tests for the WildJailbreak task class."""

    @pytest.fixture(scope="class")
    def task(self):
        """Load dataset once per test class (expensive)."""
        return WildJailbreak()

    def test_num_examples_positive(self, task):
        """Dataset should contain >200K examples after filtering."""
        n = task.num_examples()
        assert n > 200_000, f"Expected >200K examples, got {n}"

    def test_get_example_format(self, task):
        """Each example should have the correct messages structure."""
        ex = task.get_example(0)

        # Top-level keys
        assert "messages" in ex, "Example must have 'messages' key"
        assert "data_type" in ex, "Example must have 'data_type' key"

        # Messages structure
        messages = ex["messages"]
        assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert isinstance(messages[0]["content"], str)
        assert isinstance(messages[1]["content"], str)
        assert len(messages[0]["content"]) > 0, "User content must be non-empty"
        assert len(messages[1]["content"]) > 0, "Assistant content must be non-empty"

    def test_all_data_types_included(self, task):
        """Sampling 500 examples should reveal multiple data types."""
        n = task.num_examples()
        rng = random.Random(123)
        indices = rng.sample(range(n), 500)
        data_types = {task.get_example(i)["data_type"] for i in indices}

        expected_types = {
            "vanilla_harmful",
            "vanilla_benign",
            "adversarial_harmful",
            "adversarial_benign",
        }
        assert data_types == expected_types, (
            f"Expected all 4 data types, got {data_types}"
        )

    def test_slicing(self):
        """stop=10 should give exactly 10 examples via len()."""
        task = WildJailbreak(stop=10)
        assert len(task) == 10

        # Also verify we can access all 10
        for i in range(10):
            ex = task[i]
            assert "messages" in ex

    def test_visual_inspection(self, task):
        """Print 5 random samples for manual inspection (always passes)."""
        n = task.num_examples()
        rng = random.Random(99)
        indices = rng.sample(range(n), 5)

        for i in indices:
            ex = task.get_example(i)
            user_msg = ex["messages"][0]["content"]
            asst_msg = ex["messages"][1]["content"]
            print(f"\n--- Example {i} (data_type={ex['data_type']}) ---")
            print(f"USER: {user_msg[:200]}...")
            print(f"ASST: {asst_msg[:200]}...")
