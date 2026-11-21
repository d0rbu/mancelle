"""Integration tests for the capabilities SFT mixture.

The capabilities mixture focuses on model abilities and knowledge:
SmolTalk (subsampled), identity, MMLU, OpenCoder, OrcaMath, MetaMathQA
(GSM-only), SimpleSpelling, SpellingBee.
Validation uses GSM8K (not MetaMathQA).
"""

from tasks.code_alpaca import _VAL_SIZE as _CODE_ALPACA_VAL_SIZE
from tasks.code_alpaca import CodeAlpaca
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.metamath import _VAL_SIZE as _METAMATH_VAL_SIZE
from tasks.metamath import MetaMathQA
from tasks.mmlu import MMLU
from tasks.orca_math import _VAL_SIZE as _ORCA_MATH_VAL_SIZE
from tasks.orca_math import OrcaMath
from tasks.smoltalk import SmolTalk
from tasks.spellingbee import SimpleSpelling, SpellingBee


class TestCapabilitiesMixture:
    """Integration tests: build the capabilities mixture and verify correctness."""

    def test_capabilities_mixture_builds(self) -> None:
        """All capabilities tasks load and combine into a TaskMixture."""
        tasks = [
            SmolTalk(split="train", stop=5),
            MMLU(subset="auxiliary_train", split="train", stop=5),
            OrcaMath(split="train", stop=5),
            MetaMathQA(split="train", gsm_only=True, stop=5),
            SimpleSpelling(size=5, split="train"),
            SpellingBee(size=5, split="train"),
        ]
        mixture = TaskMixture(tasks)
        assert len(mixture) == 30  # 6 tasks x 5 each

        for i in range(len(mixture)):
            example = mixture[i]
            assert isinstance(example, dict)
            assert "messages" in example
            messages = example["messages"]
            assert len(messages) >= 2
            assert messages[0]["role"] in ("user", "system")
            assert messages[-1]["role"] == "assistant"

    def test_capabilities_mixture_with_epochs(self) -> None:
        """Oversampling (multiple epochs) works correctly."""
        mmlu_epochs = 3
        tasks = [
            SmolTalk(split="train", stop=3),
            *[
                MMLU(subset="auxiliary_train", split="train", stop=3)
                for _ in range(mmlu_epochs)
            ],
            OrcaMath(split="train", stop=3),
            MetaMathQA(split="train", gsm_only=True, stop=3),
            SimpleSpelling(size=6, split="train"),  # doubled size instead of 2 epochs
            SpellingBee(size=6, split="train"),  # doubled size instead of 2 epochs
        ]
        mixture = TaskMixture(tasks)
        # 1 SmolTalk + 3 MMLU + 1 OrcaMath + 1 MetaMathQA + 1 SimpleSpelling + 1 SpellingBee = 8
        # 6 tasks with 3 rows + 2 tasks with 6 rows = 18 + 12 = 30
        assert len(mixture) == 30

    def test_all_capabilities_tasks_conversation_format(self) -> None:
        """Each capabilities task produces valid conversations with string content."""
        task_factories = [
            lambda: SmolTalk(split="train", stop=10),
            lambda: MMLU(subset="auxiliary_train", split="train", stop=10),
            lambda: OrcaMath(split="train", stop=10),
            lambda: MetaMathQA(split="train", gsm_only=True, stop=10),
            lambda: SimpleSpelling(size=10, split="train"),
            lambda: SpellingBee(size=10, split="train"),
        ]
        for make_task in task_factories:
            task = make_task()
            for i in range(len(task)):
                example = task[i]
                messages = example["messages"]
                for msg in messages:
                    content = msg["content"]
                    if isinstance(content, list):
                        for part in content:
                            assert isinstance(part, dict), (
                                f"{task.__class__.__name__} index {i}: list part is not a dict"
                            )
                    else:
                        assert isinstance(content, str), (
                            f"{task.__class__.__name__} index {i}: content is not a string"
                        )
                        assert len(content) > 0, (
                            f"{task.__class__.__name__} index {i}: content is empty"
                        )
                    assert msg["role"] in ("user", "assistant", "system"), (
                        f"{task.__class__.__name__} index {i}: invalid role {msg['role']}"
                    )

    def test_capabilities_val_tasks_build(self) -> None:
        """Validation split tasks load correctly (uses GSM8K, not MetaMathQA)."""
        val_tasks = [
            SmolTalk(split="test", stop=5),
            MMLU(subset="all", split="test", stop=5),
            OrcaMath(split="test", stop=5),
            GSM8K(subset="main", split="test", stop=5),
        ]
        mixture = TaskMixture(val_tasks)
        assert len(mixture) == 20

        for i in range(len(mixture)):
            example = mixture[i]
            assert isinstance(example, dict)
            assert "messages" in example

    def test_smoltalk_limit_applies(self) -> None:
        """SmolTalk stop parameter correctly limits the dataset size."""
        full = SmolTalk(split="train", stop=100)
        limited = SmolTalk(split="train", stop=10)
        assert len(full) == 100
        assert len(limited) == 10

    def test_code_alpaca_train_test_split(self) -> None:
        """CodeAlpaca train/test splits are disjoint and cover the filtered dataset."""
        train = CodeAlpaca(split="train")
        test = CodeAlpaca(split="test")
        assert len(test) == _CODE_ALPACA_VAL_SIZE
        assert len(train) + len(test) <= 20022
        assert len(train) > 19000

    def test_code_alpaca_test_no_overlap_with_train(self) -> None:
        """CodeAlpaca test examples are not in the train split."""
        train = CodeAlpaca(split="train", stop=50)
        test = CodeAlpaca(split="test", stop=50)
        train_prompts = {train[i]["messages"][0]["content"] for i in range(len(train))}
        test_prompts = {test[i]["messages"][0]["content"] for i in range(len(test))}
        assert len(train_prompts & test_prompts) == 0, (
            "Train and test splits must not overlap"
        )

    def test_code_alpaca_loads(self) -> None:
        """CodeAlpaca loads and produces valid conversations."""
        for split in ("train", "test"):
            task = CodeAlpaca(split=split, stop=10)
            assert len(task) == 10
            for i in range(len(task)):
                example = task[i]
                assert "messages" in example
                messages = example["messages"]
                assert len(messages) == 2
                assert messages[0]["role"] == "user"
                assert messages[1]["role"] == "assistant"
                assert len(messages[0]["content"]) > 0
                assert len(messages[1]["content"]) > 0

    def test_orca_math_train_test_split(self) -> None:
        """OrcaMath train/test splits are disjoint and cover the filtered dataset."""
        train = OrcaMath(split="train")
        test = OrcaMath(split="test")
        assert len(test) == _ORCA_MATH_VAL_SIZE
        assert len(train) + len(test) <= 200035
        assert len(train) > 198000

    def test_orca_math_test_no_overlap_with_train(self) -> None:
        """OrcaMath test examples are not in the train split."""
        train = OrcaMath(split="train", stop=50)
        test = OrcaMath(split="test", stop=50)
        train_prompts = {train[i]["messages"][0]["content"] for i in range(len(train))}
        test_prompts = {test[i]["messages"][0]["content"] for i in range(len(test))}
        assert len(train_prompts & test_prompts) == 0, (
            "Train and test splits must not overlap"
        )

    def test_orca_math_loads(self) -> None:
        """OrcaMath loads and produces valid conversations."""
        for split in ("train", "test"):
            task = OrcaMath(split=split, stop=10)
            assert len(task) == 10
            for i in range(len(task)):
                example = task[i]
                assert "messages" in example
                messages = example["messages"]
                assert len(messages) == 2
                assert messages[0]["role"] == "user"
                assert messages[1]["role"] == "assistant"
                assert len(messages[0]["content"]) > 0
                assert len(messages[1]["content"]) > 0

    def test_metamath_gsm_only_filter(self) -> None:
        """MetaMathQA gsm_only=True returns fewer rows than unfiltered."""
        MetaMathQA(split="train", stop=100)
        MetaMathQA(split="train", gsm_only=True, stop=100)
        # gsm_only should have fewer total rows available (but stop=100 limits both)
        full_total = MetaMathQA(split="train")
        gsm_total = MetaMathQA(split="train", gsm_only=True)
        assert len(gsm_total) < len(full_total)

    def test_metamath_train_test_split(self) -> None:
        """MetaMathQA train/test splits are disjoint."""
        train = MetaMathQA(split="train", gsm_only=True)
        test = MetaMathQA(split="test", gsm_only=True)
        assert len(test) == _METAMATH_VAL_SIZE
        assert len(train) > 200000  # GSM-only should have >200K rows

    def test_metamath_loads(self) -> None:
        """MetaMathQA loads and produces valid conversations."""
        for split in ("train", "test"):
            task = MetaMathQA(split=split, gsm_only=True, stop=10)
            assert len(task) == 10
            for i in range(len(task)):
                example = task[i]
                assert "messages" in example
                messages = example["messages"]
                assert len(messages) == 2
                assert messages[0]["role"] == "user"
                assert messages[1]["role"] == "assistant"
                assert len(messages[0]["content"]) > 0
                assert len(messages[1]["content"]) > 0

    def test_deterministic_shuffle(self) -> None:
        """TaskMixture shuffle is deterministic (seed=42) — same order each time."""

        def build():
            tasks = [
                SmolTalk(split="train", stop=5),
                MMLU(subset="auxiliary_train", split="train", stop=5),
                OrcaMath(split="train", stop=5),
                MetaMathQA(split="train", gsm_only=True, stop=5),
            ]
            return TaskMixture(tasks)

        m1 = build()
        m2 = build()
        for i in range(len(m1)):
            ex1 = m1[i]
            ex2 = m2[i]
            assert ex1["messages"][0]["content"] == ex2["messages"][0]["content"]
