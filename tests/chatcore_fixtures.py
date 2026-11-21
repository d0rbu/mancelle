"""Offline benchmark sources, real RustBPE, and structured conversion fixtures."""

import copy
from pathlib import Path

import pytest
from datasets import Dataset
from hydra import compose, initialize_config_dir

from nanochat.tokenizer import RustBPETokenizer
from scripts.preprocess.convert_sft_to_parquet import main as convert_main
from steering.sft.structured_dataset import load_structured_rollout_artifact
from steering.sft.structured_rollout import rustbpe_tokenizer_sha256

_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def rustbpe_tokenizer():
    # Train real byte-pair merges locally; no saved tokenizer or network needed.
    corpus = [
        "Multiple Choice question: What is one plus one? two three four five "
        "Respond only with the letter of the correct answer. ABCD 1234 "
        "We add 1+1=2. #### 2 ARC-Easy ARC-Challenge auxiliary_train main all "
        "train test source row café 数学"
    ] * 20
    tokenizer = RustBPETokenizer.train_from_iterator(iter(corpus), vocab_size=384)
    assert len(tokenizer.encode(corpus[0])) < len(corpus[0])
    return tokenizer


def _benchmark_dataset(path: str, subset: str, *, split: str) -> Dataset:
    """Real task loaders/renderers consume tiny fixtures with source split IDs."""
    question = f"{subset} {split}: What is one plus one?"
    if path == "allenai/ai2_arc":
        row = {
            "question": question,
            "choices": {"text": ["two", "three"], "label": ["A", "B"]},
            "answerKey": "A",
        }
    elif path == "cais/mmlu":
        row = {
            "question": question,
            "choices": ["two", "three", "four", "five"],
            "answer": 0,
            "subject": "elementary_mathematics",
        }
        if subset == "auxiliary_train":
            row = {"train": row}
    else:
        assert path == "openai/gsm8k"
        row = {"question": question, "answer": "We add <<1+1=2>>2.\n#### 2"}
    return Dataset.from_list([row])


_SOURCE_SIZES = {"ARC-Easy": 3, "ARC-Challenge": 5, "auxiliary_train": 7, "all": 7, "main": 9}


@pytest.fixture
def benchmark_sources(monkeypatch):
    """Patch only network I/O; keep HF shuffling and every task renderer real."""
    calls = []
    source_rows = {}

    def load(path, subset, *, split):
        calls.append((path, subset, split))
        key = (path, subset, split)
        if key not in source_rows:
            template = _benchmark_dataset(path, subset, split=split)[0]
            rows = []
            for index in range(_SOURCE_SIZES[subset]):
                row = copy.deepcopy(template)
                payload = row.get("train", row)
                payload["question"] = f"{subset}/{split}/{index} café 数学"
                if path == "allenai/ai2_arc":
                    labels = ["1", "2"] if index % 2 else ["A", "B"]
                    payload["choices"] = {"text": ["two", "three"], "label": labels}
                    payload["answerKey"] = labels[index % 2]
                elif path == "cais/mmlu":
                    payload["answer"] = index % 4
                rows.append(row)
            source_rows[key] = rows
        return Dataset.from_list(source_rows[key])

    for module in ("tasks.arc", "tasks.mmlu", "tasks.gsm8k"):
        monkeypatch.setattr(f"{module}.load_dataset", load)
    return calls, source_rows


def _converter_config(output_dir, split="both", **overrides):
    with initialize_config_dir(config_dir=str(_ROOT / "configs/data"), version_base=None):
        return compose(
            config_name="convert_sft_to_parquet",
            overrides=[
                "mixture=chatcore",
                f"split={split}",
                "artifact_format=structured_nanochat_rollout_v1",
                f"output_dir={output_dir}",
                "conversations_per_shard=5",
                *(f"{key}={value}" for key, value in overrides.items()),
            ],
        )


def _load_artifact(path, tokenizer, split="train"):
    return load_structured_rollout_artifact(
        path,
        decode=tokenizer.decode,
        tokenizer_sha256=rustbpe_tokenizer_sha256(tokenizer),
        expected_mixture="chatcore",
        expected_split=split,
    )


@pytest.fixture
def converted_chatcore(tmp_path, monkeypatch, benchmark_sources, rustbpe_tokenizer):
    monkeypatch.setattr(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer", lambda: rustbpe_tokenizer
    )
    cfg = _converter_config(tmp_path / "chatcore")
    convert_main.__wrapped__(cfg)
    return cfg, _load_artifact(tmp_path / "chatcore", rustbpe_tokenizer)
