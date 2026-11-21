"""Actual local task data and fingerprints, with only HF dataset loads stubbed."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from datasets import Dataset

from errors import DataLoadError
from scripts.utils import build_hf_sft_data_manifest as manifest
from steering.sft.loss_mask import LossMaskMode
from tasks import gsm8k, mixtures, mmlu, smoltalk
from tasks.common import TaskMixture
from tasks.customjson import CustomJSON


class TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def render_conversation(self, conversation: Any) -> tuple[list[int], list[int]]:
        return [0, 2, 3, 4, 5], [0, 0, 1, 1, 1]


@pytest.fixture
def snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    base = tmp_path / "cache"
    base.mkdir()
    messages = [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}]
    (base / "identity_conversations.jsonl").write_text(json.dumps(messages) + "\n")
    (base / "words_alpha.txt").write_text("apple\nbanana\ncat\n")
    steered = tmp_path / "steered.jsonl"
    steered.write_text(json.dumps({"messages": messages}) + "\n")
    monkeypatch.setitem(mixtures._DEFAULTS, "simple_spelling_size", 3)
    monkeypatch.setitem(mixtures._DEFAULTS, "spelling_bee_size", 2)

    def load_dataset(name: str, *args: object, **kwargs: Any) -> Dataset:
        if name == "HuggingFaceTB/smol-smoltalk":
            return Dataset.from_list([{"messages": messages}] * 3)
        if name == "cais/mmlu":
            row = {
                "question": "Q?",
                "choices": ["a", "b", "c", "d"],
                "answer": 0,
                "subject": "test",
            }
            # The production val task slices to 5200 rows.
            return Dataset.from_list(
                [{"train": row}] * 3 if args[0] == "auxiliary_train" else [row] * 5200
            )
        if name == "openai/gsm8k":
            return Dataset.from_list([{"question": "1+1?", "answer": "2 #### 2"}] * 420)
        raise AssertionError(f"unexpected dataset: {name}")

    monkeypatch.setattr(
        manifest,
        "load_hf_config_and_tokenizer",
        lambda **kwargs: (
            None,
            TinyTokenizer(),
            SimpleNamespace(chat_template_sha256="a" * 64),
        ),
    )
    for module in (smoltalk, mmlu, gsm8k):
        monkeypatch.setattr(module, "load_dataset", load_dataset)
    return base, steered


def test_manifest_roundtrip_and_complete_task_coverage(
    snapshot: tuple[Path, Path], tmp_path: Path
) -> None:
    base, steered = snapshot
    payload = manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    output = tmp_path / "manifest.json"
    digest = manifest.write_manifest(payload, output)
    assert manifest.verify_manifest(payload, output) == digest
    assert hashlib.sha256(output.read_bytes()).hexdigest() == digest
    assert Path(f"{output}.sha256").read_text() == f"{digest}  manifest.json\n"
    assert payload["format"] == manifest.MANIFEST_FORMAT
    assert payload["mixture"] == "standard_steered"
    assert [task["class_name"] for task in payload["train"]["tasks"]] == [
        "SmolTalk",
        "CustomJSON",
        "CustomJSON",
        "MMLU",
        "GSM8K",
        "SimpleSpelling",
        "SpellingBee",
        "SteeredConversations",
    ]
    assert [task["class_name"] for task in payload["validation"]["tasks"]] == [
        "SmolTalk",
        "MMLU",
        "GSM8K",
    ]
    assert payload["train"]["fingerprint"]["len_mixture"] == 434
    assert (
        payload["inputs"]["identity"]["sha256"]
        == hashlib.sha256((base / "identity_conversations.jsonl").read_bytes()).hexdigest()
    )
    assert manifest.build_manifest(base_dir=base, steered_jsonl=steered) == payload
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        manifest.write_manifest(payload, output)


@pytest.mark.parametrize(
    "input_name", ["identity_conversations.jsonl", "words_alpha.txt", "steered.jsonl"]
)
def test_content_drift_with_same_row_counts_is_rejected(
    snapshot: tuple[Path, Path], tmp_path: Path, input_name: str
) -> None:
    base, steered = snapshot
    before = manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    output = tmp_path / "manifest.json"
    manifest.write_manifest(before, output)
    path = steered if input_name == "steered.jsonl" else base / input_name
    path.write_text(path.read_text().replace("Hello", "Howdy").replace("apple", "peach"))
    after = manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    assert before["train"]["fingerprint"] == after["train"]["fingerprint"]
    with pytest.raises(ValueError, match="live SFT data differs"):
        manifest.verify_manifest(after, output)


@pytest.mark.parametrize("missing", ["identity", "steered"])
def test_missing_input_fails_before_task_loading(
    snapshot: tuple[Path, Path], missing: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, steered = snapshot
    monkeypatch.setenv("NANOCHAT_BASE_DIR", "/previous")
    (steered if missing == "steered" else base / "identity_conversations.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    assert manifest.os.environ["NANOCHAT_BASE_DIR"] == "/previous"


def test_invalid_and_empty_inputs_fail(snapshot: tuple[Path, Path]) -> None:
    base, steered = snapshot
    steered.write_text("")
    with pytest.raises(ValueError, match="SFT input is empty"):
        manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    steered.write_text("not json\n")
    with pytest.raises(DataLoadError, match="not valid JSON"):
        manifest.build_manifest(base_dir=base, steered_jsonl=steered)


def test_epoch_uses_packer_and_rounds_each_rank_to_optimizer_boundary(
    snapshot: tuple[Path, Path],
) -> None:
    base, _steered = snapshot
    dataset = TaskMixture([CustomJSON(str(base / "identity_conversations.jsonl"))] * 33)
    epoch = manifest.measure_epoch(
        dataset,
        TinyTokenizer(),
        device_batch_size=1,
        loss_mask=LossMaskMode.FULL_SEQUENCE,
        max_seq_len=7,
        total_batch_size=56,
        world_size=4,
    )
    # Five-token conversations cannot share an eight-token row. Rank sizes
    # are [9,8,8,8], with two micro-batches per optimizer update.
    assert epoch["per_rank_steps"] == [5, 4, 4, 4]
    assert epoch["num_iterations"] == 5
    assert epoch["scheduled_tokens"] == 280
    with pytest.raises(ValueError, match="divisible"):
        manifest.measure_epoch(
            dataset, TinyTokenizer(), device_batch_size=3, loss_mask=LossMaskMode.FULL_SEQUENCE
        )


def test_hf_dataset_content_drift_is_bound_even_when_order_is_unchanged(
    snapshot: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, steered = snapshot
    before = manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    output = tmp_path / "manifest.json"
    manifest.write_manifest(before, output)
    changed = Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "Changed"},
                    {"role": "assistant", "content": "Answer"},
                ]
            }
        ]
        * 3
    )
    monkeypatch.setattr(smoltalk, "load_dataset", lambda *args, **kwargs: changed)
    after = manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    assert before["train"]["fingerprint"] == after["train"]["fingerprint"]
    assert before["inputs"] == after["inputs"]
    with pytest.raises(ValueError, match="live SFT data differs"):
        manifest.verify_manifest(after, output)


def test_verification_checks_bodies_without_replaying_the_measured_epoch(
    snapshot: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base, steered = snapshot
    payload = manifest.build_manifest(base_dir=base, steered_jsonl=steered)
    output = tmp_path / "manifest.json"
    manifest.write_manifest(payload, output)

    def no_replay(*args: Any, **kwargs: Any) -> None:
        pytest.fail("verification must reuse the frozen CPU horizon")

    monkeypatch.setattr(manifest, "measure_epoch", no_replay)
    verified = manifest.build_manifest(
        base_dir=base, steered_jsonl=steered, measured_epoch=payload["epoch"]
    )
    assert manifest.verify_manifest(verified, output)
    with pytest.raises(ValueError, match="geometry"):
        manifest.validate_epoch(
            payload["epoch"], device_batch_size=2, loss_mask=LossMaskMode.FULL_SEQUENCE
        )
    invalid = dict(payload["epoch"], num_iterations=999)
    with pytest.raises(ValueError, match="derived step count"):
        manifest.validate_epoch(invalid, device_batch_size=1, loss_mask=LossMaskMode.FULL_SEQUENCE)


def test_tokenizer_identity_follows_the_selected_sft_config(
    snapshot: tuple[Path, Path],
) -> None:
    base_dir, steered = snapshot
    default = manifest.build_manifest(base_dir=base_dir, steered_jsonl=steered)
    small = manifest.build_manifest(
        base_dir=base_dir, steered_jsonl=steered, sft_config_name="chat_sft_pythia_1_4b"
    )
    assert default["tokenizer"]["model_id"] == "EleutherAI/pythia-2.8b"
    assert small["tokenizer"]["model_id"] == "EleutherAI/pythia-1.4b"
    assert small["tokenizer"]["revision"] == "fedc38a16eea3bd36a96b906d78d11d2ce18ed79"
    with pytest.raises(FileNotFoundError, match="unknown SFT training config"):
        manifest.build_manifest(
            base_dir=base_dir, steered_jsonl=steered, sft_config_name="chat_sft_no_such_model"
        )
    with pytest.raises(ValueError, match="nonblank training config name"):
        manifest.build_manifest(base_dir=base_dir, steered_jsonl=steered, sft_config_name=" ")


@pytest.mark.parametrize(
    "supplied",
    [
        {"parent_export"},
        {"expected_parent_mixture"},
        {"data_manifest"},
        {"parent_export", "expected_parent_mixture"},
        {"parent_export", "data_manifest"},
        {"expected_parent_mixture", "data_manifest"},
    ],
)
def test_parent_manifest_config_is_all_or_none(tmp_path: Path, supplied: set[str]) -> None:
    values = {
        "parent_export": "/parent",
        "expected_parent_mixture": "standard_steered",
        "data_manifest": "/data.json",
    }
    cfg = SimpleNamespace(
        mode="build",
        output_path=str(tmp_path / "output.json"),
        **{name: value if name in supplied else None for name, value in values.items()},
    )
    with pytest.raises(ValueError, match=r"must be supplied together; missing=\["):
        manifest.main.__wrapped__(cfg)
