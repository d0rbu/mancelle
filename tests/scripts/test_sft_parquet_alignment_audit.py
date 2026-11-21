from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis.sft_parquet_alignment_audit import (
    AuditStatus,
    _derive_val_parquet_dir,
    _load_parquet_text_by_original_index,
    _write_summary,
    compare_task_to_parquet,
    main,
    run_audit,
)
from scripts.preprocess.convert_sft_to_parquet import (
    render_conversation_to_full_nanochat_text,
)
from steering.sft_schema import SFTSplit


class _Task:
    def __init__(self, conversations: list[dict]):
        self.conversations = conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict:
        return self.conversations[index]


def _conversation(user: str, assistant) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


class _FakeTokenizer:
    def render_conversation(self, conversation):
        ids = [101]
        for message in conversation["messages"]:
            ids.extend(ord(ch) for ch in f"<{message['role']}>")
            content = message["content"]
            if isinstance(content, list):
                for part in content:
                    ids.extend(ord(ch) for ch in f"<{part['type']}>{part['text']}")
            else:
                ids.extend(ord(ch) for ch in str(content))
        return ids, [0] * len(ids)

    def get_bos_token_id(self):
        return 101

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _render(conversation: dict) -> str:
    text, skip_reason = render_conversation_to_full_nanochat_text(
        _FakeTokenizer(),
        conversation,
    )
    assert skip_reason is None
    assert text is not None
    return text


def _write_full_parquet(parquet_dir: Path, conversations: list[dict]) -> None:
    parquet_dir.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "original_index": list(range(len(conversations))),
                "text": [_render(conversation) for conversation in conversations],
            }
        ),
        parquet_dir / "shard_00000.parquet",
    )


def _patch_task_mixtures(
    monkeypatch: pytest.MonkeyPatch,
    train_conversations: list[dict],
    val_conversations: list[dict],
) -> list[dict]:
    """Patch the shared builder and return the recorded overrides."""
    recorded: list[dict] = []

    def fake_build_task_mixtures(_mixture, **overrides):
        recorded.append(overrides)
        return _Task(train_conversations), _Task(val_conversations)

    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.build_task_mixtures",
        fake_build_task_mixtures,
    )
    return recorded


@pytest.mark.parametrize(
    "dataset", ["standard_steered", "standard_steered_balanced"]
)
def test_steered_audit_uses_the_standard_val_family(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: str,
) -> None:
    train_dir = tmp_path / dataset
    val_dir = tmp_path / "standard_val"
    train_conversations = [_conversation("train", "answer")]
    val_conversations = [_conversation("val", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _write_full_parquet(val_dir, val_conversations)
    recorded = _patch_task_mixtures(monkeypatch, train_conversations, val_conversations)
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )

    audit = run_audit(
        dataset=dataset,
        parquet_dir=train_dir,
        steered_jsonl="/data/steered.jsonl",
        max_examples=1,
    )

    assert recorded == [{"steered_jsonl": "/data/steered.jsonl"}]
    # The val sibling is standard's, not "<dataset>_val".
    assert _derive_val_parquet_dir(train_dir, dataset, "standard") == val_dir
    assert audit.val_parquet_dir == str(val_dir)
    assert audit.val is not None
    assert audit.val.status is AuditStatus.AUDITED
    assert audit.val.matched == 1


def test_arbitrary_parquet_dir_keeps_its_own_val_sibling_for_a_non_steered_dataset(
    tmp_path: Path,
) -> None:
    """``--parquet-dir /arbitrary/path`` keeps ITS sibling, not the family's.

    This is the case the ``if val_dataset != dataset`` guard exists for: an
    unconditional ``with_name(val_dataset)`` would silently redirect an
    operator's own directory onto ``<base>/<dataset>_val``.
    """
    assert _derive_val_parquet_dir(
        tmp_path / "run-2026-09-02", "capabilities", "capabilities"
    ) == tmp_path / "run-2026-09-02_val"


def test_arbitrary_parquet_dir_still_resolves_onto_the_family_for_a_steered_dataset(
    tmp_path: Path,
) -> None:
    assert _derive_val_parquet_dir(
        tmp_path / "run-2026-09-02", "standard_steered", "standard"
    ) == tmp_path / "standard_val"


def test_steered_audit_requires_the_steered_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_task_mixtures(monkeypatch, [], [])
    with pytest.raises(
        ValueError, match=r"mixture 'standard_steered' requires steered_jsonl=<path>"
    ):
        run_audit(dataset="standard_steered", parquet_dir=tmp_path, max_examples=1)


def test_steered_audit_not_converted_val_names_the_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train_dir = tmp_path / "standard_steered"
    train_conversations = [_conversation("train", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _patch_task_mixtures(monkeypatch, train_conversations, [_conversation("v", "a")])
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )

    audit = run_audit(
        dataset="standard_steered",
        parquet_dir=train_dir,
        steered_jsonl="/data/steered.jsonl",
        max_examples=1,
    )

    assert audit.val is not None
    assert audit.val.status is AuditStatus.NOT_CONVERTED
    assert audit.val.recovery_command is not None
    assert "mixture=standard split=val" in audit.val.recovery_command


def test_render_helper_matches_tool_use_full_nanochat_text() -> None:
    text = _render(
        _conversation(
            "count",
            [
                {"type": "text", "text": "Let me check"},
                {"type": "python", "text": "1 + 1"},
                {"type": "python_output", "text": "2"},
                {"type": "text", "text": "\n#### 2"},
            ],
        )
    )

    assert "<user>count" in text
    assert "<assistant><text>Let me check" in text
    assert "<python>1 + 1" in text
    assert "<python_output>2" in text


def test_compare_task_to_parquet_detects_split_mismatch() -> None:
    train_conv = _conversation("train question", "train answer")
    val_conv = _conversation("val question", "val answer")
    parquet_rows = {0: _render(train_conv)}

    train_alignment = compare_task_to_parquet(
        task=_Task([train_conv]),
        split=SFTSplit.TRAIN,
        parquet_rows=parquet_rows,
        max_examples=1,
        tokenizer=_FakeTokenizer(),
    )
    val_alignment = compare_task_to_parquet(
        task=_Task([val_conv]),
        split=SFTSplit.VAL,
        parquet_rows=parquet_rows,
        max_examples=1,
        tokenizer=_FakeTokenizer(),
    )

    assert train_alignment.matched == 1
    assert train_alignment.mismatched == 0
    assert val_alignment.matched == 0
    assert val_alignment.mismatched == 1
    assert val_alignment.first_mismatches[0]["reason"] == "rendered_text_mismatch"


def test_compare_task_to_parquet_detects_missing_rows(tmp_path: Path) -> None:
    # Exercise the same full_nanochat_text shape written to parquet without
    # needing the full production mixture-building path.
    parquet_path = tmp_path / "shard_00000.parquet"
    pq.write_table(
        pa.table(
            {
                "original_index": [1],
                "text": [_render(_conversation("other", "answer"))],
            }
        ),
        parquet_path,
    )
    task = _Task([_conversation("missing", "answer")])

    alignment = compare_task_to_parquet(
        task=task,
        split=SFTSplit.TRAIN,
        parquet_rows={},
        max_examples=1,
        tokenizer=_FakeTokenizer(),
    )

    assert alignment.checked == 1
    assert alignment.missing_parquet_rows == 1
    assert alignment.first_mismatches[0]["reason"] == "missing_parquet_row"


def test_compare_task_to_parquet_skips_invalid_task_rows_when_missing() -> None:
    alignment = compare_task_to_parquet(
        task=_Task([{"messages": [{"role": "user", "content": "no answer"}]}]),
        split=SFTSplit.TRAIN,
        parquet_rows={},
        max_examples=1,
        tokenizer=_FakeTokenizer(),
    )

    assert alignment.checked == 1
    assert alignment.skipped_invalid_task_rows == 1
    assert alignment.missing_parquet_rows == 0
    assert alignment.mismatched == 0


@pytest.mark.parametrize(
    "legacy_column",
    ["messages", "assistant_loss_mask"],
)
def test_load_parquet_text_by_original_index_rejects_legacy_schema(
    tmp_path: Path,
    legacy_column: str,
) -> None:
    parquet_path = tmp_path / "shard_00000.parquet"
    pq.write_table(
        pa.table(
            {
                "original_index": [0],
                legacy_column: [json.dumps([{"role": "assistant", "content": "old"}])],
            }
        ),
        parquet_path,
    )

    with pytest.raises(ValueError, match="full_nanochat_text schema"):
        _load_parquet_text_by_original_index(tmp_path, {0})


def test_run_audit_derives_and_reads_val_parquet_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "capabilities"
    val_dir = tmp_path / "capabilities_val"
    train_conversations = [_conversation("train", "answer")]
    val_conversations = [_conversation("val", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _write_full_parquet(val_dir, val_conversations)
    _patch_task_mixtures(monkeypatch, train_conversations, val_conversations)
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )

    audit = run_audit(
        dataset="capabilities",
        parquet_dir=train_dir,
        max_examples=1,
    )

    assert _derive_val_parquet_dir(train_dir, "capabilities", "capabilities") == val_dir
    assert audit.val_parquet_dir == str(val_dir)
    assert audit.train.matched == 1
    assert audit.val is not None
    assert audit.val.status is AuditStatus.AUDITED
    assert audit.val.matched == 1
    assert audit.val.mismatched == 0


def test_run_audit_honors_val_parquet_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "train"
    val_dir = tmp_path / "custom_val"
    train_conversations = [_conversation("train", "answer")]
    val_conversations = [_conversation("val", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _write_full_parquet(val_dir, val_conversations)
    _patch_task_mixtures(monkeypatch, train_conversations, val_conversations)
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )

    audit = run_audit(
        dataset="capabilities",
        parquet_dir=train_dir,
        val_parquet_dir=val_dir,
        max_examples=1,
    )

    assert audit.val_parquet_dir == str(val_dir)
    assert audit.val is not None and audit.val.matched == 1


def test_missing_val_is_reported_as_not_converted_with_recovery_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train_dir = tmp_path / "safety"
    train_conversations = [_conversation("train", "answer")]
    val_conversations = [_conversation("val", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _patch_task_mixtures(monkeypatch, train_conversations, val_conversations)
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )

    audit = run_audit(
        dataset="safety",
        parquet_dir=train_dir,
        max_examples=1,
    )
    summary_path = tmp_path / "summary.md"
    _write_summary(summary_path, audit)

    assert audit.val is not None
    assert audit.val.status is AuditStatus.NOT_CONVERTED
    assert audit.val.recovery_command == (
        "uv run python -m scripts.preprocess.convert_sft_to_parquet mixture=safety split=val"
    )
    summary = summary_path.read_text()
    assert "val: not_converted" in summary
    assert audit.val.recovery_command in summary


@pytest.mark.parametrize(
    ("allow_missing", "expected_status"),
    [(False, 1), (True, 0)],
)
def test_cli_exit_status_for_missing_val(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_missing: bool,
    expected_status: int,
) -> None:
    train_dir = tmp_path / "safety"
    train_conversations = [_conversation("train", "answer")]
    val_conversations = [_conversation("val", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _patch_task_mixtures(monkeypatch, train_conversations, val_conversations)
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )
    argv = [
        "sft_parquet_alignment_audit",
        "--dataset",
        "safety",
        "--parquet-dir",
        str(train_dir),
        "--out-dir",
        str(tmp_path / "out"),
        "--max-examples",
        "1",
    ]
    if allow_missing:
        argv.append("--allow-missing-val")
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == expected_status


def test_cli_forwards_the_steered_jsonl_to_the_mixture_builder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--steered-jsonl`` must reach ``build_task_mixtures``, not be dropped."""
    train_dir = tmp_path / "standard_steered"
    val_dir = tmp_path / "standard_val"
    train_conversations = [_conversation("train", "answer")]
    val_conversations = [_conversation("val", "answer")]
    _write_full_parquet(train_dir, train_conversations)
    _write_full_parquet(val_dir, val_conversations)
    recorded = _patch_task_mixtures(monkeypatch, train_conversations, val_conversations)
    monkeypatch.setattr(
        "scripts.analysis.sft_parquet_alignment_audit.get_tokenizer",
        lambda: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sft_parquet_alignment_audit",
            "--dataset",
            "standard_steered",
            "--parquet-dir",
            str(train_dir),
            "--steered-jsonl",
            "/data/steered.jsonl",
            "--out-dir",
            str(tmp_path / "out"),
            "--max-examples",
            "1",
        ],
    )

    assert main() == 0
    assert recorded == [{"steered_jsonl": "/data/steered.jsonl"}]


def test_cli_rejects_non_positive_max_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sft_parquet_alignment_audit",
            "--dataset",
            "safety",
            "--max-examples",
            "0",
        ],
    )

    with pytest.raises(TypeError, match="PositiveInt"):
        main()
