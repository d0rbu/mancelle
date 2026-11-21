from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.analysis import sft_supervision_audit as audit


class _FakeTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode(self, text, prepend=None):
        ids = [ord(ch) for ch in text]
        return ([prepend] if prepend is not None else []) + ids

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


def _write_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "abc",
        "abcdef",
    ]
    table = pa.table(
        {
            "text": rows,
            "original_index": [0, 1],
        }
    )
    pq.write_table(table, path)


def test_audit_supervision_counts_full_sequence_loss_positions(monkeypatch, tmp_path: Path) -> None:
    parquet_dir = tmp_path / "capabilities"
    _write_parquet(parquet_dir / "shard_00000.parquet")
    monkeypatch.setattr(audit, "get_tokenizer", lambda: _FakeTokenizer())

    stats, coverage = audit.audit_supervision(
        dataset="capabilities",
        parquet_dir=parquet_dir,
        feature_dir=None,
        max_examples=None,
        row_capacity=1025,
        min_loss_tokens=1,
        load_features_flag=False,
    )

    assert coverage is None
    assert stats.scanned_examples == 2
    assert stats.full_sequence_eligible_examples == 2
    assert stats.conditioned_effective_examples == 2
    assert stats.full_sequence_loss_positions == 9
    assert stats.conditioned_effective_loss_positions == 9


@pytest.mark.parametrize(
    "legacy_column",
    ["response", "assistant_loss_mask"],
)
def test_iter_parquet_rows_rejects_legacy_sft_columns(
    tmp_path: Path,
    legacy_column: str,
) -> None:
    parquet_dir = tmp_path / "capabilities"
    parquet_dir.mkdir()
    pq.write_table(
        pa.table(
            {
                "text": ["rendered"],
                "original_index": [0],
                legacy_column: ["old"],
            }
        ),
        parquet_dir / "shard_00000.parquet",
    )

    with pytest.raises(ValueError, match="legacy SFT parquet schema"):
        list(audit._iter_parquet_rows(parquet_dir))
