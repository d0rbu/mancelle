"""Unit tests for axbench-bpb parquet writers."""
from __future__ import annotations

import pyarrow.parquet as pq

from steering.eval.axbench_bpb.parquet_writer import (
    BPB_SCORE_SCHEMA,
    write_bpb_score_rows,
)
from steering.eval.axbench_bpb.types import BpbScoreRow


def test_write_bpb_score_rows_roundtrip(tmp_path):
    rows = [
        BpbScoreRow(
            method_name="diffmean_decoder_only", concept_id=42,
            mode="positive", content_class="concept_rich",
            target="decoder_only", coefficient_str="1.0",
            sample_id="abc123", bpb_intervened=1.234,
            n_output_tokens=10, n_output_bytes=40, truncated=False,
        ),
    ]
    path = tmp_path / "bpb_score.parquet"
    write_bpb_score_rows(rows, path)
    table = pq.read_table(path)
    assert table.schema == BPB_SCORE_SCHEMA
    assert table.num_rows == 1
    assert table["method_name"][0].as_py() == "diffmean_decoder_only"
    assert table["coefficient_str"][0].as_py() == "1.0"
    assert table["bpb_intervened"][0].as_py() == 1.234


def test_write_bpb_score_rows_multiple_methods(tmp_path):
    """Multiple methods and concepts coexist in one parquet."""
    rows = [
        BpbScoreRow(
            method_name=f"method_{i % 3}", concept_id=i,
            mode="positive" if i % 2 == 0 else "negative",
            content_class="concept_rich",
            target="encdec" if i % 2 == 0 else "decoder_only",
            coefficient_str=str(float(i)), sample_id=f"s{i}",
            bpb_intervened=float(i) * 0.5,
            n_output_tokens=10, n_output_bytes=40, truncated=False,
        )
        for i in range(6)
    ]
    path = tmp_path / "multi_method.parquet"
    write_bpb_score_rows(rows, path)
    table = pq.read_table(path)
    assert table.num_rows == 6
    assert set(table["method_name"].to_pylist()) == {"method_0", "method_1", "method_2"}


def test_score_schema_has_conditioned_bpb_columns():
    assert "method_name" in [f.name for f in BPB_SCORE_SCHEMA]
    assert "bpb_intervened" in [f.name for f in BPB_SCORE_SCHEMA]
