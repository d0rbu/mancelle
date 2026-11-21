from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_bpb import live_ours_vocab_contract as contract_module
from steering.eval.axbench_bpb.live_ours_vocab_contract import (
    AFFECTED_LIVE_OURS_METHODS,
    LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME,
    assert_aggregate_live_ours_vocab_contract,
    enforce_live_ours_vocab_contract,
    live_ours_vocab_contract_identity,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError

_AFFECTED = "ours_extract_and_add_encdec"
_UNAFFECTED = "control_encdec"
_FP_A = "a" * 64
_FP_B = "b" * 64


class _Vocab:
    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint

    def fingerprint(self) -> str:
        return self._fingerprint


def _model(fingerprint: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(feature_vocab=None if fingerprint is None else _Vocab(fingerprint))


def _method(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name)


def _write_methods(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"method_name": names}), path)


def _marker(scores_dir: Path) -> Path:
    return scores_dir / LIVE_OURS_VOCAB_CONTRACT_MARKER_NAME


def test_affected_method_set_is_derived_as_the_canonical_seven() -> None:
    assert AFFECTED_LIVE_OURS_METHODS == {
        "ours_extract_and_add_encdec",
        "ours_replace_random_encdec",
        "ours_replace_all_encdec",
        "neg_ours_remove_encdec",
        "neg_ours_add_to_negatives_encdec",
        "neg_ours_replace_random_negatives_encdec",
        "neg_ours_replace_all_negatives_encdec",
    }


@pytest.mark.parametrize("fingerprint", [None, _FP_A])
def test_current_affected_method_stamps_expected_identity(
    tmp_path: Path,
    fingerprint: str | None,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(_AFFECTED)],
        model=_model(fingerprint),
    )
    assert _marker(scores_dir).read_text().strip() == (
        live_ours_vocab_contract_identity(_model(fingerprint))
    )


def test_stale_marker_is_replaced_when_no_affected_rows_exist(
    tmp_path: Path,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    _write_methods(scores_dir / "rank_000" / "batch_000000.parquet", [_UNAFFECTED])
    _marker(scores_dir).write_text(f"axbench-bpb-live-ours-v1:{_FP_B}\n")

    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(_AFFECTED)],
        model=_model(_FP_A),
    )

    assert _marker(scores_dir).read_text().strip().endswith(_FP_A)


def test_irrelevant_run_leaves_any_marker_untouched(tmp_path: Path) -> None:
    scores_dir = tmp_path / "bpb_scores"
    scores_dir.mkdir()
    marker = _marker(scores_dir)
    marker.write_text("arbitrary-legacy-text\n")

    # The no-affected/no-current row of the decision table must not even need a
    # model exposing feature_vocab.
    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(_UNAFFECTED)],
        model=None,
    )

    assert marker.read_text() == "arbitrary-legacy-text\n"


def test_relevant_contract_requires_loaded_model(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="loaded encoder-decoder model"):
        enforce_live_ours_vocab_contract(
            scores_dir=tmp_path,
            methods=[_method(_AFFECTED)],
            model=None,
        )


def test_marker_read_error_is_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    scores_dir.mkdir()
    marker = _marker(scores_dir)
    marker.write_text("present")
    original_read_text = Path.read_text

    def fail_marker_read(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> str:
        if path == marker:
            raise OSError("simulated read failure")
        return original_read_text(
            path,
            encoding=encoding,
            errors=errors,
            newline=newline,
        )

    monkeypatch.setattr(Path, "read_text", fail_marker_read)
    with pytest.raises(AxBenchBpbError, match="failed to read.*simulated"):
        enforce_live_ours_vocab_contract(
            scores_dir=scores_dir,
            methods=[_method(_AFFECTED)],
            model=_model(_FP_A),
        )


def test_exact_marker_accepts_existing_affected_rows_across_ranks(
    tmp_path: Path,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    _write_methods(scores_dir / "rank_001" / "batch_000004.parquet", [_AFFECTED])
    expected = live_ours_vocab_contract_identity(_model(_FP_A))
    _marker(scores_dir).write_text(f"{expected}\n")

    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(_UNAFFECTED)],
        model=_model(_FP_A),
    )


@pytest.mark.parametrize("current_method", [_AFFECTED, _UNAFFECTED])
@pytest.mark.parametrize("initial_marker", [None, f"axbench-bpb-live-ours-v1:{_FP_B}"])
def test_concurrent_rank_marker_write_is_reread_after_shard_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    current_method: str,
    initial_marker: str | None,
) -> None:
    """A peer may stamp the marker and flush while this rank scans shards."""
    scores_dir = tmp_path / "bpb_scores"
    expected = live_ours_vocab_contract_identity(_model(_FP_A))
    if initial_marker is not None:
        scores_dir.mkdir(parents=True)
        _marker(scores_dir).write_text(f"{initial_marker}\n")

    def simulate_peer_progress(_scores_dir: Path) -> bool:
        _marker(scores_dir).parent.mkdir(parents=True, exist_ok=True)
        _marker(scores_dir).write_text(f"{expected}\n")
        _write_methods(
            scores_dir / "rank_001" / "batch_000000.parquet",
            [_AFFECTED],
        )
        return True

    monkeypatch.setattr(
        contract_module,
        "_durable_scores_contain_affected_rows",
        simulate_peer_progress,
    )

    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(current_method)],
        model=_model(_FP_A),
    )

    assert _marker(scores_dir).read_text().strip() == expected


def test_existing_affected_rows_reject_mismatch_when_current_methods_are_unaffected(
    tmp_path: Path,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    _write_methods(scores_dir / "rank_001" / "batch_000004.parquet", [_AFFECTED])
    _marker(scores_dir).write_text(f"axbench-bpb-live-ours-v1:{_FP_B}\n")

    with pytest.raises(AxBenchBpbError, match="fresh run_tag|clear the old"):
        enforce_live_ours_vocab_contract(
            scores_dir=scores_dir,
            methods=[_method(_UNAFFECTED)],
            model=_model(_FP_A),
        )


def test_exact_marker_for_current_affected_method_does_not_scan_shards(
    tmp_path: Path,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    corrupt_shard = scores_dir / "rank_000" / "batch_000000.parquet"
    corrupt_shard.parent.mkdir(parents=True)
    corrupt_shard.write_text("not parquet")
    expected = live_ours_vocab_contract_identity(_model(_FP_A))
    _marker(scores_dir).write_text(f"{expected}\n")

    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(_AFFECTED)],
        model=_model(_FP_A),
    )


def test_unreadable_method_column_fails_instead_of_guessing(
    tmp_path: Path,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    shard = scores_dir / "rank_000" / "batch_000000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.table({"not_method_name": [_AFFECTED]}), shard)

    with pytest.raises(AxBenchBpbError, match="cannot establish.*method_name"):
        enforce_live_ours_vocab_contract(
            scores_dir=scores_dir,
            methods=[_method(_AFFECTED)],
            model=_model(_FP_A),
        )


@pytest.mark.parametrize(
    "marker_text",
    [None, "malformed", f"axbench-bpb-live-ours-v1:{_FP_B}"],
)
def test_existing_affected_rows_reject_missing_malformed_or_mismatched_marker(
    tmp_path: Path,
    marker_text: str | None,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    _write_methods(scores_dir / "rank_009" / "batch_000000.parquet", [_AFFECTED])
    if marker_text is not None:
        _marker(scores_dir).write_text(f"{marker_text}\n")

    with pytest.raises(AxBenchBpbError, match="fresh run_tag|clear the old"):
        enforce_live_ours_vocab_contract(
            scores_dir=scores_dir,
            methods=[_method(_AFFECTED)],
            model=_model(_FP_A),
        )


def test_deleting_affected_shards_allows_stale_marker_replacement(
    tmp_path: Path,
) -> None:
    scores_dir = tmp_path / "bpb_scores"
    shard = scores_dir / "rank_000" / "batch_000000.parquet"
    _write_methods(shard, [_AFFECTED])
    _marker(scores_dir).write_text(f"axbench-bpb-live-ours-v1:{_FP_B}\n")
    shard.unlink()

    enforce_live_ours_vocab_contract(
        scores_dir=scores_dir,
        methods=[_method(_AFFECTED)],
        model=_model(_FP_A),
    )

    assert _marker(scores_dir).read_text().strip().endswith(_FP_A)


def test_aggregate_ignores_marker_for_unaffected_rows(tmp_path: Path) -> None:
    table = pa.table({"method_name": [_UNAFFECTED]})
    assert_aggregate_live_ours_vocab_contract(
        scores_table=table,
        scores_dir=tmp_path,
    )


def test_aggregate_requires_method_name_column(tmp_path: Path) -> None:
    with pytest.raises(AxBenchBpbError, match="missing required method_name"):
        assert_aggregate_live_ours_vocab_contract(
            scores_table=pa.table({"other": [_AFFECTED]}),
            scores_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "marker_text",
    [None, "", "malformed", "axbench-bpb-live-ours-v2:" + _FP_A],
)
def test_aggregate_rejects_affected_rows_without_well_formed_marker(
    tmp_path: Path,
    marker_text: str | None,
) -> None:
    if marker_text is not None:
        tmp_path.mkdir(parents=True, exist_ok=True)
        _marker(tmp_path).write_text(marker_text)
    table = pa.table({"method_name": [_AFFECTED]})
    with pytest.raises(AxBenchBpbError, match="fresh run_tag|score it again"):
        assert_aggregate_live_ours_vocab_contract(
            scores_table=table,
            scores_dir=tmp_path,
        )


@pytest.mark.parametrize("fingerprint", ["uncompacted", _FP_A])
def test_aggregate_accepts_supported_identity_shapes(
    tmp_path: Path,
    fingerprint: str,
) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    _marker(tmp_path).write_text(f"axbench-bpb-live-ours-v1:{fingerprint}\n")
    assert_aggregate_live_ours_vocab_contract(
        scores_table=pa.table({"method_name": [_AFFECTED]}),
        scores_dir=tmp_path,
    )


def test_marker_is_not_discovered_as_a_parquet_shard(tmp_path: Path) -> None:
    scores_dir = tmp_path / "bpb_scores"
    scores_dir.mkdir()
    _marker(scores_dir).write_text("not parquet")
    _write_methods(scores_dir / "rank_000" / "batch_000000.parquet", [_UNAFFECTED])
    assert [path.name for path in scores_dir.rglob("*.parquet")] == ["batch_000000.parquet"]
