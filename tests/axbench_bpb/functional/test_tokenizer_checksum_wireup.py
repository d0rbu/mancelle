"""Slice 17 audit (2026-05-05) P1.B regression test.

``assert_tokenizer_checksum_matches`` previously had a TODO indicating
the wire-up call site was missing. The fix wires both
``assert_tokenizer_checksum_matches`` and ``assert_bos_lengths_match``
into the CLI ``_stage_prepare_context_cache`` (executed at startup
before any feature-cache extraction).

Tests:
1. ``assert_tokenizer_checksum_matches`` correctly resolves the
   stats path inside ``feature_dir`` via ``find_statistics_path``.
2. Missing stats file raises a clear ``AxBenchBpbError``.
3. Missing checksum metadata is treated as a soft warning (matches
   reference training script behaviour).
4. Mismatched checksum raises ``AxBenchBpbError``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from safetensors.torch import save_file

from steering.eval.axbench_bpb.tokenization import (
    assert_bos_lengths_match,
    assert_tokenizer_checksum_matches,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError
from steering.stats_io import (
    SAFETENSORS_EXT,
    STATS_STEM,
    save_tokenizer_checksum,
)


def _write_stats(feature_dir: Path, *, with_checksum: str | None = None) -> Path:
    """Write a minimal stats safetensors at <feature_dir>/feature_statistics.safetensors."""
    feature_dir.mkdir(parents=True, exist_ok=True)
    stats_path = feature_dir / f"{STATS_STEM}{SAFETENSORS_EXT}"
    # Save with ONE tensor so the file is valid safetensors.
    save_file(
        {"placeholder": torch.zeros(1, dtype=torch.float32)},
        str(stats_path),
    )
    if with_checksum is not None:
        save_tokenizer_checksum(stats_path, with_checksum)
    return stats_path


def _build_mock_tokenizer(checksum_value: str = "deadbeef") -> MagicMock:
    """Build a mock tokenizer with a deterministic checksum."""
    tok = MagicMock()
    # compute_tokenizer_checksum probes via ``decode``/``encode`` — patch via
    # parametrize at call site, OR override compute_tokenizer_checksum
    # via monkeypatch. Here we use the latter pattern.
    return tok


def test_assert_tokenizer_checksum_no_stats_file_raises(tmp_path):
    """If feature_dir has no stats file, raise AxBenchBpbError."""
    fdir = tmp_path / "features"
    fdir.mkdir()
    tok = MagicMock()
    with pytest.raises(AxBenchBpbError, match="no feature_statistics"):
        assert_tokenizer_checksum_matches(tokenizer=tok, feature_dir=fdir)


def test_assert_tokenizer_checksum_missing_metadata_logs_warning(tmp_path, monkeypatch):
    """Stats file present but no checksum metadata → log warning, no raise."""
    fdir = tmp_path / "features"
    _write_stats(fdir, with_checksum=None)
    tok = MagicMock()
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.compute_tokenizer_checksum",
        lambda _t: "abc123",
    )
    # Should NOT raise.
    assert_tokenizer_checksum_matches(tokenizer=tok, feature_dir=fdir)


def test_assert_tokenizer_checksum_match_passes(tmp_path, monkeypatch):
    """Matching checksum → no raise."""
    fdir = tmp_path / "features"
    _write_stats(fdir, with_checksum="match123")
    tok = MagicMock()
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.compute_tokenizer_checksum",
        lambda _t: "match123",
    )
    assert_tokenizer_checksum_matches(tokenizer=tok, feature_dir=fdir)


def test_assert_tokenizer_checksum_mismatch_raises(tmp_path, monkeypatch):
    """Mismatched checksum → AxBenchBpbError."""
    fdir = tmp_path / "features"
    _write_stats(fdir, with_checksum="storedabc")
    tok = MagicMock()
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.compute_tokenizer_checksum",
        lambda _t: "actualxyz",
    )
    with pytest.raises(AxBenchBpbError, match="Tokenizer checksum mismatch"):
        assert_tokenizer_checksum_matches(tokenizer=tok, feature_dir=fdir)


def test_assert_bos_lengths_match_equal_lengths():
    """Equal-length BOS strings → no raise."""
    tok_a = MagicMock()
    tok_a.get_bos_token_id.return_value = 1
    tok_a.decode.return_value = "<bos>"
    tok_b = MagicMock()
    tok_b.get_bos_token_id.return_value = 2
    tok_b.decode.return_value = "<BOS>"  # same length
    # No raise.
    assert_bos_lengths_match(
        tokenizer_extraction=tok_a, tokenizer_nanochat=tok_b,
    )


def test_assert_bos_lengths_mismatch_raises():
    """Different-length BOS strings → AxBenchBpbError."""
    tok_a = MagicMock()
    tok_a.get_bos_token_id.return_value = 1
    tok_a.decode.return_value = "<bos>"
    tok_b = MagicMock()
    tok_b.get_bos_token_id.return_value = 2
    tok_b.decode.return_value = "<bigBOS>"  # longer
    with pytest.raises(AxBenchBpbError, match="BOS decoded-length mismatch"):
        assert_bos_lengths_match(
            tokenizer_extraction=tok_a, tokenizer_nanochat=tok_b,
        )


def test_axbench_bpb_eval_imports_guards():
    """The CLI module must import the guards (P1.B wire-up)."""
    import scripts.eval.axbench_bpb_eval as cli_mod

    assert hasattr(cli_mod, "assert_tokenizer_checksum_matches"), (
        "P1.B fix: CLI must import assert_tokenizer_checksum_matches"
    )
    assert hasattr(cli_mod, "assert_bos_lengths_match"), (
        "P1.B fix: CLI must import assert_bos_lengths_match"
    )
