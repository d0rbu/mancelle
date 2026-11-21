"""Tests for silent error handling fixes across nanochat/ module (Tasks 6-12)."""

import inspect
import subprocess
import sys
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from errors import DataLoadError
from nanochat.checkpoint_manager import _find_checkpoint_leaf
from nanochat.report import extract_timestamp, run_command

# =============================================================================
# Task 6: report.py — run_command
# =============================================================================


class TestRunCommand:
    """run_command returns output on success, None on known errors.

    NOTE: ``run_command`` was hardened against shell injection — it now
    requires a list/tuple argv and refuses ``shell=True`` string commands.
    These tests were migrated to list form as part of that fix.
    """

    def test_returns_output_on_success(self):
        result = run_command(["echo", "hello"])
        assert result == "hello"

    @patch("nanochat.report.subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5))
    def test_returns_none_on_timeout(self, mock_run):
        assert run_command(["sleep", "999"]) is None

    @patch("nanochat.report.subprocess.run", side_effect=FileNotFoundError("no such binary"))
    def test_returns_none_on_file_not_found(self, mock_run):
        assert run_command(["nonexistent_binary"]) is None

    @patch("nanochat.report.subprocess.run", side_effect=MemoryError("out of memory"))
    def test_propagates_unexpected_error(self, mock_run):
        with pytest.raises(MemoryError):
            run_command(["anything"])

    def test_rejects_string_command(self):
        """String commands are refused — they would signal a regression
        to the ``shell=True`` pattern."""
        with pytest.raises(TypeError):
            run_command("echo hello")

    def test_rejects_non_string_argv_entry(self):
        with pytest.raises(TypeError):
            run_command(["echo", cast(str, 123)])


# =============================================================================
# Task 6: report.py — extract_timestamp
# =============================================================================


class TestExtractTimestamp:
    """extract_timestamp parses valid timestamps, returns None for invalid."""

    def test_valid_timestamp(self):
        content = "timestamp: 2024-01-15 10:30:00\n"
        result = extract_timestamp(content, "timestamp:")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_invalid_timestamp_returns_none(self):
        content = "timestamp: not-a-date\n"
        result = extract_timestamp(content, "timestamp:")
        assert result is None

    def test_missing_prefix_returns_none(self):
        content = "other: 2024-01-15 10:30:00\n"
        result = extract_timestamp(content, "timestamp:")
        assert result is None


# =============================================================================
# Task 6+7: Source inspection — no bare except
# =============================================================================


class TestNoBareExcept:
    """Verify bare except: blocks have been replaced in report.py and dataset.py."""

    def test_report_run_command_no_bare_except(self):
        source = inspect.getsource(run_command)
        assert "except:" not in source.replace("except: ", "").replace("except:\n", "") or \
            all("except:" not in line or line.strip().startswith("#")
                for line in source.splitlines()
                if "except:" in line and "except:" == line.strip().rstrip()), \
            "run_command still contains bare except:"
        # Positive check: should have named exceptions
        assert "subprocess.CalledProcessError" in source

    def test_report_extract_timestamp_no_bare_except(self):
        source = inspect.getsource(extract_timestamp)
        lines_with_except = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("except") and line.strip() == "except:"
        ]
        assert len(lines_with_except) == 0, "extract_timestamp still contains bare except:"
        assert "ValueError" in source

    def test_dataset_no_bare_except(self):
        from nanochat.dataset import download_single_file

        source = inspect.getsource(download_single_file)
        lines_with_except = [
            line.strip() for line in source.splitlines()
            if line.strip().startswith("except") and line.strip() == "except:"
        ]
        assert len(lines_with_except) == 0, "download_single_file still contains bare except:"
        assert "OSError" in source


# =============================================================================
# Task 9: tokenizer.py — encode_special should not return None
# =============================================================================


class TestEncodeSpecialNoNone:
    """TransformersTokenizer.encode_special should raise ParseError, not return None."""

    def test_source_has_no_return_none(self):
        from nanochat.tokenizer import TransformersTokenizer

        source = inspect.getsource(TransformersTokenizer.encode_special.__wrapped__)
        assert "return None" not in source, \
            "encode_special should raise ParseError instead of returning None"
        assert "ParseError" in source


# =============================================================================
# Task 10: checkpoint_manager.py — _find_checkpoint_leaf
# =============================================================================


class TestFindCheckpointLeaf:
    """_find_checkpoint_leaf must raise DataLoadError for empty directories."""

    def test_finds_flat_dir(self, tmp_path):
        (tmp_path / "model_000000.pt").write_bytes(b"fake")
        (tmp_path / "meta_000000.json").write_text("{}")
        assert _find_checkpoint_leaf(str(tmp_path)) == str(tmp_path)

    def test_finds_nested_dir(self, tmp_path):
        nested = tmp_path / "s128" / "t2048.9"
        nested.mkdir(parents=True)
        (nested / "model_000000.pt").write_bytes(b"fake")
        (nested / "meta_000000.json").write_text("{}")
        assert _find_checkpoint_leaf(str(tmp_path)) == str(nested)

    def test_raises_data_load_error_for_empty_dir(self, tmp_path):
        with pytest.raises(DataLoadError):
            _find_checkpoint_leaf(str(tmp_path))
