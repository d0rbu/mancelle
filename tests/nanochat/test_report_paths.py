"""Tests for report path construction with checkpoint_dir parameter."""

import os
from pathlib import Path
from unittest.mock import patch


from nanochat.report import Report, DummyReport, get_report


class TestGetReport:
    """Test get_report() routes reports to the correct directory."""

    def test_with_checkpoint_dir(self, tmp_path: Path) -> None:
        """get_report(checkpoint_dir) creates report under checkpoint_dir/report/."""
        ckpt_dir = str(tmp_path / "base_checkpoints" / "d26")
        os.makedirs(ckpt_dir)
        with patch("nanochat.report.get_dist_info", return_value=(False, 0, 0, 1)):
            report = get_report(checkpoint_dir=ckpt_dir)
        assert isinstance(report, Report)
        assert report.report_dir == os.path.join(ckpt_dir, "report")

    def test_without_checkpoint_dir_falls_back(self, tmp_path: Path) -> None:
        """get_report() without checkpoint_dir uses {base_dir}/report/."""
        with patch.dict(os.environ, {"NANOCHAT_BASE_DIR": str(tmp_path)}):
            with patch("nanochat.report.get_dist_info", return_value=(False, 0, 0, 1)):
                report = get_report()
        assert isinstance(report, Report)
        assert report.report_dir == os.path.join(str(tmp_path), "report")

    def test_non_rank_zero_returns_dummy(self, tmp_path: Path) -> None:
        """Non-rank-0 process gets DummyReport regardless of checkpoint_dir."""
        with patch("nanochat.report.get_dist_info", return_value=(True, 1, 1, 2)):
            report = get_report(checkpoint_dir=str(tmp_path))
        assert isinstance(report, DummyReport)


class TestReportLog:
    """Test Report.log() writes to the correct nested path."""

    def test_log_creates_file_in_report_dir(self, tmp_path: Path) -> None:
        ckpt_dir = str(tmp_path / "base_checkpoints" / "d26")
        report_dir = os.path.join(ckpt_dir, "report")
        report = Report(report_dir)
        report.log("Base model training", [{"loss": 1.5}])
        expected_file = os.path.join(report_dir, "base-model-training.md")
        assert os.path.exists(expected_file)
        content = open(expected_file).read()
        assert "## Base model training" in content
        assert "loss: 1.5" in content

    def test_log_sft_in_sft_checkpoint_dir(self, tmp_path: Path) -> None:
        sft_dir = str(
            tmp_path / "base_checkpoints" / "d26" / "standard_sft_checkpoints"
        )
        report_dir = os.path.join(sft_dir, "report")
        report = Report(report_dir)
        report.log("SFT", [{"min_val_bpb": 2.1}])
        expected_file = os.path.join(report_dir, "sft.md")
        assert os.path.exists(expected_file)
        assert not os.path.exists(os.path.join(str(tmp_path), "report", "sft.md"))


class TestReportReset:
    def test_reset_writes_header(self, tmp_path: Path) -> None:
        report_dir = str(tmp_path / "report")
        report = Report(report_dir)
        report.reset()
        header_file = os.path.join(report_dir, "header.md")
        assert os.path.exists(header_file)
        content = open(header_file).read()
        assert "# nanochat training report" in content
        assert "Run started:" in content


class TestReportGenerate:
    def test_generate_creates_report_in_checkpoint_dir(self, tmp_path: Path) -> None:
        report_dir = str(tmp_path / "base_checkpoints" / "d26" / "report")
        report = Report(report_dir)
        report.reset()
        report.log("Base model training", [{"loss": 1.5}])
        report.generate()
        report_file = os.path.join(report_dir, "report.md")
        assert os.path.exists(report_file)
        content = open(report_file).read()
        assert "Base model training" in content
