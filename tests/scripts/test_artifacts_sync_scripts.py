"""CLI parsing tests for runs/upload_artifacts.sh and runs/download_artifacts.sh.

Both scripts support a `--list` short-circuit that prints the resolved
category set and exits without invoking the Python sync. Tests use it to
verify selection logic without paying for a real Hydra/HF round trip.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UPLOAD = REPO_ROOT / "runs" / "upload_artifacts.sh"
DOWNLOAD = REPO_ROOT / "runs" / "download_artifacts.sh"


def _list(script: Path, *args: str, env_extra: dict[str, str] | None = None) -> str:
    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(
        ["bash", str(script), "--list", *args],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env, check=True,
    )
    return result.stdout


def _run_expecting_failure(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(script), *args],
        capture_output=True, text=True, cwd=REPO_ROOT, check=False,
    )


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_default_set_excludes_features(script):
    """Features are opt-in — they're huge and Claudio-style operators don't need them."""
    out = _list(script)
    assert "base_checkpoints" in out
    assert "sft_checkpoints" in out
    assert "encdec_checkpoints" in out
    assert "features" not in out


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_all_flag_includes_features(script):
    out = _list(script, "--all")
    for cat in ("base_checkpoints", "sft_checkpoints", "encdec_checkpoints", "features"):
        assert cat in out, f"--all must include {cat}, got: {out!r}"


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_explicit_single_category(script):
    out = _list(script, "features")
    assert "features" in out
    assert "base_checkpoints" not in out
    assert "sft_checkpoints" not in out
    assert "encdec_checkpoints" not in out


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_explicit_multiple_categories_preserves_order(script):
    out = _list(script, "encdec_checkpoints", "features")
    assert "encdec_checkpoints" in out
    assert "features" in out
    assert "base_checkpoints" not in out
    # Order matters in the printed list.
    assert out.index("encdec_checkpoints") < out.index("features")


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_unknown_category_fails_loudly(script):
    result = _run_expecting_failure(script, "definitely_not_a_real_category")
    assert result.returncode != 0
    assert "Unknown category" in result.stderr
    # Error should list valid categories so the operator can fix the typo.
    assert "base_checkpoints" in result.stderr


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_unknown_flag_fails_loudly(script):
    result = _run_expecting_failure(script, "--bogus-flag")
    assert result.returncode != 0
    assert "Unknown flag" in result.stderr


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_help_flag_shows_usage_and_exits_zero(script):
    result = subprocess.run(
        ["bash", str(script), "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    )
    assert "Usage" in result.stdout or "usage" in result.stdout


@pytest.mark.parametrize("script", [UPLOAD, DOWNLOAD], ids=["upload", "download"])
def test_categories_can_be_mixed_with_flags_in_any_order(script):
    """`script features --dry-run` and `script --dry-run features` should both list `features`."""
    out_a = _list(script, "features", "--dry-run")
    out_b = _list(script, "--dry-run", "features")
    assert "features" in out_a
    assert "features" in out_b
    # And neither pulls in the default set.
    assert "base_checkpoints" not in out_a
    assert "base_checkpoints" not in out_b


# ---- upload-only ----

def test_upload_delete_flag_recognized():
    """--delete should be parseable without erroring (it's the destructive one)."""
    out = _list(UPLOAD, "--delete", "base_checkpoints")
    assert "base_checkpoints" in out


# ---- download-only ----

def test_download_dry_run_env_var_still_works():
    """Backward compat: DRY_RUN=true env var still recognized on download side."""
    out = _list(DOWNLOAD, env_extra={"DRY_RUN": "true"})
    # --list short-circuits before sync; just verify the script accepted the env.
    assert "base_checkpoints" in out
