"""Structural checks for runs/gemma3_l22_f65k_pipeline.sh (no execution)."""

import subprocess
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "runs"
    / "gemma3_l22_f65k_pipeline.sh"
)


def test_script_is_valid_bash():
    r = subprocess.run(
        ["bash", "-n", str(SCRIPT)], capture_output=True, text=True, check=False
    )
    assert r.returncode == 0, r.stderr


def test_step0b_uses_validate_features_not_heredoc():
    text = SCRIPT.read_text()
    assert "validate_features_gemma3_l22_f65k" in text
    assert "python -m steering.validate_features" in text
    # The old hand-rolled heredoc must be gone.
    assert "PYEOF" not in text
    assert "read_shard_metadata" not in text


def test_pipeline_invariants_intact():
    text = SCRIPT.read_text()
    assert "set -euo pipefail" in text
    assert "check_disk" in text
    for cfg in (
        "extract_features_gemma3_l22_f65k",
        "compute_statistics_gemma3_l22_f65k",
        "precompute_percentiles_gemma3_l22_f65k",
        "precompute_token_boundaries_gemma3_l22_f65k",
        "precompute_packed_rows_gemma3_l22_f65k",
    ):
        assert cfg in text, f"missing stage config {cfg}"
    # Step 0b must still run a disk check after the validate invocation
    # (parity with every other step).
    after_step0b = text.split("validate_features_gemma3_l22_f65k", 1)[1]
    assert "check_disk" in after_step0b, (
        "check_disk must follow the Step 0b validate_features invocation"
    )
