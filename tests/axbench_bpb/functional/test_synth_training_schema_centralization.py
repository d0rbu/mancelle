"""Lock that all axbench-bpb test sites that need positive/negative
synth-training schemas import them from the production source of truth
(axbench_base.synth.positive_mode / negative_mode), preventing the
4-way duplication this PR removed from drifting back."""
from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_TEST_DIR = _REPO_ROOT / "tests" / "axbench_bpb"


def test_no_local_pa_schema_for_training_data():
    """No test file under tests/axbench_bpb/ should redefine pa.schema()
    for the positive/negative training schemas — they live in
    axbench_base.synth.{positive,negative}_mode."""
    forbidden_patterns = [
        "_POSITIVE_SCHEMA = pa.schema",
        "_NEGATIVE_SCHEMA = pa.schema",
        "_POS_SCHEMA = pa.schema",
        "_NEG_SCHEMA = pa.schema",
        "POSITIVE_SCHEMA = pa.schema",
        "NEGATIVE_SCHEMA = pa.schema",
    ]
    self_path = Path(__file__).resolve()
    offenders = []
    for path in _TEST_DIR.rglob("*.py"):
        if path.resolve() == self_path:
            continue  # skip this meta-test, which lists the patterns as literals
        text = path.read_text()
        for pattern in forbidden_patterns:
            if pattern in text:
                offenders.append(f"{path.relative_to(_REPO_ROOT)}: {pattern!r}")
    assert not offenders, (
        "Re-introduced duplication of training-data schemas in tests:\n"
        + "\n".join(offenders)
        + "\n\nImport from steering.eval.axbench_base.synth.positive_mode "
          "(TRAINING_SCHEMA) / negative_mode (NEG_TRAINING_SCHEMA) instead."
    )
