"""Generate the issue #1504 conditioned-SFT pipeline identity oracle."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tests.steering.test_encdec_sft_pipeline_identity import SCENARIOS, capture_scenario

ORACLE_COMMIT = "90f6014c93111004d22ae14867ce2464a2babf89"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_PERMITTED_DIRTY_PATHS = frozenset(
    {
        "docs/superpowers/evidence/pr1.md",
        "scripts/benchmark/benchmark_sft_data_pipeline.py",
        "tests/steering/test_encdec_sft_pipeline_identity.py",
        "tests/steering/goldens/gen_encdec_sft_pipeline_identity_golden.py",
        "tests/steering/goldens/encdec_sft_pipeline_identity_golden.json",
    }
)


def _guard_oracle_checkout() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    ).stdout.strip()
    if head != ORACLE_COMMIT:
        # Test/harness/docs-only descendants preserve the production oracle.
        production_diff = subprocess.run(
            [
                "git",
                "diff",
                "--quiet",
                f"{ORACLE_COMMIT}..{head}",
                "--",
                ".",
                ":!tests",
                ":!scripts/benchmark",
                ":!docs",
            ],
            check=False,
            cwd=_REPO_ROOT,
        )
        if production_diff.returncode == 1:
            raise RuntimeError(
                f"oracle requires HEAD={ORACLE_COMMIT} or no production diff from it; "
                f"got {head}"
            )
        if production_diff.returncode != 0:
            raise RuntimeError(
                f"failed to compare production files at HEAD={head} to {ORACLE_COMMIT}"
            )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    ).stdout.splitlines()
    unexpected: list[str] = []
    for line in status:
        path_text = line[3:]
        paths = [part.strip() for part in path_text.split(" -> ")]
        if any(path not in _PERMITTED_DIRTY_PATHS for path in paths):
            unexpected.append(line)
    if unexpected:
        raise RuntimeError("oracle checkout has unrelated changes:\n" + "\n".join(unexpected))


def build_capture() -> dict[str, list[str]]:
    _guard_oracle_checkout()
    with tempfile.TemporaryDirectory(prefix="encdec-sft-pipeline-golden-") as tmp:
        root = Path(tmp)
        return {name: capture_scenario(spec, root / name) for name, spec in SCENARIOS.items()}


def main() -> None:
    print(json.dumps(build_capture(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
