"""Import and lightweight-wiring regressions for audit PR3 (GA-013..015)."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch as t

import steering.gradient_attribution as ga
from errors import ModelWeightLoadError
from steering.sae_resolver import SAERelease, coerce_config_l0

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCER_PATH = REPO_ROOT / "scripts/preprocess/precompute_gradient_attribution.py"
CORE_PATH = REPO_ROOT / "steering/gradient_attribution.py"


def _function_local_imports(path: Path) -> list[ast.Import | ast.ImportFrom]:
    tree = ast.parse(path.read_text())
    found: list[ast.Import | ast.ImportFrom] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        found.extend(
            child
            for child in ast.walk(node)
            if isinstance(child, (ast.Import, ast.ImportFrom))
        )
    return found


def test_audited_gradient_files_have_only_module_scope_imports():
    offenders = []
    for path in (PRODUCER_PATH, CORE_PATH):
        offenders.extend(
            f"{path.relative_to(REPO_ROOT)}:{node.lineno}"
            for node in _function_local_imports(path)
        )
    assert offenders == []


def test_fresh_producer_import_is_lightweight_and_does_not_patch_backward():
    script = """
import json
import sys
import torch

before = torch.Tensor.backward
import scripts.preprocess.precompute_gradient_attribution as producer
payload = {
    "backward_unchanged": before is torch.Tensor.backward,
    "extract_features": "steering.extract_features" in sys.modules,
    "nnterp": "nnterp" in sys.modules,
    "nnsight": "nnsight" in sys.modules,
    "hist_is_list": isinstance(producer._live_cuda_hist(), list),
}
print(json.dumps(payload, sort_keys=True))
"""
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    assert payload == {
        "backward_unchanged": True,
        "extract_features": False,
        "hist_is_list": True,
        "nnterp": False,
        "nnsight": False,
    }


@pytest.mark.parametrize(
    ("release", "raw", "expected"),
    [
        ("gemmascope", 176, 176),
        ("gemmascope", None, None),
        ("gemmascope_2", "medium", "medium"),
    ],
)
def test_public_config_l0_coercion_is_lightweight(release, raw, expected):
    got = coerce_config_l0(SAERelease(release), raw)
    assert getattr(got, "value", got) == expected


def test_gradient_core_does_not_runtime_import_documented_loader_error():
    tree = ast.parse(CORE_PATH.read_text())
    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "errors"
        for alias in node.names
    }
    assert "ModelWeightLoadError" not in imported


def test_build_gradient_model_propagates_loader_error_without_local_binding(
    monkeypatch,
):
    def fail_loader(*_args, **_kwargs):
        raise ModelWeightLoadError("fixture weight failure")

    monkeypatch.setattr(ga, "load_gemma3_text_decoder", fail_loader)
    with pytest.raises(ModelWeightLoadError, match="fixture weight failure"):
        ga.build_gradient_model("gemma-3-4b-pt", "cpu", t.float32)
