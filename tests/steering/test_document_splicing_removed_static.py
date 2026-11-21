from __future__ import annotations

import importlib.util
from pathlib import Path

REPO_ROOT = Path(__file__).parents[2]

REMOVED_DOCUMENT_SPLICING_FILES = [
    "configs/data/splice_documents.yaml",
    "scripts/preprocess/splice_documents.py",
    "scripts/benchmark/benchmark_splice_documents.py",
    "scripts/benchmark/benchmark_splice_stability.py",
    "steering/splice.py",
    "docs/pipelines/document-splicing.md",
    "tests/reference_splice.py",
    "tests/scripts/test_benchmark_splice_documents_smoke.py",
    "tests/scripts/test_benchmark_splice_stability.py",
    "tests/scripts/test_splice_documents_e2e.py",
    "tests/scripts/test_splice_documents_integration.py",
    "tests/steering/test_splice_property.py",
    "tests/steering/test_splice_shard.py",
    "tests/steering/test_splice_swap_properties.py",
    "tests/steering/test_splice_tokenizer_helpers.py",
    "tests/steering/test_splice_unit.py",
    "tests/steering/test_splice_windowed_stability.py",
]

FORBIDDEN_DOCUMENT_SPLICING_REFERENCES = [
    "document-splicing",
    "splice_documents",
    "_splice_manifest",
    "out/data/spliced",
    # Keep these legacy benchmark names precise: the supported post-extraction
    # feature-shard splice benchmark intentionally remains in active surfaces.
    "benchmark_splice_documents",
    "benchmark_splice_stability",
    "steering.splice",
    "Spliced corpus",
    "Splice junction",
    "Round-trip-stable splice",
]


def test_deleted_document_splicing_files_do_not_exist() -> None:
    for rel in REMOVED_DOCUMENT_SPLICING_FILES:
        assert not (REPO_ROOT / rel).exists(), rel


def test_document_splicing_references_absent_from_active_surfaces() -> None:
    allowed = {"tests/steering/test_document_splicing_removed_static.py"}
    scan_roots = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs",
        REPO_ROOT / "configs",
        REPO_ROOT / "scripts",
        REPO_ROOT / "steering",
        REPO_ROOT / "tests",
        REPO_ROOT / "runs",
    ]
    offenders: list[str] = []

    for root in scan_roots:
        paths = [root] if root.is_file() else list(root.rglob("*"))
        for path in paths:
            if not path.is_file():
                continue
            rel = path.relative_to(REPO_ROOT).as_posix()
            if rel in allowed:
                continue
            if any(part in path.parts for part in (".venv", "__pycache__")):
                continue
            text = path.read_text(errors="ignore")
            for forbidden in FORBIDDEN_DOCUMENT_SPLICING_REFERENCES:
                if forbidden in text:
                    offenders.append(f"{rel}: {forbidden}")

    assert offenders == []


def test_splice_only_tokenizer_helpers_are_not_exported() -> None:
    helper_path = REPO_ROOT / "tests" / "cross_tokenizer_helpers.py"
    spec = importlib.util.spec_from_file_location(
        "cross_tokenizer_helpers_static",
        helper_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "SimpleTokenizer")
    assert not hasattr(module, "BosSimpleTokenizer")
    assert not hasattr(module, "MergeTokenizer")
