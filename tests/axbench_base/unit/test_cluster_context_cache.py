"""Tests for AxBench-Base's adapter to main's context-cache metadata."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import pytest

from steering.eval.axbench_base import cluster_context_cache as cache_adapter


def test_context_cache_adapter_adds_only_feature_contract_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    def _main_metadata(**kwargs: Any) -> dict[str, Any]:
        captured.append(kwargs)
        return {
            "feature_mask_canonical": {"mode": "whitelist"},
            "cluster_selection_semantics": "main-v2",
            "stats_source_sha256": "9" * 64,
        }

    monkeypatch.setattr(
        cache_adapter,
        "clustered_context_cache_metadata",
        _main_metadata,
    )
    feature_mask = object()
    consolidation = object()
    feature_vocab = object()
    contract = SimpleNamespace(
        feature_mask=feature_mask,
        consolidation=consolidation,
        statistics=SimpleNamespace(file_sha256="9" * 64),
        selection=SimpleNamespace(preselection_seed=42),
        checkpoint=SimpleNamespace(
            resolved_tag="encdec-d12",
            step=7,
            model_file=SimpleNamespace(sha256="a" * 64),
            metadata_file=SimpleNamespace(sha256="b" * 64),
            feature_vocab=feature_vocab,
        ),
        feature_contract_fingerprint="c" * 64,
    )

    metadata = cache_adapter.clustered_context_cache_metadata_for_contract(
        cast(Any, contract)
    )

    assert len(captured) == 1
    assert captured[0]["feature_mask"] is feature_mask
    assert captured[0]["feature_clusters"] is consolidation
    assert captured[0]["feature_vocab"] is feature_vocab
    assert captured[0]["stats_source_sha256"] == "9" * 64
    provenance = captured[0]["checkpoint_seed_provenance"]
    assert provenance.metadata_dict() == {
        "checkpoint_resolved_tag": "encdec-d12",
        "checkpoint_step": 7,
        "checkpoint_model_sha256": "a" * 64,
        "checkpoint_metadata_sha256": "b" * 64,
        "checkpoint_preselection_seed": 42,
    }
    assert metadata == {
        "feature_mask_canonical": {"mode": "whitelist"},
        "cluster_selection_semantics": "main-v2",
        "stats_source_sha256": "9" * 64,
        "feature_contract_fingerprint": "c" * 64,
    }
    assert "checkpoint_selection_hash" not in metadata
