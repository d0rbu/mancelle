from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import steering.eval.axbench_bpb_adv.cluster_preflight as cluster_preflight
from steering.eval.axbench_bpb_adv.cluster_preflight import (
    build_cluster_contract_request,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError

_MASK = {"mode": "whitelist", "path": "/tmp/mask.txt"}
_CLUSTERS = {"path": "/tmp/clusters.yaml"}


def _config(*, step: Any = None, **overrides: Any) -> AxBenchBpbAdvConfig:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/feature_statistics.safetensors",
        feature_mask=dict(_MASK),
        feature_clusters=dict(_CLUSTERS),
        base_methods=["feature_diffmean_encdec", "ours_gcg_encdec"],
        selection_splice_batch_size=64,
        selection_manifest_path="/tmp/selection/manifest.json",
        encdec_target={
            "source": "local",
            "model_tag": "clustered-model",
            "step": step,
        },
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.mark.parametrize(
    ("raw_step", "expected_step"),
    [
        (None, None),
        ("last", None),
        (0, 0),
        (7, 7),
    ],
)
def test_cluster_preflight_accepts_supported_checkpoint_steps(
    raw_step: object,
    expected_step: int | None,
) -> None:
    request = build_cluster_contract_request(_config(step=raw_step))

    assert request is not None
    assert request.checkpoint_step == expected_step
    if raw_step in (0, 7):
        assert request.checkpoint_step == raw_step
        assert type(request.checkpoint_step) is int
    assert request.checkpoint_source == "local"
    assert request.checkpoint_model_tag == "clustered-model"
    assert request.checkpoint_tag == "clustered-model"
    assert request.selection_splice_batch_size == 64
    assert request.selection_manifest_path == "/tmp/selection/manifest.json"
    assert request.enabled_methods == (
        "feature_diffmean_encdec",
        "ours_gcg_encdec",
    )
    assert request.expected_feature_contract_path == (
        Path("/tmp/source") / "feature_contract.json"
    )


def test_cluster_preflight_propagates_forced_shard_revalidation() -> None:
    request = build_cluster_contract_request(
        _config(force_cluster_shard_revalidation=True)
    )

    assert request is not None
    assert request.force_cluster_shard_revalidation is True


@pytest.mark.parametrize("raw_step", [True, 3.0, "7", -1])
def test_cluster_preflight_rejects_unsupported_checkpoint_steps(
    raw_step: object,
) -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="step must be",
    ):
        build_cluster_contract_request(_config(step=raw_step))


@pytest.mark.parametrize(
    ("missing", "encdec_target"),
    [
        (
            "source",
            {"model_tag": "clustered-model", "step": "last"},
        ),
        (
            "model_tag",
            {"source": "local", "step": "last"},
        ),
    ],
)
def test_cluster_preflight_requires_checkpoint_source_and_model_tag(
    missing: str,
    encdec_target: dict[str, object],
) -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match=rf"encdec_target\.{missing} must be a non-empty string",
    ):
        build_cluster_contract_request(
            _config(encdec_target=encdec_target)
        )


def test_cluster_preflight_rejects_clusters_without_mask() -> None:
    with pytest.raises(
        AxBenchBpbAdvError,
        match="requires feature_mask with feature_clusters",
    ):
        build_cluster_contract_request(_config(feature_mask=None))


def test_cluster_preflight_disabled_path_does_not_touch_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_path(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("Path construction would touch the filesystem seam")

    monkeypatch.setattr(cluster_preflight, "Path", forbidden_path)
    cfg = _config(
        feature_mask=None,
        feature_clusters=None,
        encdec_target={},
    )

    assert build_cluster_contract_request(cfg) is None


def test_cluster_preflight_forwards_null_selection_coordinates_exactly() -> None:
    request = build_cluster_contract_request(
        _config(
            base_methods=None,
            selection_splice_batch_size=None,
            selection_manifest_path=None,
        )
    )

    assert request is not None
    assert request.selection_splice_batch_size is None
    assert request.selection_manifest_path is None
    assert request.enabled_methods is None
