from __future__ import annotations

from pathlib import Path

import pytest
import torch as t
from safetensors.torch import save as save_safetensors_bytes

import steering.eval.axbench_base.direction_artifacts as direction_artifacts
from steering.eval.axbench_base.direction_artifacts import (
    DirectionArtifactRequest,
    expected_artifact_kind,
    load_direction_artifact,
    require_direction_artifacts,
    validate_direction_tree,
)
from steering.eval.axbench_base.types import AxBenchBaseError


def _path(
    *,
    family: str = "diffmean",
    target: str = "decoder_only",
    mode: str = "positive",
    concept_id: int = 7,
) -> Path:
    return Path(f"baselines/{family}/{target}/{mode}/concept_{concept_id:04d}.safetensors")


def _payload(
    *,
    family: str = "diffmean",
    target: str = "decoder_only",
    mode: str = "positive",
    concept_id: int = 7,
    layer: str = "6",
    direction: t.Tensor | None = None,
    metadata_updates: dict[str, str | None] | None = None,
    extra_tensors: dict[str, t.Tensor] | None = None,
) -> bytes:
    metadata: dict[str, str] = {
        "concept_id": str(concept_id),
        "target_model": target,
        "mode": mode,
        "layer": layer,
    }
    for key, value in (metadata_updates or {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    tensors = {
        "direction": (
            t.tensor([1.0, -2.0, 3.0], dtype=t.float32) if direction is None else direction
        )
    }
    tensors.update(extra_tensors or {})
    return save_safetensors_bytes(tensors, metadata=metadata)


@pytest.mark.parametrize("family", ["diffmean", "linear_probe"])
@pytest.mark.parametrize("target", ["decoder_only", "encdec"])
@pytest.mark.parametrize("mode", ["positive", "negative"])
@pytest.mark.parametrize("concept_id", [0, 27])
def test_load_direction_artifact_accepts_all_coordinates_and_retained_bytes(
    family: str,
    target: str,
    mode: str,
    concept_id: int,
) -> None:
    path = _path(
        family=family,
        target=target,
        mode=mode,
        concept_id=concept_id,
    )
    extras = (
        {
            "coef": t.ones(3, dtype=t.float64),
            "intercept": t.zeros(1, dtype=t.float64),
        }
        if family == "linear_probe"
        else None
    )
    loaded = load_direction_artifact(
        _payload(
            family=family,
            target=target,
            mode=mode,
            concept_id=concept_id,
            extra_tensors=extras,
        ),
        artifact_path=path,
        request=DirectionArtifactRequest(
            family=family,
            target_model=target,
            mode=mode,
            concept_id=concept_id,
            layer=6,
        ),
        error_type=AxBenchBaseError,
    )

    assert loaded.identity.coordinate == (family, target, mode, concept_id)
    assert loaded.identity.layer == 6
    assert t.equal(loaded.direction, t.tensor([1.0, -2.0, 3.0]))


def test_cluster_stamped_artifact_requires_and_validates_kind() -> None:
    path = _path(family="linear_probe")
    fingerprint = "a" * 64
    loaded = load_direction_artifact(
        _payload(
            family="linear_probe",
            metadata_updates={
                "feature_contract_fingerprint": fingerprint,
                "source_corpus_sha256": "b" * 64,
                "artifact_kind": expected_artifact_kind("linear_probe"),
            },
        ),
        artifact_path=path,
        error_type=AxBenchBaseError,
    )
    assert loaded.identity.artifact_kind == "linear_probe_direction_v1"

    for artifact_kind in (None, "diffmean_direction_v1"):
        updates = {
            "feature_contract_fingerprint": fingerprint,
            "source_corpus_sha256": "b" * 64,
            "artifact_kind": artifact_kind,
        }
        with pytest.raises(AxBenchBaseError, match="artifact_kind"):
            load_direction_artifact(
                _payload(family="linear_probe", metadata_updates=updates),
                artifact_path=path,
                error_type=AxBenchBaseError,
            )


def test_legacy_unclustered_artifact_may_carry_correct_kind() -> None:
    loaded = load_direction_artifact(
        _payload(metadata_updates={"artifact_kind": "diffmean_direction_v1"}),
        artifact_path=_path(),
        error_type=AxBenchBaseError,
    )
    assert loaded.identity.artifact_kind == "diffmean_direction_v1"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"concept_id": None}, "missing metadata 'concept_id'"),
        ({"target_model": None}, "invalid metadata coordinates"),
        ({"mode": None}, "invalid metadata coordinates"),
        ({"layer": None}, "missing metadata 'layer'"),
        ({"concept_id": "07"}, "noncanonical concept_id"),
        ({"layer": "06"}, "noncanonical layer"),
        ({"layer": "-1"}, "noncanonical layer"),
        ({"target_model": "encdec"}, "identity mismatch"),
        ({"mode": "negative"}, "identity mismatch"),
    ],
)
def test_metadata_identity_failures_name_the_artifact(
    updates: dict[str, str | None],
    message: str,
) -> None:
    path = _path()
    with pytest.raises(AxBenchBaseError, match=message) as exc_info:
        load_direction_artifact(
            _payload(metadata_updates=updates),
            artifact_path=path,
            error_type=AxBenchBaseError,
        )
    assert str(path) in str(exc_info.value)


def test_shape_compatible_target_swap_fails_by_identity() -> None:
    path = _path(target="encdec")
    payload = _payload(target="decoder_only", direction=t.ones(768))
    with pytest.raises(AxBenchBaseError, match="identity mismatch"):
        load_direction_artifact(
            payload,
            artifact_path=path,
            error_type=AxBenchBaseError,
        )


def test_primary_layer_regression_names_artifact_and_both_layers() -> None:
    path = _path()
    with pytest.raises(AxBenchBaseError, match="layer mismatch") as exc_info:
        load_direction_artifact(
            _payload(layer="10"),
            artifact_path=path,
            request=DirectionArtifactRequest(
                family="diffmean",
                target_model="decoder_only",
                mode="positive",
                concept_id=7,
                layer=6,
            ),
            error_type=AxBenchBaseError,
        )
    message = str(exc_info.value)
    assert str(path) in message
    assert "requested/applied layer=6" in message
    assert "artifact training layer=10" in message


@pytest.mark.parametrize(
    ("direction", "message"),
    [
        (t.ones((1, 3)), "non-empty 1D"),
        (t.tensor([], dtype=t.float32), "non-empty 1D"),
        (t.ones(3, dtype=t.int64), "floating point"),
        (t.tensor([1.0, float("nan")]), "NaN or Inf"),
        (t.tensor([1.0, float("inf")]), "NaN or Inf"),
    ],
)
def test_direction_tensor_contract(direction: t.Tensor, message: str) -> None:
    with pytest.raises(AxBenchBaseError, match=message):
        load_direction_artifact(
            _payload(direction=direction),
            artifact_path=_path(),
            error_type=AxBenchBaseError,
        )


def test_missing_direction_key_and_malformed_safetensors_fail() -> None:
    payload = save_safetensors_bytes(
        {"other": t.ones(3)},
        metadata={
            "concept_id": "7",
            "target_model": "decoder_only",
            "mode": "positive",
            "layer": "6",
        },
    )
    with pytest.raises(AxBenchBaseError, match="tensor key 'direction'"):
        load_direction_artifact(
            payload,
            artifact_path=_path(),
            error_type=AxBenchBaseError,
        )
    with pytest.raises(AxBenchBaseError, match="not valid safetensors"):
        load_direction_artifact(
            b"not safetensors",
            artifact_path=_path(),
            error_type=AxBenchBaseError,
        )


def test_all_zero_direction_is_valid() -> None:
    loaded = load_direction_artifact(
        _payload(direction=t.zeros(5)),
        artifact_path=_path(),
        error_type=AxBenchBaseError,
    )
    assert not bool(loaded.direction.any())


@pytest.mark.skipif(
    not hasattr(t, "float8_e4m3fn"),
    reason="PyTorch build has no float8_e4m3fn dtype",
)
def test_float8_direction_uses_safe_finiteness_fallback() -> None:
    direction = t.tensor([1.0, -2.0], dtype=t.float8_e4m3fn)

    loaded = load_direction_artifact(
        _payload(direction=direction),
        artifact_path=_path(),
        error_type=AxBenchBaseError,
    )

    assert loaded.direction.dtype == t.float8_e4m3fn


@pytest.mark.skipif(
    not hasattr(t, "float8_e4m3fn"),
    reason="PyTorch build has no float8_e4m3fn dtype",
)
def test_float8_direction_finiteness_fallback_still_rejects_nan() -> None:
    direction = t.tensor([float("nan")], dtype=t.float8_e4m3fn)

    with pytest.raises(AxBenchBaseError, match="contains NaN or Inf"):
        load_direction_artifact(
            _payload(direction=direction),
            artifact_path=_path(),
            error_type=AxBenchBaseError,
        )


def test_direction_tree_derives_layer_and_rejects_mixed_layers() -> None:
    root = Path("baselines")
    first = _path(concept_id=1).relative_to(root).as_posix()
    second = _path(concept_id=2).relative_to(root).as_posix()
    validation = validate_direction_tree(
        root,
        retained_payloads={
            first: _payload(concept_id=1),
            second: _payload(concept_id=2),
        },
        error_type=AxBenchBaseError,
    )
    assert validation.direction_training_layer == 6
    assert len(validation.artifacts) == 2

    with pytest.raises(AxBenchBaseError, match="disagree on training layer"):
        validate_direction_tree(
            root,
            retained_payloads={
                first: _payload(concept_id=1, layer="6"),
                second: _payload(concept_id=2, layer="10"),
            },
            error_type=AxBenchBaseError,
        )


def test_directionless_tree_has_null_layer_and_runtime_requirement_rejects() -> None:
    validation = validate_direction_tree(
        Path("baselines"),
        retained_payloads={"prompting/descriptions.json": b"{}"},
        error_type=AxBenchBaseError,
    )
    assert validation.direction_training_layer is None
    with pytest.raises(AxBenchBaseError, match="direction_training_layer=null"):
        require_direction_artifacts(
            validation,
            [
                DirectionArtifactRequest(
                    family="diffmean",
                    target_model="decoder_only",
                    mode="positive",
                    concept_id=7,
                    layer=6,
                )
            ],
            surface="test runtime",
            error_type=AxBenchBaseError,
        )
    require_direction_artifacts(
        validation,
        [],
        surface="directionless control",
        error_type=AxBenchBaseError,
    )


@pytest.mark.parametrize(
    "relative",
    [
        "unknown/decoder_only/positive/concept_0007.safetensors",
        "diffmean/decoder_only/concept_0007.safetensors",
        "diffmean/decoder_only/positive/not_canonical.safetensors",
        "../diffmean/decoder_only/positive/concept_0007.safetensors",
    ],
)
def test_direction_tree_rejects_unknown_or_malformed_entries(relative: str) -> None:
    with pytest.raises(AxBenchBaseError):
        validate_direction_tree(
            Path("baselines"),
            retained_payloads={relative: _payload()},
            error_type=AxBenchBaseError,
        )


def test_path_backed_loader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.safetensors"
    target.write_bytes(_payload())
    path = tmp_path / "diffmean" / "decoder_only" / "positive" / "concept_0007.safetensors"
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    with pytest.raises(AxBenchBaseError, match="real regular file"):
        load_direction_artifact(path, error_type=AxBenchBaseError)


def test_path_backed_loader_rejects_symlinked_coordinate_directory(
    tmp_path: Path,
) -> None:
    outside_family = tmp_path / "outside" / "diffmean"
    target = outside_family / "decoder_only" / "positive" / "concept_0007.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(_payload())
    baselines = tmp_path / "baselines"
    baselines.mkdir()
    (baselines / "diffmean").symlink_to(outside_family, target_is_directory=True)
    path = baselines / "diffmean" / "decoder_only" / "positive" / target.name

    with pytest.raises(AxBenchBaseError, match="real regular file"):
        load_direction_artifact(path, error_type=AxBenchBaseError)


def test_path_backed_loader_decodes_metadata_and_tensor_from_one_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "diffmean" / "decoder_only" / "positive" / "concept_0007.safetensors"
    path.parent.mkdir(parents=True)
    original = _payload(direction=t.tensor([1.0, 2.0, 3.0]))
    replacement = _payload(direction=t.tensor([10.0, 20.0, 30.0]))
    path.write_bytes(original)
    real_metadata_reader = direction_artifacts.read_safetensors_metadata_from_bytes

    def replace_after_metadata_read(payload: bytes) -> dict[str, str] | None:
        metadata = real_metadata_reader(payload)
        path.write_bytes(replacement)
        return metadata

    monkeypatch.setattr(
        direction_artifacts,
        "read_safetensors_metadata_from_bytes",
        replace_after_metadata_read,
    )

    loaded = load_direction_artifact(path, error_type=AxBenchBaseError)

    assert path.read_bytes() == replacement
    assert t.equal(loaded.direction, t.tensor([1.0, 2.0, 3.0]))


def test_path_backed_tree_rejects_empty_unknown_coordinate_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "baselines"
    (root / "diffmean" / "wrong_target").mkdir(parents=True)

    with pytest.raises(AxBenchBaseError, match="unknown target directory"):
        validate_direction_tree(root, error_type=AxBenchBaseError)
