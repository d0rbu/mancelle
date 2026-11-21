"""Scientific math and project direction artifact tests."""

from __future__ import annotations

import json
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
from safetensors import safe_open
from safetensors.torch import save_file
from sklearn.exceptions import ConvergenceWarning

from nanochat.common import COMPUTE_DTYPE
from scripts.eval.axbench_adv.pressures import Pressure
from steering.conditional.encoder_decoder.eval_residual import post_cross_attention_residual_counter
from steering.eval.axbench_adv import direction_fitting as fitting_module
from steering.eval.axbench_adv.counter import (
    ProjectCounterExpectations,
    load_counter_direction,
    load_project_gcg_activation_direction,
    preflight_project_direction_fit,
)
from steering.eval.axbench_adv.direction_fitting import (
    FIT_MANIFEST_NAME,
    SIGNED_CALIBRATION_KEY,
    artifact_metadata,
    fit_direction_pair,
    fit_recipe_identity,
    publish_fit_manifest,
    write_direction_artifact,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_base.direction_scale import require_direction_max_activation
from steering.eval.axbench_base.types import AxBenchBaseError
from steering.eval.axbench_upstream.encdec_baselines.directions import (
    FittedDirection,
    fit_diffmean_direction,
    fit_probe_direction,
)
from steering.safetensors_io import deterministic_safetensors_bytes

SHA_A = "a" * 64
SHA_B = "b" * 64


def _shared_identity() -> dict:
    return {
        "checkpoint": {
            "target_model": {
                "name": "model_001125.pt",
                "byte_length": 10,
                "sha256": SHA_A,
            },
            "companion_metadata": {
                "name": "meta_001125.json",
                "byte_length": 11,
                "sha256": SHA_B,
            },
        },
        "fit": fit_recipe_identity(layer=1, decoder_width=3, seed=42),
        "reproducibility": {"notes": "ignored by the scientific consumer"},
    }


def _fitted(family: str) -> FittedDirection:
    return FittedDirection(
        direction=t.tensor([1.0, 0.0, 0.0], dtype=t.float32),
        max_activation=2.0,
        metadata={"method": "probe" if family == "linear_probe" else "diffmean"},
    )


def _publication(tmp_path: Path):
    root = tmp_path / "directions"
    identity = _shared_identity()
    paths = [
        write_direction_artifact(
            root / family / "encdec" / "positive" / "concept_0000.safetensors",
            family=family,
            concept_id=0,
            concept="gardening",
            fitted=_fitted(family),
            shared_identity=identity,
        )
        for family in ("diffmean", "linear_probe")
    ]
    publish_fit_manifest(root, shared_identity=identity, artifacts=paths)
    return root, ProjectCounterExpectations(**_expectation_fields())


def _expectation_fields() -> dict:
    return {
        "target_model_sha256": SHA_A,
        "target_metadata_sha256": SHA_B,
        "concept_texts": {0: "gardening"},
        "decoder_width": 3,
        "layer": 1,
        "fit_seed": 42,
    }


def _rewrite_artifact(
    root: Path,
    *,
    direction: t.Tensor | None = None,
    metadata_changes: dict[str, str | None] | None = None,
) -> Path:
    path = root / "diffmean/encdec/positive/concept_0000.safetensors"
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        current_direction = handle.get_tensor("direction")
        metadata = dict(handle.metadata() or {})
    for key, value in (metadata_changes or {}).items():
        if value is None:
            metadata.pop(key, None)
        else:
            metadata[key] = value
    save_file(
        {"direction": current_direction if direction is None else direction},
        str(path),
        metadata=metadata,
    )
    return path


def test_unequal_length_math_reuses_exact_pure_fit_functions() -> None:
    positive = [t.tensor([[3.0, 0.0], [3.0, 0.0], [0.0, 3.0]]), t.tensor([[0.0, 3.0]])]
    negative = [t.tensor([[0.0, 0.0]]), t.tensor([[1.0, 1.0], [1.0, 1.0]])]
    fitted = fit_direction_pair(
        positive, negative, layer=21, seed=42, concept_id=7, surface="fixture"
    )
    expected_diffmean = fit_diffmean_direction(
        positive, negative, intervention_layer=21, surface="fixture DiffMean"
    )
    expected_probe = fit_probe_direction(
        positive,
        negative,
        intervention_layer=21,
        seed=42,
        concept_id=7,
        surface="fixture LinearProbe",
    )
    t.testing.assert_close(
        fitted["diffmean"].direction, expected_diffmean.direction, rtol=0, atol=0
    )
    t.testing.assert_close(
        fitted["linear_probe"].direction, expected_probe.direction, rtol=0, atol=0
    )
    assert fitted["diffmean"].max_activation == expected_diffmean.max_activation
    assert fitted["linear_probe"].max_activation == expected_probe.max_activation
    assert not t.equal(
        t.cat(positive).mean(0),
        t.stack([row.mean(0) for row in positive]).mean(0),
    )


def test_probe_seed_is_deterministic() -> None:
    positive = [t.tensor([[2.0, 0.0], [1.0, 0.0]]), t.tensor([[1.0, 0.2]])]
    negative = [t.tensor([[0.0, 2.0]]), t.tensor([[0.2, 1.0]])]
    first = fit_direction_pair(
        positive, negative, layer=1, seed=9, concept_id=3, surface="determinism"
    )["linear_probe"]
    second = fit_direction_pair(
        positive, negative, layer=1, seed=9, concept_id=3, surface="determinism"
    )["linear_probe"]
    assert t.equal(first.direction, second.direction)
    assert first.metadata == second.metadata


def test_probe_convergence_warning_fails_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    real_fit = fitting_module.fit_probe_direction

    def warning_fit(*args, **kwargs):
        warnings.warn("did not converge", ConvergenceWarning, stacklevel=2)
        return real_fit(*args, **kwargs)

    monkeypatch.setattr(fitting_module, "fit_probe_direction", warning_fit)
    with pytest.raises(RuntimeError, match="convergence warning"):
        fit_direction_pair(
            [t.tensor([[2.0, 0.0]]), t.tensor([[1.0, 0.1]])],
            [t.tensor([[0.0, 2.0]]), t.tensor([[0.1, 1.0]])],
            layer=1,
            seed=9,
            concept_id=3,
            surface="warning",
        )


def test_artifact_roundtrip_preflight_and_old_noise_is_accepted(tmp_path: Path) -> None:
    root, expectations = _publication(tmp_path)
    manifest_path = root / FIT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    manifest["shared_identity"].update(
        {"runtime": {"dirty_worktree": True}, "old_identity_digest": "stale"}
    )
    manifest["artifacts"][0].update({"sha256": "0" * 64, "byte_length": 0})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    preflight = preflight_project_direction_fit(root, expectations)
    assert set(preflight.artifacts) == {("diffmean", 0), ("linear_probe", 0)}
    loaded = load_counter_direction(
        pressure=Pressure.DIFFMEAN_COUNTER,
        method="SelfConditioning",
        concept_id=0,
        baseline_train=tmp_path,
        baseline_layer=28,
        project_fit=preflight,
    )
    t.testing.assert_close(loaded.direction, t.tensor([1.0, 0.0, 0.0]), rtol=0, atol=0)
    assert "fit_manifest_sha256" not in loaded.provenance


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"target_model_sha256": "9" * 64}, "target_model"),
        ({"target_metadata_sha256": "9" * 64}, "target_metadata"),
        ({"decoder_width": 4}, "fit_recipe"),
        ({"layer": 2}, "fit_recipe"),
        ({"fit_seed": 43}, "fit_recipe"),
    ],
)
def test_preflight_rejects_wrong_scientific_binding(
    tmp_path: Path, change: dict, match: str
) -> None:
    root, expected = _publication(tmp_path)
    with pytest.raises(AxBenchAdvError, match=match):
        preflight_project_direction_fit(root, replace(expected, **change))


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"decoder_width": 0}, "decoder_width"),
        ({"decoder_width": True}, "decoder_width"),
        ({"layer": -1}, "layer"),
        ({"layer": True}, "layer"),
        ({"fit_seed": -1}, "fit_seed"),
        ({"fit_seed": True}, "fit_seed"),
        ({"site": "block_output"}, "site"),
        ({"concept_texts": {}}, "concept_texts"),
        ({"concept_texts": {True: "gardening"}}, "concept_texts"),
        ({"concept_texts": {-1: "gardening"}}, "concept_texts"),
        ({"concept_texts": {0: ""}}, "concept_texts"),
        ({"concept_texts": {0: 1}}, "concept_texts"),
    ],
)
def test_project_counter_expectations_reject_invalid_scientific_coordinates(
    change: dict,
    match: str,
) -> None:
    with pytest.raises(AxBenchAdvError, match=match):
        ProjectCounterExpectations(**{**_expectation_fields(), **change})


def test_preflight_rejects_wrong_concept_and_incomplete_coverage(tmp_path: Path) -> None:
    root, expected = _publication(tmp_path)
    with pytest.raises(AxBenchAdvError, match="concept provenance"):
        preflight_project_direction_fit(root, replace(expected, concept_texts={0: "wrong concept"}))
    (root / "linear_probe/encdec/positive/concept_0000.safetensors").unlink()
    with pytest.raises(AxBenchAdvError, match="missing project fit artifact"):
        preflight_project_direction_fit(root, expected)


@pytest.mark.parametrize("mutation", ["duplicate", "missing_family"])
def test_preflight_rejects_duplicate_or_incomplete_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    root, expectations = _publication(tmp_path)
    manifest_path = root / FIT_MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text())
    if mutation == "duplicate":
        manifest["artifacts"].append(dict(manifest["artifacts"][0]))
        match = "duplicates coordinate"
    else:
        manifest["artifacts"] = [
            record for record in manifest["artifacts"] if record["family"] != "linear_probe"
        ]
        match = "coverage differs"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(AxBenchAdvError, match=match):
        preflight_project_direction_fit(root, expectations)


def test_preflight_rejects_actual_vector_width_independent_of_recipe(tmp_path: Path) -> None:
    root, expectations = _publication(tmp_path)
    _rewrite_artifact(root, direction=t.tensor([1.0, 0.0], dtype=t.float32))

    with pytest.raises(AxBenchAdvError, match="width mismatch"):
        preflight_project_direction_fit(root, expectations)


@pytest.mark.parametrize("payload", [None, "{", "[]"])
def test_preflight_rejects_absent_or_malformed_manifest(
    tmp_path: Path,
    payload: str | None,
) -> None:
    root, expectations = _publication(tmp_path)
    manifest_path = root / FIT_MANIFEST_NAME
    if payload is None:
        manifest_path.unlink()
    else:
        manifest_path.write_text(payload)

    with pytest.raises(AxBenchAdvError, match="invalid project fit manifest"):
        preflight_project_direction_fit(root, expectations)


def test_writer_and_preflight_reject_non_normalized_directions(tmp_path: Path) -> None:
    invalid = FittedDirection(
        direction=t.tensor([2.0, 0.0, 0.0]),
        max_activation=2.0,
        metadata={"method": "diffmean"},
    )
    with pytest.raises(ValueError, match="normalization bound"):
        write_direction_artifact(
            tmp_path / "writer/diffmean/encdec/positive/concept_0000.safetensors",
            family="diffmean",
            concept_id=0,
            concept="gardening",
            fitted=invalid,
            shared_identity=_shared_identity(),
        )
    root, expectations = _publication(tmp_path / "preflight")
    _rewrite_artifact(root, direction=t.tensor([2.0, 0.0, 0.0]))
    with pytest.raises(AxBenchAdvError, match="normalization bound"):
        preflight_project_direction_fit(root, expectations)


@pytest.mark.parametrize(
    ("direction", "match"),
    [
        (t.tensor([[1.0, 0.0, 0.0]], dtype=t.float32), "non-empty 1D"),
        (t.tensor([1, 0, 0], dtype=t.int64), "floating point"),
        (t.tensor([1.0, 0.0, 0.0], dtype=t.float64), "must be float32"),
        (t.tensor([float("nan"), 0.0, 0.0], dtype=t.float32), "NaN or Inf"),
    ],
)
def test_preflight_rejects_malformed_direction_tensor(
    tmp_path: Path, direction: t.Tensor, match: str
) -> None:
    root, expectations = _publication(tmp_path)
    _rewrite_artifact(root, direction=direction)
    with pytest.raises(AxBenchAdvError, match=match):
        preflight_project_direction_fit(root, expectations)


@pytest.mark.parametrize(
    ("metadata_changes", "match"),
    [
        ({"capture_site": "block_output"}, "capture site"),
        ({"provenance_json": "{"}, "provenance is malformed"),
        ({"concept_id": "1"}, "identity mismatch"),
    ],
)
def test_preflight_rejects_wrong_artifact_coordinates(
    tmp_path: Path,
    metadata_changes: dict[str, str | None],
    match: str,
) -> None:
    root, expectations = _publication(tmp_path)
    _rewrite_artifact(root, metadata_changes=metadata_changes)
    with pytest.raises(AxBenchAdvError, match=match):
        preflight_project_direction_fit(root, expectations)


def test_manifest_is_written_last_and_refuses_replacement(tmp_path: Path) -> None:
    root, _expectations = _publication(tmp_path)
    manifest = json.loads((root / FIT_MANIFEST_NAME).read_text())
    artifacts = [root / row["path"] for row in manifest["artifacts"]]
    with pytest.raises(FileExistsError, match="already exists"):
        publish_fit_manifest(root, shared_identity=manifest["shared_identity"], artifacts=artifacts)


@pytest.mark.parametrize("peak", [-1.0, 0.0])
def test_nonpositive_real_fit_preserves_counter_and_rejects_scaled_consumers(
    tmp_path: Path,
    peak: float,
) -> None:
    # A shared negative offset does not reverse the positive-minus-negative
    # contrast. Both real fits point toward the positive class, even though
    # every token projects nonpositively onto them (no mocked estimator).
    positive = [t.tensor([[peak, 0.0, 0.0], [peak - 1, 0.0, 0.0]])] * 20
    negative = [t.tensor([[peak - 3, 0.0, 0.0], [peak - 4, 0.0, 0.0]])] * 20
    fitted = fit_direction_pair(
        positive, negative, layer=1, seed=42, concept_id=0, surface="offset"
    )
    raw = t.cat(positive).mean(0) - t.cat(negative).mean(0)
    expected = raw / (raw.norm() + t.finfo(t.float32).eps)
    assert t.equal(fitted["diffmean"].direction, expected)
    for fit in fitted.values():
        assert fit.direction[0] > 0
        assert fit.max_activation == float((t.cat(positive + negative) @ fit.direction).max())
        assert fit.max_activation <= 0
    # Unchanged default producer contract rejects this exact fit population.
    for fit_function in (fit_diffmean_direction, fit_probe_direction):
        kwargs = {"seed": 42} if fit_function is fit_probe_direction else {}
        with pytest.raises(AxBenchBaseError, match="non-positive max_activation"):
            fit_function(positive, negative, intervention_layer=1, surface="scaled", **kwargs)

    identity = _shared_identity()
    root = tmp_path / "directions"
    paths = [
        write_direction_artifact(
            root / family / "encdec/positive/concept_0000.safetensors",
            family=family,
            concept_id=0,
            concept="gardening",
            fitted=fit,
            shared_identity=identity,
        )
        for family, fit in fitted.items()
    ]
    publish_fit_manifest(root, shared_identity=identity, artifacts=paths)
    preflight = preflight_project_direction_fit(
        root, ProjectCounterExpectations(**_expectation_fields())
    )
    for family, pressure in (
        ("diffmean", Pressure.DIFFMEAN_COUNTER),
        ("linear_probe", Pressure.LINEAR_PROBE_COUNTER),
    ):
        loaded = load_counter_direction(
            pressure=pressure,
            method="SelfConditioning",
            concept_id=0,
            baseline_train=tmp_path,
            baseline_layer=28,
            project_fit=preflight,
        )
        assert t.equal(loaded.direction, fitted[family].direction)
        model = SimpleNamespace(
            dec_config=SimpleNamespace(n_layer=2, n_embd=3),
            training=False,
            _post_cross_attention_eval_callback=None,
            get_device=lambda: t.device("cpu"),
        )
        residual = t.ones(1, 2, 3, dtype=COMPUTE_DTYPE)
        for magnitude in (0.25, 0.5, 1.0, 2.0):
            with post_cross_attention_residual_counter(
                model,
                layer=1,
                direction=loaded.direction,
                magnitude=magnitude,
            ):
                actual = model._post_cross_attention_eval_callback(1, residual)
                expected_residual = residual - magnitude * fitted[family].direction.to(
                    COMPUTE_DTYPE
                )
                assert t.equal(actual, expected_residual)
            assert model._post_cross_attention_eval_callback is None
        metadata = preflight.artifacts[(family, 0)].loaded.metadata_dict()
        assert "max_activation" not in metadata
        assert float(metadata[SIGNED_CALIBRATION_KEY]) == fitted[family].max_activation
        with pytest.raises(AxBenchBaseError, match="missing 'max_activation'"):
            require_direction_max_activation(
                metadata,
                direction=loaded.direction,
                expected_concept_id=0,
                expected_mode="positive",
                expected_target_model="encdec",
                expected_layer=1,
                surface="BPB",
            )
    with pytest.raises(AxBenchAdvError, match="missing 'max_activation'"):
        load_project_gcg_activation_direction(preflight, concept_id=0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_counter_fit_and_writer_reject_nonfinite(tmp_path: Path, bad: float) -> None:
    with pytest.raises(ValueError, match="non-finite activations"):
        fit_direction_pair(
            [t.tensor([[bad, 0.0, 0.0]])],
            [t.zeros(1, 3)],
            layer=1,
            seed=42,
            concept_id=0,
            surface="invalid",
        )
    with pytest.raises(AxBenchBaseError, match="non-finite max_activation"):
        write_direction_artifact(
            tmp_path / "diffmean/encdec/positive/concept_0000.safetensors",
            family="diffmean",
            concept_id=0,
            concept="gardening",
            fitted=replace(_fitted("diffmean"), max_activation=bad),
            shared_identity=_shared_identity(),
        )
    with pytest.raises(ValueError, match="finite nonempty float32 vector"):
        write_direction_artifact(
            tmp_path / "diffmean/encdec/positive/concept_0000.safetensors",
            family="diffmean",
            concept_id=0,
            concept="gardening",
            fitted=replace(_fitted("diffmean"), direction=t.tensor([bad, 0.0, 0.0])),
            shared_identity=_shared_identity(),
        )


@pytest.mark.parametrize(
    "metadata_changes",
    [
        {"max_activation": None},
        {"max_activation": "nan"},
        {"max_activation": "-1.0"},
        {"max_activation": "0.0"},
        {SIGNED_CALIBRATION_KEY: "-1.0"},  # ambiguous fields
        {"max_activation": None, SIGNED_CALIBRATION_KEY: "nan"},
        {"max_activation": None, SIGNED_CALIBRATION_KEY: "inf"},
        {"max_activation": None, SIGNED_CALIBRATION_KEY: "-inf"},
        {"max_activation": None, SIGNED_CALIBRATION_KEY: "garbage"},
        {"max_activation": None, SIGNED_CALIBRATION_KEY: "1.0"},
    ],
)
def test_counter_preflight_rejects_invalid_calibration(
    tmp_path: Path, metadata_changes: dict
) -> None:
    root, expectations = _publication(tmp_path)
    _rewrite_artifact(root, metadata_changes=metadata_changes)
    with pytest.raises(AxBenchAdvError):
        preflight_project_direction_fit(root, expectations)


def test_signed_calibration_cannot_describe_zero_vector(tmp_path: Path) -> None:
    root, expectations = _publication(tmp_path)
    _rewrite_artifact(
        root,
        direction=t.zeros(3),
        metadata_changes={
            "max_activation": None,
            SIGNED_CALIBRATION_KEY: "-1.0",
        },
    )
    with pytest.raises(AxBenchAdvError, match="requires a nonzero direction"):
        preflight_project_direction_fit(root, expectations)


@pytest.mark.parametrize("zero", [False, True])
@pytest.mark.parametrize("family", ["diffmean", "linear_probe"])
def test_positive_and_zero_publication_bytes_keep_legacy_contract(
    tmp_path: Path, zero: bool, family: str
) -> None:
    fit = _fitted(family)
    if zero:
        fit = replace(fit, direction=t.zeros(3), max_activation=0.0)
    identity = _shared_identity()
    # Independent pre-change metadata oracle, including exact JSON encoding.
    metadata = {
        "concept_id": "0",
        "layer": "1",
        "mode": "positive",
        "target_model": "encdec",
        "artifact_kind": f"{family}_direction_v1",
        "max_activation": repr(fit.max_activation),
        "capture_site": "post_cross_attention",
        "provenance_json": json.dumps(
            {
                "concept_id": 0,
                "concept": "gardening",
                "family": family,
                "fit_metadata": fit.metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    }
    assert (
        artifact_metadata(
            family=family,
            concept_id=0,
            concept="gardening",
            fitted=fit,
            shared_identity=identity,
        )
        == metadata
    )
    path = write_direction_artifact(
        tmp_path / family / "encdec/positive/concept_0000.safetensors",
        family=family,
        concept_id=0,
        concept="gardening",
        fitted=fit,
        shared_identity=identity,
    )
    assert path.read_bytes() == deterministic_safetensors_bytes(
        {"direction": fit.direction}, metadata=metadata
    )
